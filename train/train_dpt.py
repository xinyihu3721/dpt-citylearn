"""
Gate 4 Step 3: DPT training loop. AdamW + linear warmup (then constant lr), gradient clipping,
resumable checkpointing (wall-time limits will interrupt long runs), yaml-driven config.

Loss = mean over the 9 action dims of cross-entropy(bin_logits_dim, binned_label_dim).
Logs train loss and per-dim bin accuracy.
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from model.dpt import DPT
from model.discretize import ActionDiscretizer
from data.dpt_dataset import DPTDataset, collate_dpt

CHESCA_REPO = os.path.join(PROJECT_ROOT, 'oracle', 'chesca_repo')
SCHEMA = os.path.join('data', 'schemas', 'citylearn_challenge_2023_phase_2_local_evaluation', 'schema.json')


def get_action_bounds():
    """Construct the env briefly (no simulation loop) just to read the ACTUAL action_space
    low/high -- do NOT assume [-1,1]."""
    cwd_before = os.getcwd()
    os.chdir(CHESCA_REPO)
    sys.path.insert(0, CHESCA_REPO)
    from citylearn.citylearn import CityLearnEnv
    from rewards.user_reward import SubmissionReward
    env = CityLearnEnv(SCHEMA, reward_function=SubmissionReward)
    low, high = env.action_space[0].low.copy(), env.action_space[0].high.copy()
    os.chdir(cwd_before)
    return low, high


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_dataloader(cfg):
    ds = DPTDataset(
        seeds=cfg['data']['seeds'],
        h_max=cfg['data']['h_max'],
        epoch_size=cfg['data']['epoch_size'],
        seed_rng=cfg['seed'],
    )
    loader = DataLoader(
        ds, batch_size=cfg['train']['batch_size'], shuffle=False, num_workers=0,
        collate_fn=lambda b: collate_dpt(b, h_max=cfg['data']['h_max']),
    )
    return ds, loader


def lr_lambda_factory(warmup_steps):
    def lr_lambda(step):
        return min(1.0, (step + 1) / warmup_steps)
    return lr_lambda


def compute_loss_and_acc(bin_logits, target_bins):
    """bin_logits: list of (B, n_bins) tensors, one per action dim.
    target_bins: (B, action_dim) long tensor."""
    losses = []
    accs = []
    for d, logits in enumerate(bin_logits):
        target_d = target_bins[:, d]
        loss_d = F.cross_entropy(logits, target_d)
        acc_d = (logits.argmax(dim=-1) == target_d).float().mean()
        losses.append(loss_d)
        accs.append(acc_d)
    total_loss = torch.stack(losses).mean()
    return total_loss, torch.stack(losses), torch.stack(accs)


def save_checkpoint(path, model, optimizer, scheduler, global_step, cfg):
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'global_step': global_step,
        'config': cfg,
    }, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=os.path.join(PROJECT_ROOT, 'configs', 'train_dpt.yaml'))
    args = parser.parse_args()

    cfg = load_config(args.config)
    torch.manual_seed(cfg['seed'])

    device_str = cfg['train']['device']
    if device_str == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError("config requests device=cuda but torch.cuda.is_available() is False")
    device = torch.device(device_str)

    low, high = get_action_bounds()
    print("Action bounds used for discretization (NOT assumed [-1,1]):")
    for i in range(len(low)):
        print(f"  dim {i}: low={low[i]:.7f} high={high[i]:.7f}")
    discretizer = ActionDiscretizer(low, high, cfg['model']['n_bins'])

    ds, loader = build_dataloader(cfg)
    print(f"Context pool sizes per seed: {[ds.context_pool[s]['obs'].shape[0] for s in ds.seeds]}")
    print(f"Label pool sizes per seed: {[ds.label_pool[s]['obs'].shape[0] for s in ds.seeds]}")

    model = DPT(
        d_model=cfg['model']['d_model'],
        n_layers=cfg['model']['n_layers'],
        n_heads=cfg['model']['n_heads'],
        dropout=cfg['model']['dropout'],
        n_bins=cfg['model']['n_bins'],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['train']['lr'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lr_lambda_factory(cfg['train']['warmup_steps'])
    )

    os.makedirs(cfg['train']['checkpoint_dir'], exist_ok=True)
    latest_path = os.path.join(cfg['train']['checkpoint_dir'], 'latest.pt')

    global_step = 0
    if os.path.exists(latest_path):
        ckpt = torch.load(latest_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        global_step = ckpt['global_step']
        print(f"RESUMED from checkpoint at global_step={global_step}")
    else:
        print("No checkpoint found, starting fresh from global_step=0")

    total_steps = cfg['train']['total_steps']
    grad_clip = cfg['train']['grad_clip']
    log_every = cfg['train']['log_every']
    checkpoint_every = cfg['train']['checkpoint_every']
    action_dim = discretizer.action_dim

    model.train()
    data_iter = iter(loader)
    while global_step < total_steps:
        batch = next(data_iter)
        context_obs = batch['context_obs'].to(device)
        context_action = batch['context_action'].to(device)
        context_reward = batch['context_reward'].to(device)
        context_next_obs = batch['context_next_obs'].to(device)
        context_mask = batch['context_mask'].to(device)
        query_obs = batch['query_obs'].to(device)
        action_label = batch['action_label'].cpu().numpy()

        target_bins = torch.from_numpy(discretizer.label_to_bin(action_label)).long().to(device)

        bin_logits = model(context_obs, context_action, context_reward, context_next_obs,
                            context_mask, query_obs)
        loss, per_dim_loss, per_dim_acc = compute_loss_and_acc(bin_logits, target_bins)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        global_step += 1

        if global_step % log_every == 0 or global_step == 1:
            lr_now = scheduler.get_last_lr()[0]
            print(f"step {global_step}/{total_steps} loss={loss.item():.4f} "
                  f"mean_acc={per_dim_acc.mean().item():.4f} lr={lr_now:.6f} "
                  f"per_dim_acc={[round(a, 3) for a in per_dim_acc.tolist()]}")

        if global_step % checkpoint_every == 0 or global_step == total_steps:
            step_path = os.path.join(cfg['train']['checkpoint_dir'], f'step_{global_step}.pt')
            save_checkpoint(step_path, model, optimizer, scheduler, global_step, cfg)
            save_checkpoint(latest_path, model, optimizer, scheduler, global_step, cfg)
            print(f"saved checkpoint: {step_path}")

    # final report using a fresh evaluation batch (larger, different rng) for a cleaner readout
    model.eval()
    eval_ds = DPTDataset(seeds=cfg['data']['seeds'], h_max=cfg['data']['h_max'],
                          epoch_size=512, seed_rng=cfg['seed'] + 999)
    eval_loader = DataLoader(eval_ds, batch_size=256, shuffle=False, num_workers=0,
                              collate_fn=lambda b: collate_dpt(b, h_max=cfg['data']['h_max']))
    with torch.no_grad():
        eval_batch = next(iter(eval_loader))
        context_obs = eval_batch['context_obs'].to(device)
        context_action = eval_batch['context_action'].to(device)
        context_reward = eval_batch['context_reward'].to(device)
        context_next_obs = eval_batch['context_next_obs'].to(device)
        context_mask = eval_batch['context_mask'].to(device)
        query_obs = eval_batch['query_obs'].to(device)
        action_label = eval_batch['action_label'].cpu().numpy()
        target_bins = torch.from_numpy(discretizer.label_to_bin(action_label)).long().to(device)
        bin_logits = model(context_obs, context_action, context_reward, context_next_obs,
                            context_mask, query_obs)
        final_loss, final_per_dim_loss, final_per_dim_acc = compute_loss_and_acc(bin_logits, target_bins)

    print("\n=== FINAL EVAL (fresh 256-example batch, different rng) ===")
    print(f"final_loss={final_loss.item():.4f}")
    print(f"final_per_dim_acc={final_per_dim_acc.tolist()}")
    print(f"final_mean_acc={final_per_dim_acc.mean().item():.4f}")


if __name__ == '__main__':
    main()
