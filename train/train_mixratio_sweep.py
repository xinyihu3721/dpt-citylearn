"""
Part 2, step 3: re-run the mix-ratio frontier (exploratory% in {60,50,40,25}) on the HARD
(anchor-family x capacity) task distribution. All 4 ratios retrained fresh this round (the task
distribution changed, so r3's old checkpoint doesn't carry over). Reports, on the HELD-OUT
(anchor C x unseen capacity 1.075) task: h=0 self-play score, best self-play score over h, the
in-context delta (h=0 - best), and training accuracy. training longer this round: total_steps=5000
(up from 4000) given the harder, more numerous training tasks.

Resumable per-ratio (training checkpoints + eval results both checkpoint incrementally).
"""
import copy
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.use_deterministic_algorithms(True)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from model.dpt import DPT
from model.discretize import ActionDiscretizer
from data.dpt_dataset import collate_dpt
from data.dpt_dataset_hard import HardTaskDPTDataset
from envs.combined_task_sampler import TRAIN_TASKS, HELD_OUT_TASK, build_env

CHESCA_REPO = os.path.join(PROJECT_ROOT, 'oracle', 'chesca_repo')
NORMALIZER_PATH = os.path.join(PROJECT_ROOT, 'data', 'normalizer_hard.npz')
CHECKPOINT_BASE_DIR = os.path.join(PROJECT_ROOT, 'checkpoints_hard_sweep')
RESULTS_BASE_PATH = os.path.join(PROJECT_ROOT, 'results', 'gate6_results')

RATIOS = {
    'r1': {'exploratory': 0.60, 'chesca_noisy': 0.20, 'selfplay': 0.20},
    'r2': {'exploratory': 0.50, 'chesca_noisy': 0.25, 'selfplay': 0.25},
    'r3': {'exploratory': 0.40, 'chesca_noisy': 0.30, 'selfplay': 0.30},
    'r4': {'exploratory': 0.25, 'chesca_noisy': 0.375, 'selfplay': 0.375},
}
RATIO_EXPL_PCT = {'r1': 60, 'r2': 50, 'r3': 40, 'r4': 25}

H_SWEEP = [0, 24, 72, 168, 256]
EVAL_SEEDS = [55555, 1020, 1025]  # confirmed nonzero outages, phase_3_1 buildings 4/5/6

BASE_HPARAMS = dict(
    seed=0, h_max=256, epoch_size=2_000_000, d_model=256, n_layers=4, n_heads=8, dropout=0.1,
    n_bins=21, batch_size=64, lr=3.0e-4, warmup_steps=200, total_steps=5000, grad_clip=1.0,
    checkpoint_every=500, log_every=20,
)


def setup_paths():
    os.chdir(CHESCA_REPO)
    sys.path.insert(0, CHESCA_REPO)


def get_action_bounds():
    from rewards.user_reward import SubmissionReward
    label, rel_path, subset, m = TRAIN_TASKS[0]
    env = build_env(rel_path, subset, m, SubmissionReward)
    return env.action_space[0].low.copy(), env.action_space[0].high.copy()


def lr_lambda_factory(warmup_steps):
    def lr_lambda(step):
        return min(1.0, (step + 1) / warmup_steps)
    return lr_lambda


def compute_loss_and_acc(bin_logits, target_bins):
    losses, accs = [], []
    for d, logits in enumerate(bin_logits):
        target_d = target_bins[:, d]
        losses.append(F.cross_entropy(logits, target_d))
        accs.append((logits.argmax(dim=-1) == target_d).float().mean())
    return torch.stack(losses).mean(), torch.stack(losses), torch.stack(accs)


def save_checkpoint(path, model, optimizer, scheduler, global_step, hp):
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(), 'global_step': global_step, 'hparams': hp}, path)


def train_one_ratio(ratio_name, mix_ratios, device, discretizer):
    hp = dict(BASE_HPARAMS)
    checkpoint_dir = os.path.join(CHECKPOINT_BASE_DIR, ratio_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    latest_path = os.path.join(checkpoint_dir, 'latest.pt')

    torch.manual_seed(hp['seed'])
    ds = HardTaskDPTDataset(tasks=TRAIN_TASKS, h_max=hp['h_max'], epoch_size=hp['epoch_size'],
                             seed_rng=hp['seed'], mix_ratios=mix_ratios)
    loader = DataLoader(ds, batch_size=hp['batch_size'], shuffle=False, num_workers=0,
                         collate_fn=lambda b: collate_dpt(b, h_max=hp['h_max']))

    model = DPT(d_model=hp['d_model'], n_layers=hp['n_layers'], n_heads=hp['n_heads'],
                dropout=hp['dropout'], n_bins=hp['n_bins']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp['lr'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda_factory(hp['warmup_steps']))

    global_step = 0
    if os.path.exists(latest_path):
        ckpt = torch.load(latest_path, map_location=device)
        model.load_state_dict(ckpt['model']); optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler']); global_step = ckpt['global_step']
        print(f"[{ratio_name}] RESUMED at global_step={global_step}")
    else:
        print(f"[{ratio_name}] starting fresh")

    model.train()
    data_iter = iter(loader)
    while global_step < hp['total_steps']:
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
        loss, _, per_dim_acc = compute_loss_and_acc(bin_logits, target_bins)

        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), hp['grad_clip'])
        optimizer.step(); scheduler.step()
        global_step += 1

        if global_step % hp['log_every'] == 0 or global_step == 1:
            print(f"[{ratio_name}] step {global_step}/{hp['total_steps']} loss={loss.item():.4f} "
                  f"mean_acc={per_dim_acc.mean().item():.4f}")
        if global_step % hp['checkpoint_every'] == 0 or global_step == hp['total_steps']:
            save_checkpoint(os.path.join(checkpoint_dir, f'step_{global_step}.pt'), model, optimizer,
                             scheduler, global_step, hp)
            save_checkpoint(latest_path, model, optimizer, scheduler, global_step, hp)

    # final training accuracy on a fresh batch
    model.eval()
    eval_ds = HardTaskDPTDataset(tasks=TRAIN_TASKS, h_max=hp['h_max'], epoch_size=512,
                                  seed_rng=hp['seed'] + 999, mix_ratios=mix_ratios)
    eval_loader = DataLoader(eval_ds, batch_size=256, shuffle=False, num_workers=0,
                              collate_fn=lambda b: collate_dpt(b, h_max=hp['h_max']))
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
        _, _, final_per_dim_acc = compute_loss_and_acc(bin_logits, target_bins)
    train_acc = float(final_per_dim_acc.mean().item())
    print(f"[{ratio_name}] final training accuracy: {train_acc:.4f}")
    return model, train_acc


class ContextBuffer:
    def __init__(self, h_max):
        self.h_max = h_max
        self.obs, self.action, self.reward, self.next_obs = [], [], [], []

    def append(self, obs, action, reward, next_obs):
        self.obs.append(obs); self.action.append(action)
        self.reward.append(reward); self.next_obs.append(next_obs)
        if len(self.obs) > self.h_max:
            self.obs.pop(0); self.action.pop(0); self.reward.pop(0); self.next_obs.pop(0)

    def __len__(self):
        return len(self.obs)


def dpt_decode_action(model, discretizer, mean, std, buf, query_obs_raw, device):
    h = len(buf)
    if h > 0:
        ctx_obs = (np.asarray(buf.obs, dtype=np.float64) - mean) / std
        ctx_next_obs = (np.asarray(buf.next_obs, dtype=np.float64) - mean) / std
        ctx_action = np.asarray(buf.action, dtype=np.float64)
        ctx_reward = np.asarray(buf.reward, dtype=np.float64)
    else:
        ctx_obs = np.zeros((0, 52)); ctx_next_obs = np.zeros((0, 52))
        ctx_action = np.zeros((0, 9)); ctx_reward = np.zeros((0,))

    context_obs = torch.from_numpy(ctx_obs).float().unsqueeze(0).to(device)
    context_next_obs = torch.from_numpy(ctx_next_obs).float().unsqueeze(0).to(device)
    context_action = torch.from_numpy(ctx_action).float().unsqueeze(0).to(device)
    context_reward = torch.from_numpy(ctx_reward).float().unsqueeze(0).to(device)
    context_mask = torch.ones(1, h, dtype=torch.bool, device=device)
    query_obs = torch.from_numpy((query_obs_raw - mean) / std).float().unsqueeze(0).to(device)

    with torch.no_grad():
        bin_logits = model(context_obs, context_action, context_reward, context_next_obs,
                            context_mask, query_obs)
    bin_idx = np.array([logits[0].argmax(dim=-1).item() for logits in bin_logits])
    return discretizer.bin_to_action(bin_idx[None, :])[0]


def run_self_play_episode(h_max, seed, model, discretizer, mean, std, device):
    from rewards.user_reward import SubmissionReward
    label, rel_path, subset, m = HELD_OUT_TASK
    env = build_env(rel_path, subset, m, SubmissionReward, outage_seed=seed)
    low, high = env.action_space[0].low, env.action_space[0].high
    buf = ContextBuffer(h_max)
    obs = env.reset()
    done = False
    while not done:
        cur_obs = np.asarray(obs[0])
        action = dpt_decode_action(model, discretizer, mean, std, buf, cur_obs, device)
        action = np.clip(action, low, high)
        next_obs, reward, done, info = env.step([list(action)])
        buf.append(cur_obs, action, reward[0], np.asarray(next_obs[0]))
        obs = next_obs
    return env.evaluate_citylearn_challenge()['average_score']['value']


def load_results():
    if os.path.exists(RESULTS_BASE_PATH + '.json'):
        with open(RESULTS_BASE_PATH + '.json') as f:
            return json.load(f)
    return {}


def save_results(results):
    tmp = RESULTS_BASE_PATH + '.json.tmp'
    with open(tmp, 'w') as f:
        json.dump(results, f, indent=2)
    os.replace(tmp, RESULTS_BASE_PATH + '.json')


def evaluate_ratio(ratio_name, model, discretizer, device, results):
    norm = np.load(NORMALIZER_PATH)
    mean, std = norm['mean'], norm['std']
    for h in H_SWEEP:
        key = f"{ratio_name}__online__selfplay__h{h}"
        if key in results:
            continue
        scores = []
        for seed in EVAL_SEEDS:
            s = run_self_play_episode(h, seed, model, discretizer, mean, std, device)
            scores.append(float(s))
        results[key] = {"scores": scores, "mean": float(np.mean(scores))}
        save_results(results)
        print(f"[{ratio_name}] h={h}: mean_score={results[key]['mean']:.4f}")
    return results


def main():
    setup_paths()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    low, high = get_action_bounds()
    print("Action bounds:")
    for i in range(len(low)):
        print(f"  dim {i}: low={low[i]:.7f} high={high[i]:.7f}")
    discretizer = ActionDiscretizer(low, high, BASE_HPARAMS['n_bins'])

    results = load_results()
    train_accs = results.get('_train_accs', {})

    for ratio_name, mix_ratios in RATIOS.items():
        print(f"\n{'='*70}\nRATIO {ratio_name}: {mix_ratios} (exploratory={RATIO_EXPL_PCT[ratio_name]}%)\n{'='*70}")
        t0 = time.perf_counter()
        model, train_acc = train_one_ratio(ratio_name, mix_ratios, device, discretizer)
        train_accs[ratio_name] = train_acc
        results['_train_accs'] = train_accs
        save_results(results)
        print(f"[{ratio_name}] training done, {time.perf_counter()-t0:.0f}s elapsed, evaluating...")

        model.eval()
        results = evaluate_ratio(ratio_name, model, discretizer, device, results)
        print(f"[{ratio_name}] fully done, {time.perf_counter()-t0:.0f}s total elapsed")

    print("\n\n" + "=" * 90)
    print("PART 2 HARD-TASK MIX-RATIO FRONTIER")
    print("=" * 90)
    print(f"{'ratio':<6} {'expl%':>6} {'h=0':>8} {'best':>8} {'best@h':>8} {'delta(h0-best)':>15} {'train_acc':>10}")
    for name in ['r1', 'r2', 'r3', 'r4']:
        h0 = results[f"{name}__online__selfplay__h0"]['mean']
        best_h = min(H_SWEEP, key=lambda h: results[f"{name}__online__selfplay__h{h}"]['mean'])
        best = results[f"{name}__online__selfplay__h{best_h}"]['mean']
        delta = h0 - best
        tacc = train_accs[name]
        print(f"{name:<6} {RATIO_EXPL_PCT[name]:>6} {h0:>8.4f} {best:>8.4f} {best_h:>8} {delta:>15.4f} {tacc:>10.4f}")


if __name__ == '__main__':
    main()
