"""
Group 1: settle the operating point and close Gate 6.

STEP 1 (do first, may invalidate everything else): evaluate r2's held-out score at its existing
checkpoints (1000-5000, already on disk), extend training to 15000 (3x the ACTUAL 5000 steps used
-- note: the original ask said "6000 steps used so far", but the real figure recorded in
gate6_sweep.log is 5000; using 15000 = 3x actual), evaluating held-out score at further
checkpoints (8000, 11000, 15000). Applies a plateau/still-improving test and only proceeds to
Steps 2-3 if plateaued (this branch is pre-specified by the user, not an interpretive choice).

STEP 2 (conditional): robustness of r2 vs r3/r4 on a WIDER held-out set -- additional unseen
capacity multipliers on the same held-out anchor (Family C), since no second genuinely-distinct
held-out anchor family exists among the admitted (52,9)-conformable 2023-challenge anchors (Gate
5a's checksum work already exhausted the pool: A, B for training, C held out -- reported, not
silently skipped).

STEP 3 (conditional): full 8-KPI breakdown (not just average_score) for the chosen model vs
CHESCA vs RBC, multiple confirmed-nonzero-outage seeds, mean + spread per KPI.

Resumable: each phase's results checkpoint to disk incrementally.
"""
import copy
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.use_deterministic_algorithms(True)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from model.dpt import DPT
from model.discretize import ActionDiscretizer
from data.dpt_dataset import collate_dpt
from data.dpt_dataset_hard import HardTaskDPTDataset
from envs.combined_task_sampler import TRAIN_TASKS, HELD_OUT_TASK, FAMILY_C_HELDOUT, build_env

CHESCA_REPO = os.path.join(PROJECT_ROOT, 'oracle', 'chesca_repo')
NORMALIZER_PATH = os.path.join(PROJECT_ROOT, 'data', 'normalizer_hard.npz')
CHECKPOINT_BASE_DIR = os.path.join(PROJECT_ROOT, 'checkpoints_hard_sweep')
RESULTS_PATH = os.path.join(PROJECT_ROOT, 'results', 'gate6b_group1_results.json')

RATIOS = {
    'r1': {'exploratory': 0.60, 'chesca_noisy': 0.20, 'selfplay': 0.20},
    'r2': {'exploratory': 0.50, 'chesca_noisy': 0.25, 'selfplay': 0.25},
    'r3': {'exploratory': 0.40, 'chesca_noisy': 0.30, 'selfplay': 0.30},
    'r4': {'exploratory': 0.25, 'chesca_noisy': 0.375, 'selfplay': 0.375},
}

ORIGINAL_TOTAL_STEPS = 5000  # what was ACTUALLY used, not the "6000" in the ask
EXTENDED_TOTAL_STEPS = 15000  # 3x actual
PHASE1_EXISTING_CHECKPOINTS = [1000, 2000, 3000, 4000, 5000]
PHASE1_NEW_CHECKPOINTS = [8000, 11000, 15000]
PHASE1_H_VALUES = [0, 72]
EVAL_SEEDS = [55555, 1020, 1025]  # confirmed nonzero outages, Family C (phase_3_1 buildings 4/5/6)
PLATEAU_THRESHOLD = 0.02  # matches the established GPU-noise floor

PHASE2_WIDER_CAPACITIES = [0.775, 0.925, 1.075, 1.225]  # 1.075 = original held-out point, kept for continuity
PHASE2_H_VALUES = [0, 24, 72]

BASE_HPARAMS = dict(seed=0, h_max=256, epoch_size=2_000_000, d_model=256, n_layers=4, n_heads=8,
                     dropout=0.1, n_bins=21, batch_size=64, lr=3.0e-4, warmup_steps=200,
                     grad_clip=1.0, checkpoint_every=500, log_every=20)


def setup_paths():
    os.chdir(CHESCA_REPO)
    sys.path.insert(0, CHESCA_REPO)


def get_action_bounds():
    from rewards.user_reward import SubmissionReward
    label, rel_path, subset, m = TRAIN_TASKS[0]
    env = build_env(rel_path, subset, m, SubmissionReward)
    return env.action_space[0].low.copy(), env.action_space[0].high.copy()


def load_results():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return {}


def save_results(results):
    tmp = RESULTS_PATH + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(results, f, indent=2)
    os.replace(tmp, RESULTS_PATH)


def load_model_from_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device)
    hp = ckpt.get('hparams', BASE_HPARAMS)
    model = DPT(d_model=hp['d_model'], n_layers=hp['n_layers'], n_heads=hp['n_heads'],
                dropout=hp['dropout'], n_bins=hp['n_bins']).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model


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


def run_self_play_episode(h_max, seed, capacity, model, discretizer, mean, std, device):
    from rewards.user_reward import SubmissionReward
    _, rel_path, subset = FAMILY_C_HELDOUT
    env = build_env(rel_path, subset, capacity, SubmissionReward, outage_seed=seed)
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
    return env.evaluate_citylearn_challenge()


def eval_checkpoint_heldout(ckpt_path, discretizer, device, h_values, capacities, seeds, tag, results):
    norm = np.load(NORMALIZER_PATH)
    mean, std = norm['mean'], norm['std']
    model = load_model_from_checkpoint(ckpt_path, device)

    out = {}
    for capacity in capacities:
        for h in h_values:
            key = f"{tag}__cap{capacity:.3f}__h{h}"
            if key in results:
                out[(capacity, h)] = results[key]['mean']
                continue
            scores = []
            for seed in seeds:
                metrics = run_self_play_episode(h, seed, capacity, model, discretizer, mean, std, device)
                scores.append(float(metrics['average_score']['value']))
            results[key] = {"scores": scores, "mean": float(np.mean(scores))}
            save_results(results)
            out[(capacity, h)] = results[key]['mean']
            print(f"  [{tag}] cap={capacity} h={h}: mean_score={results[key]['mean']:.4f}")
    return out, results


# ============================== PHASE 1 ==============================

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


def extend_r2_training(device, discretizer, results):
    hp = dict(BASE_HPARAMS)
    hp['total_steps'] = EXTENDED_TOTAL_STEPS
    checkpoint_dir = os.path.join(CHECKPOINT_BASE_DIR, 'r2')
    latest_path = os.path.join(checkpoint_dir, 'latest.pt')

    torch.manual_seed(hp['seed'])
    ds = HardTaskDPTDataset(tasks=TRAIN_TASKS, h_max=hp['h_max'], epoch_size=hp['epoch_size'],
                             seed_rng=hp['seed'], mix_ratios=RATIOS['r2'])
    loader = DataLoader(ds, batch_size=hp['batch_size'], shuffle=False, num_workers=0,
                         collate_fn=lambda b: collate_dpt(b, h_max=hp['h_max']))

    model = DPT(d_model=hp['d_model'], n_layers=hp['n_layers'], n_heads=hp['n_heads'],
                dropout=hp['dropout'], n_bins=hp['n_bins']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp['lr'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda_factory(hp['warmup_steps']))

    ckpt = torch.load(latest_path, map_location=device)
    model.load_state_dict(ckpt['model']); optimizer.load_state_dict(ckpt['optimizer'])
    scheduler.load_state_dict(ckpt['scheduler']); global_step = ckpt['global_step']
    print(f"[r2-extend] resumed at global_step={global_step}, extending to {hp['total_steps']}")

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

        if global_step % hp['log_every'] == 0:
            print(f"[r2-extend] step {global_step}/{hp['total_steps']} loss={loss.item():.4f} "
                  f"mean_acc={per_dim_acc.mean().item():.4f}")
        if global_step in PHASE1_NEW_CHECKPOINTS or global_step == hp['total_steps']:
            step_path = os.path.join(checkpoint_dir, f'step_{global_step}.pt')
            save_checkpoint(step_path, model, optimizer, scheduler, global_step, hp)
            save_checkpoint(latest_path, model, optimizer, scheduler, global_step, hp)
            print(f"[r2-extend] checkpointed at step {global_step}")

    return checkpoint_dir


def phase1(device, discretizer, results):
    print("\n" + "=" * 70)
    print("STEP 1: longer-training check for r2")
    print("=" * 70)
    checkpoint_dir = os.path.join(CHECKPOINT_BASE_DIR, 'r2')

    trend = {}
    for step in PHASE1_EXISTING_CHECKPOINTS:
        ckpt_path = os.path.join(checkpoint_dir, f'step_{step}.pt')
        out, results = eval_checkpoint_heldout(ckpt_path, discretizer, device, PHASE1_H_VALUES,
                                                [1.075], EVAL_SEEDS, f"phase1_step{step}", results)
        trend[step] = min(out[(1.075, h)] for h in PHASE1_H_VALUES)  # best-over-h at this checkpoint

    extend_r2_training(device, discretizer, results)

    for step in PHASE1_NEW_CHECKPOINTS:
        ckpt_path = os.path.join(checkpoint_dir, f'step_{step}.pt')
        out, results = eval_checkpoint_heldout(ckpt_path, discretizer, device, PHASE1_H_VALUES,
                                                [1.075], EVAL_SEEDS, f"phase1_step{step}", results)
        trend[step] = min(out[(1.075, h)] for h in PHASE1_H_VALUES)

    print("\nHeld-out best-score-over-h trend across training (lower better):")
    all_steps = PHASE1_EXISTING_CHECKPOINTS + PHASE1_NEW_CHECKPOINTS
    for step in all_steps:
        print(f"  step {step:>6}: {trend[step]:.4f}")

    last_delta = trend[all_steps[-2]] - trend[all_steps[-1]]
    second_last_delta = trend[all_steps[-3]] - trend[all_steps[-2]]
    plateaued = abs(last_delta) < PLATEAU_THRESHOLD and abs(second_last_delta) < PLATEAU_THRESHOLD
    print(f"\nDelta (step {all_steps[-2]} -> {all_steps[-1]}): {last_delta:+.4f}")
    print(f"Delta (step {all_steps[-3]} -> {all_steps[-2]}): {second_last_delta:+.4f}")
    print(f"VERDICT: {'PLATEAUED' if plateaued else 'STILL IMPROVING'} "
          f"(threshold={PLATEAU_THRESHOLD})")

    final_checkpoint = os.path.join(checkpoint_dir, f'step_{all_steps[-1]}.pt')
    return plateaued, final_checkpoint, results


# ============================== PHASE 2 ==============================

def phase2(r2_checkpoint, device, discretizer, results):
    print("\n" + "=" * 70)
    print("STEP 2: robustness of r2 vs r3/r4 on wider held-out set")
    print("=" * 70)
    print("Note: no second genuinely-distinct held-out anchor family exists among the admitted "
          "(52,9)-conformable 2023-challenge anchors (Gate 5a's checksum work found only 3 "
          "distinct data families: A, B used for training, C held out) -- robustness check uses "
          "additional unseen CAPACITY settings on Family C instead.")

    checkpoints = {
        'r2': r2_checkpoint,
        'r3': os.path.join(CHECKPOINT_BASE_DIR, 'r3', 'step_5000.pt'),
        'r4': os.path.join(CHECKPOINT_BASE_DIR, 'r4', 'step_5000.pt'),
    }

    table = {}
    for name, ckpt_path in checkpoints.items():
        out, results = eval_checkpoint_heldout(ckpt_path, discretizer, device, PHASE2_H_VALUES,
                                                PHASE2_WIDER_CAPACITIES, EVAL_SEEDS[:2], f"phase2_{name}", results)
        table[name] = out

    print("\nWider held-out robustness table (best-over-h per capacity, lower better):")
    header = f"{'ratio':<6}" + "".join(f"{'cap='+str(c):>12}" for c in PHASE2_WIDER_CAPACITIES) + f"{'mean':>10}"
    print(header)
    means = {}
    for name in ['r2', 'r3', 'r4']:
        row_vals = [min(table[name][(c, h)] for h in PHASE2_H_VALUES) for c in PHASE2_WIDER_CAPACITIES]
        means[name] = float(np.mean(row_vals))
        print(f"{name:<6}" + "".join(f"{v:>12.4f}" for v in row_vals) + f"{means[name]:>10.4f}")

    best_name = min(means, key=lambda n: means[n])
    spread = max(means.values()) - min(means.values())
    print(f"\nBest mean: {best_name} ({means[best_name]:.4f}); spread across ratios: {spread:.4f} "
          f"({'within' if spread < PLATEAU_THRESHOLD else 'BEYOND'} the ~0.02 noise floor)")

    return checkpoints, means, results


# ============================== PHASE 3 ==============================

def run_chesca_episode(seed, capacity):
    from rewards.user_reward import SubmissionReward
    from agents.user_agent import SubmissionAgent
    from local_evaluation import WrapperEnv
    _, rel_path, subset = FAMILY_C_HELDOUT
    env = build_env(rel_path, subset, capacity, SubmissionReward, outage_seed=seed)
    env_data = dict(observation_names=env.observation_names, action_names=env.action_names,
                     observation_space=env.observation_space, action_space=env.action_space,
                     time_steps=env.time_steps, random_seed=None, episode_tracker=None,
                     seconds_per_time_step=None, buildings_metadata=env.get_metadata()['buildings'])
    agent = SubmissionAgent(WrapperEnv(env_data))
    obs = env.reset()
    actions = agent.register_reset(obs)
    done = False
    while not done:
        obs, reward, done, info = env.step(actions)
        if not done:
            actions = agent.predict(obs)
    return env.evaluate_citylearn_challenge()


def run_rbc_episode(seed, capacity):
    from rewards.user_reward import SubmissionReward
    from citylearn.agents.rbc import BasicRBC
    _, rel_path, subset = FAMILY_C_HELDOUT
    env = build_env(rel_path, subset, capacity, SubmissionReward, outage_seed=seed)
    agent = BasicRBC(env)
    obs = env.reset()
    done = False
    while not done:
        actions = agent.predict(obs)
        obs, reward, done, info = env.step(actions)
    return env.evaluate_citylearn_challenge()


KPI_KEYS = ['carbon_emissions_total', 'discomfort_proportion', 'ramping_average',
            'daily_one_minus_load_factor_average', 'daily_peak_average', 'annual_peak_average',
            'one_minus_thermal_resilience_proportion', 'power_outage_normalized_unserved_energy_total']


def phase3(chosen_name, chosen_checkpoint, device, discretizer, results):
    print("\n" + "=" * 70)
    print(f"STEP 3: full 8-KPI breakdown for {chosen_name} vs CHESCA vs RBC")
    print("=" * 70)
    norm = np.load(NORMALIZER_PATH)
    mean, std = norm['mean'], norm['std']
    model = load_model_from_checkpoint(chosen_checkpoint, device)
    capacity = 1.075  # original held-out capacity, most representative

    # use the CHOSEN ratio's own best-known h (from the original hard-task frontier), not a
    # blind h=256 -- h=256 is often NOT the best operating point per every frontier table so far.
    BEST_KNOWN_H = {'r1': 72, 'r2': 24, 'r3': 24, 'r4': 72}
    model_h = BEST_KNOWN_H.get(chosen_name, 24)
    print(f"Using h={model_h} for '{chosen_name}' (its own best-known operating point, not h=256)")

    per_policy_kpis = {}
    for policy in ['model', 'chesca', 'rbc']:
        key_prefix = f"phase3_{policy}_h{model_h}" if policy == 'model' else f"phase3_{policy}"
        kpi_runs = {k: [] for k in KPI_KEYS + ['average_score']}
        for seed in EVAL_SEEDS:
            key = f"{key_prefix}__seed{seed}"
            if key not in results:
                if policy == 'model':
                    metrics = run_self_play_episode(model_h, seed, capacity, model, discretizer, mean, std, device)
                elif policy == 'chesca':
                    metrics = run_chesca_episode(seed, capacity)
                else:
                    metrics = run_rbc_episode(seed, capacity)
                results[key] = {k: float(metrics[k]['value']) for k in KPI_KEYS} | \
                                {'average_score': float(metrics['average_score']['value'])}
                save_results(results)
            for k in KPI_KEYS + ['average_score']:
                kpi_runs[k].append(results[key][k])
            print(f"  [{policy}] seed={seed}: average_score={results[key]['average_score']:.4f}")
        per_policy_kpis[policy] = kpi_runs

    print("\nNote: any seed with ZERO outage occurrences for this task gives NaN for the two")
    print("outage-conditional KPIs (M, S) -- excluded via nanmean/nanstd, not silently averaged in.")
    for policy in ['model', 'chesca', 'rbc']:
        for k in ['one_minus_thermal_resilience_proportion', 'power_outage_normalized_unserved_energy_total']:
            vals = per_policy_kpis[policy][k]
            n_nan = sum(1 for v in vals if v != v)  # NaN != NaN
            if n_nan:
                print(f"  {policy}.{k}: {n_nan}/{len(vals)} seed(s) NaN (zero-outage), excluded from mean")

    print("\n8-KPI table (nanmean +/- nanstd across seeds, lower better for all):")
    header = f"{'KPI':<45} {'model':>16} {'chesca':>16} {'rbc':>16}"
    print(header)
    for k in KPI_KEYS + ['average_score']:
        row = []
        for policy in ['model', 'chesca', 'rbc']:
            vals = np.array(per_policy_kpis[policy][k])
            row.append(f"{np.nanmean(vals):.3f}+/-{np.nanstd(vals):.3f}")
        print(f"{k:<45} {row[0]:>16} {row[1]:>16} {row[2]:>16}")

    print("\nFlagging KPIs where model wins on composite but loses badly on individual KPIs "
          "(especially M=thermal resilience, S=unserved energy):")
    model_avg = np.nanmean(per_policy_kpis['model']['average_score'])
    chesca_avg = np.nanmean(per_policy_kpis['chesca']['average_score'])
    for k in ['one_minus_thermal_resilience_proportion', 'power_outage_normalized_unserved_energy_total']:
        model_k = np.nanmean(per_policy_kpis['model'][k])
        chesca_k = np.nanmean(per_policy_kpis['chesca'][k])
        if model_avg <= chesca_avg and model_k > chesca_k + PLATEAU_THRESHOLD:
            print(f"  FLAG: model wins composite ({model_avg:.3f} <= {chesca_avg:.3f}) but LOSES on "
                  f"{k} ({model_k:.3f} > {chesca_k:.3f})")
        else:
            print(f"  {k}: model={model_k:.3f} chesca={chesca_k:.3f} -- no flag")

    return results


def main():
    setup_paths()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    low, high = get_action_bounds()
    discretizer = ActionDiscretizer(low, high, BASE_HPARAMS['n_bins'])

    results = load_results()

    t0 = time.perf_counter()
    plateaued, final_r2_checkpoint, results = phase1(device, discretizer, results)
    print(f"\n[Phase 1 complete, {time.perf_counter()-t0:.0f}s elapsed]")

    if not plateaued:
        print("\n" + "!" * 70)
        print("STOPPING per instructions: held-out score is STILL IMPROVING at the longer training")
        print("budget. The whole r1-r4 frontier was undertrained and needs a longer-training rerun")
        print("before the ranking can be trusted. NOT proceeding to Steps 2-3.")
        print("!" * 70)
        return

    print("\n[Phase 1 verdict: PLATEAUED -- proceeding to Step 2]")
    checkpoints, means, results = phase2(final_r2_checkpoint, device, discretizer, results)
    print(f"\n[Phase 2 complete, {time.perf_counter()-t0:.0f}s elapsed]")

    chosen_name = min(means, key=lambda n: means[n])
    print(f"\n[Proceeding to Step 3 with chosen_name={chosen_name} (best mean on wider held-out set)]")
    results = phase3(chosen_name, checkpoints[chosen_name], device, discretizer, results)

    print(f"\n=== TOTAL RUNTIME: {time.perf_counter()-t0:.0f}s ({(time.perf_counter()-t0)/60:.1f} min) ===")


if __name__ == '__main__':
    main()
