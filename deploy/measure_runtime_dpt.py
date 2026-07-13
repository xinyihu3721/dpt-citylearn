"""
Runtime-per-prediction measurement: DPT (GPU) on the held-out task (Family C, capacity=1.075x),
seed=55555 (same seed used as phase3's first eval seed in gate6b_group1.py, for consistency).

Two measurements, both using the CHOSEN model (r3, checkpoints_hard_sweep/r3/step_5000.pt):
  1. Real full-episode rollout at h_max=256 (the trained H_max) -- per-step wall-clock timed
     individually (context assembly + model forward + action decode), with torch.cuda.synchronize()
     bracketing each timed region so GPU-async kernel launches don't make the CPU-side timer lie.
     First WARMUP_STEPS steps are discarded from the reported stats (CUDA init / cudnn autotune
     overhead), but the full per-step series (including h at each step) is saved so the discarded
     steps are visible, not silently dropped.
  2. Isolated fixed-context-length microbenchmark at h in {0, 128, 256}: context buffers are
     snapshotted from the real episode at the point each h is first reached, then the same
     (context, query) pair is fed through dpt_decode_action REPS times (after WARMUP reps
     discarded) to get a clean mean+std at each fixed h, since the real-episode series only
     passes through each intermediate h once.

Run interactively in a GPU session, e.g.:
  module load miniconda && conda activate citylearn && \
  LD_LIBRARY_PATH=<path-to-your-conda-env>/lib:$LD_LIBRARY_PATH \
  python deploy/measure_runtime_dpt.py
"""
import json
import os
import sys
import time

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
CHESCA_REPO = os.path.join(PROJECT_ROOT, 'oracle', 'chesca_repo')

from model.dpt import DPT
from model.discretize import ActionDiscretizer
from envs.combined_task_sampler import FAMILY_C_HELDOUT, build_env

CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, 'checkpoints_hard_sweep', 'r3', 'step_5000.pt')
NORMALIZER_PATH = os.path.join(PROJECT_ROOT, 'data', 'normalizer_hard.npz')
OUT_PATH = os.path.join(PROJECT_ROOT, 'results', 'results_runtime_dpt.json')

CAPACITY = 1.075
SEED = 55555
H_MAX = 256
WARMUP_STEPS = 10          # discarded from the real-episode per-step stats
FIXED_H_VALUES = [0, 128, 256]
FIXED_H_REPS = 200
FIXED_H_WARMUP = 20


def load_model_from_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device)
    hp = ckpt['hparams']
    model = DPT(d_model=hp['d_model'], n_layers=hp['n_layers'], n_heads=hp['n_heads'],
                dropout=hp['dropout'], n_bins=hp['n_bins']).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, hp


class ContextBuffer:
    def __init__(self, h_max):
        self.h_max = h_max
        self.obs, self.action, self.reward, self.next_obs = [], [], [], []

    def append(self, obs, action, reward, next_obs):
        self.obs.append(obs); self.action.append(action)
        self.reward.append(reward); self.next_obs.append(next_obs)
        if len(self.obs) > self.h_max:
            self.obs.pop(0); self.action.pop(0); self.reward.pop(0); self.next_obs.pop(0)

    def snapshot(self):
        return (list(self.obs), list(self.action), list(self.reward), list(self.next_obs))

    def __len__(self):
        return len(self.obs)


def build_tensors(buf_snapshot, query_obs_raw, mean, std, device, obs_dim, action_dim):
    obs, action, reward, next_obs = buf_snapshot
    h = len(obs)
    if h > 0:
        ctx_obs = (np.asarray(obs, dtype=np.float64) - mean) / std
        ctx_next_obs = (np.asarray(next_obs, dtype=np.float64) - mean) / std
        ctx_action = np.asarray(action, dtype=np.float64)
        ctx_reward = np.asarray(reward, dtype=np.float64)
    else:
        ctx_obs = np.zeros((0, obs_dim)); ctx_next_obs = np.zeros((0, obs_dim))
        ctx_action = np.zeros((0, action_dim)); ctx_reward = np.zeros((0,))

    context_obs = torch.from_numpy(ctx_obs).float().unsqueeze(0).to(device)
    context_next_obs = torch.from_numpy(ctx_next_obs).float().unsqueeze(0).to(device)
    context_action = torch.from_numpy(ctx_action).float().unsqueeze(0).to(device)
    context_reward = torch.from_numpy(ctx_reward).float().unsqueeze(0).to(device)
    context_mask = torch.ones(1, h, dtype=torch.bool, device=device)
    query_obs = torch.from_numpy((query_obs_raw - mean) / std).float().unsqueeze(0).to(device)
    return context_obs, context_action, context_reward, context_next_obs, context_mask, query_obs


def timed_decode(model, discretizer, tensors, device, use_cuda):
    context_obs, context_action, context_reward, context_next_obs, context_mask, query_obs = tensors
    if use_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        bin_logits = model(context_obs, context_action, context_reward, context_next_obs,
                            context_mask, query_obs)
        bin_idx = np.array([logits[0].argmax(dim=-1).item() for logits in bin_logits])
        action = discretizer.bin_to_action(bin_idx[None, :])[0]
    if use_cuda:
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    return action, (t1 - t0)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_cuda = device.type == 'cuda'
    gpu_name = torch.cuda.get_device_name(0) if use_cuda else None
    print(f"device={device}, gpu_name={gpu_name}")

    model, hp = load_model_from_checkpoint(CHECKPOINT_PATH, device)
    discretizer = ActionDiscretizer(*_action_bounds(), hp['n_bins'])
    norm = np.load(NORMALIZER_PATH)
    mean, std = norm['mean'], norm['std']
    obs_dim = mean.shape[0]

    _, rel_path, subset = FAMILY_C_HELDOUT
    from rewards.user_reward import SubmissionReward
    sys.path.insert(0, CHESCA_REPO)
    os.chdir(CHESCA_REPO)
    env = build_env(rel_path, subset, CAPACITY, SubmissionReward, outage_seed=SEED)
    action_dim = env.action_space[0].shape[0]
    low, high = env.action_space[0].low, env.action_space[0].high

    buf = ContextBuffer(H_MAX)
    fixed_h_snapshots = {}  # h -> (buf_snapshot, query_obs_raw)

    obs = env.reset()
    done = False
    step_times, step_hs = [], []
    step_idx = 0
    while not done:
        cur_obs = np.asarray(obs[0])
        h_now = len(buf)
        if h_now in FIXED_H_VALUES and h_now not in fixed_h_snapshots:
            fixed_h_snapshots[h_now] = (buf.snapshot(), cur_obs.copy())

        tensors = build_tensors(buf.snapshot(), cur_obs, mean, std, device, obs_dim, action_dim)
        action, dt = timed_decode(model, discretizer, tensors, device, use_cuda)
        step_times.append(dt)
        step_hs.append(h_now)

        action = np.clip(action, low, high)
        next_obs, reward, done, info = env.step([list(action)])
        buf.append(cur_obs, action, reward[0], np.asarray(next_obs[0]))
        obs = next_obs
        step_idx += 1

    n_steps = step_idx
    step_times = np.array(step_times)
    step_hs = np.array(step_hs)
    print(f"Episode complete: n_steps={n_steps}")

    kept = step_times[WARMUP_STEPS:]
    real_episode_stats = {
        'n_steps': int(n_steps),
        'warmup_steps_discarded': WARMUP_STEPS,
        'mean_ms': float(kept.mean() * 1000),
        'std_ms': float(kept.std() * 1000),
        'total_episode_wallclock_s': float(step_times.sum()),
        'h_max_reached': int(step_hs.max()),
        'steps_at_h_max': int((step_hs == H_MAX).sum()),
    }
    print(f"Real-episode (h grows 0->{H_MAX} then holds): "
          f"mean={real_episode_stats['mean_ms']:.3f}ms std={real_episode_stats['std_ms']:.3f}ms "
          f"(first {WARMUP_STEPS} steps discarded), total={real_episode_stats['total_episode_wallclock_s']:.2f}s")

    # missing fixed-h snapshots (e.g. h=256 might not be hit if H_MAX reached exactly at last step
    # -- won't happen here since n_steps >> H_MAX, but guard anyway)
    for h in FIXED_H_VALUES:
        if h not in fixed_h_snapshots:
            print(f"WARNING: h={h} never observed during the real episode; skipping fixed-h bench for it")

    fixed_h_results = {}
    for h in FIXED_H_VALUES:
        if h not in fixed_h_snapshots:
            continue
        buf_snap, query_obs_raw = fixed_h_snapshots[h]
        tensors = build_tensors(buf_snap, query_obs_raw, mean, std, device, obs_dim, action_dim)
        times = []
        for rep in range(FIXED_H_REPS + FIXED_H_WARMUP):
            _, dt = timed_decode(model, discretizer, tensors, device, use_cuda)
            times.append(dt)
        times = np.array(times[FIXED_H_WARMUP:])
        fixed_h_results[h] = {
            'mean_ms': float(times.mean() * 1000),
            'std_ms': float(times.std() * 1000),
            'reps': FIXED_H_REPS,
            'warmup_reps_discarded': FIXED_H_WARMUP,
        }
        print(f"Fixed h={h}: mean={fixed_h_results[h]['mean_ms']:.3f}ms std={fixed_h_results[h]['std_ms']:.3f}ms "
              f"(n={FIXED_H_REPS} reps, {FIXED_H_WARMUP} warmup discarded)")

    out = {
        'controller': 'dpt_r3',
        'checkpoint': CHECKPOINT_PATH,
        'device': str(device),
        'gpu_name': gpu_name,
        'task': 'familyC_cap1.075x',
        'seed': SEED,
        'h_max_trained': H_MAX,
        'real_episode': real_episode_stats,
        'fixed_h': {str(k): v for k, v in fixed_h_results.items()},
        'per_step_times_ms': (step_times * 1000).tolist(),
        'per_step_h': step_hs.tolist(),
    }
    tmp = OUT_PATH + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(out, f, indent=2)
    os.replace(tmp, OUT_PATH)
    print(f"Saved: {OUT_PATH}")


def _action_bounds():
    sys.path.insert(0, CHESCA_REPO)
    os.chdir(CHESCA_REPO)
    from rewards.user_reward import SubmissionReward
    from envs.combined_task_sampler import TRAIN_TASKS
    label, rel_path, subset, m = TRAIN_TASKS[0]
    env = build_env(rel_path, subset, m, SubmissionReward)
    return env.action_space[0].low.copy(), env.action_space[0].high.copy()


if __name__ == '__main__':
    main()
