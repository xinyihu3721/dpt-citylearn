"""
Part 2, step 2: per-task (anchor x capacity) harvest for the HARD task distribution -- CHESCA
labels (clean), exploratory context (7-seed pooled random + RBC), chesca_noisy context, and
self-generated context (greedy + sampled self-play from the LOCKED r3 model -- query-first, so
the current model/dpt.py.DPT class is used directly, no legacy shim needed this time).

One SGE array task per TRAIN_TASK (10 total: 2 anchor families x 5 capacity multipliers).
"""
import json
import os
import sys
import time

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHESCA_REPO = os.path.join(PROJECT_ROOT, 'oracle', 'chesca_repo')
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, CHESCA_REPO)

import citylearn
from citylearn.agents.rbc import BasicRBC
from agents.user_agent import SubmissionAgent
from rewards.user_reward import SubmissionReward
from local_evaluation import WrapperEnv

from envs.combined_task_sampler import build_env, task_name, TRAIN_TASKS, OUTAGE_SEED
from model.dpt import DPT
from model.discretize import ActionDiscretizer

NOISE_STD = 0.05
N_RANDOM_SEEDS = 7
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
LABELS_DIR = os.path.join(DATA_DIR, 'labels_hard')
CONTEXT_DIR = os.path.join(DATA_DIR, 'context_hard')
R3_CHECKPOINT = os.path.join(PROJECT_ROOT, 'checkpoints_locked', 'r3_operating_model.pt')
NORMALIZER_R3 = os.path.join(DATA_DIR, 'normalizer_capacity.npz')  # r3's own normalizer, used ONLY to decode ITS OWN self-play actions correctly
H_MAX = 256


def build_chesca_agent(env):
    env_data = dict(
        observation_names=env.observation_names, action_names=env.action_names,
        observation_space=env.observation_space, action_space=env.action_space,
        time_steps=env.time_steps, random_seed=None, episode_tracker=None,
        seconds_per_time_step=None, buildings_metadata=env.get_metadata()['buildings'],
    )
    return SubmissionAgent(WrapperEnv(env_data))


def save_rollout(name, obs_arr, action_arr, reward_arr, next_obs_arr, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{name}.npz')
    np.savez(out_path, obs=obs_arr, action=action_arr, reward=reward_arr, next_obs=next_obs_arr)
    return out_path


def rollout_chesca_labels(rel_path, subset, m):
    env = build_env(rel_path, subset, m, SubmissionReward)
    agent = build_chesca_agent(env)
    obs = env.reset()
    actions = agent.register_reset(obs)

    obs_hist, action_hist, reward_hist, next_obs_hist = [], [], [], []
    done = False
    while not done:
        cur_obs = obs[0]
        next_obs, reward, done, info = env.step(actions)
        obs_hist.append(cur_obs); action_hist.append(list(actions[0]))
        reward_hist.append(reward[0]); next_obs_hist.append(next_obs[0])
        obs = next_obs
        if not done:
            actions = agent.predict(obs)

    return (np.asarray(obs_hist, dtype=np.float64), np.asarray(action_hist, dtype=np.float64),
            np.asarray(reward_hist, dtype=np.float64), np.asarray(next_obs_hist, dtype=np.float64))


def rollout_random_one_seed(rel_path, subset, m, rng_seed):
    env = build_env(rel_path, subset, m, SubmissionReward)
    rng = np.random.default_rng(rng_seed)
    low, high = env.action_space[0].low, env.action_space[0].high
    obs = env.reset()
    obs_hist, action_hist, reward_hist, next_obs_hist = [], [], [], []
    done = False
    while not done:
        action = rng.uniform(low, high)
        cur_obs = obs[0]
        next_obs, reward, done, info = env.step([list(action)])
        obs_hist.append(cur_obs); action_hist.append(list(action))
        reward_hist.append(reward[0]); next_obs_hist.append(next_obs[0])
        obs = next_obs
    return (np.asarray(obs_hist, dtype=np.float64), np.asarray(action_hist, dtype=np.float64),
            np.asarray(reward_hist, dtype=np.float64), np.asarray(next_obs_hist, dtype=np.float64))


def rollout_rbc(rel_path, subset, m):
    env = build_env(rel_path, subset, m, SubmissionReward)
    agent = BasicRBC(env)
    obs = env.reset()
    obs_hist, action_hist, reward_hist, next_obs_hist = [], [], [], []
    done = False
    while not done:
        cur_obs = obs[0]
        actions = agent.predict(obs)
        next_obs, reward, done, info = env.step(actions)
        obs_hist.append(cur_obs); action_hist.append(list(actions[0]))
        reward_hist.append(reward[0]); next_obs_hist.append(next_obs[0])
        obs = next_obs
    return (np.asarray(obs_hist, dtype=np.float64), np.asarray(action_hist, dtype=np.float64),
            np.asarray(reward_hist, dtype=np.float64), np.asarray(next_obs_hist, dtype=np.float64))


def rollout_chesca_noisy(rel_path, subset, m):
    env = build_env(rel_path, subset, m, SubmissionReward)
    agent = build_chesca_agent(env)
    rng = np.random.default_rng(OUTAGE_SEED + 1_000_000)
    low, high = env.action_space[0].low, env.action_space[0].high
    obs = env.reset()
    clean_actions = agent.register_reset(obs)
    obs_hist, action_hist, reward_hist, next_obs_hist = [], [], [], []
    done = False
    while not done:
        noisy_action = np.clip(np.asarray(clean_actions[0]) + rng.normal(0.0, NOISE_STD, size=low.shape), low, high)
        cur_obs = obs[0]
        next_obs, reward, done, info = env.step([list(noisy_action)])
        obs_hist.append(cur_obs); action_hist.append(list(noisy_action))
        reward_hist.append(reward[0]); next_obs_hist.append(next_obs[0])
        obs = next_obs
        if not done:
            clean_actions = agent.predict(obs)
    return (np.asarray(obs_hist, dtype=np.float64), np.asarray(action_hist, dtype=np.float64),
            np.asarray(reward_hist, dtype=np.float64), np.asarray(next_obs_hist, dtype=np.float64))


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


def dpt_decode_action(model, discretizer, mean, std, buf, query_obs_raw, device, sample, rng):
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
    bin_idx = np.zeros(discretizer.action_dim, dtype=np.int64)
    for d, logits in enumerate(bin_logits):
        if sample:
            probs = torch.softmax(logits[0], dim=-1).cpu().numpy()
            bin_idx[d] = rng.choice(len(probs), p=probs)
        else:
            bin_idx[d] = logits[0].argmax(dim=-1).item()
    return discretizer.bin_to_action(bin_idx[None, :])[0]


_r3_model_cache = None


def get_r3_model_and_discretizer(low, high):
    global _r3_model_cache
    if _r3_model_cache is None:
        device = torch.device('cpu')
        ckpt = torch.load(R3_CHECKPOINT, map_location=device)
        cfg = ckpt['config']
        model = DPT(d_model=cfg['model']['d_model'], n_layers=cfg['model']['n_layers'],
                    n_heads=cfg['model']['n_heads'], dropout=cfg['model']['dropout'],
                    n_bins=cfg['model']['n_bins']).to(device)
        model.load_state_dict(ckpt['model'])
        model.eval()
        discretizer = ActionDiscretizer(low, high, cfg['model']['n_bins'])
        _r3_model_cache = (model, discretizer, device)
    return _r3_model_cache


def rollout_selfplay(rel_path, subset, m, sample, rng_seed):
    env = build_env(rel_path, subset, m, SubmissionReward)
    low, high = env.action_space[0].low, env.action_space[0].high
    model, discretizer, device = get_r3_model_and_discretizer(low, high)
    norm = np.load(NORMALIZER_R3)
    mean, std = norm['mean'], norm['std']

    buf = ContextBuffer(H_MAX)
    rng = np.random.default_rng(rng_seed)
    obs = env.reset()
    obs_hist, action_hist, reward_hist, next_obs_hist = [], [], [], []
    done = False
    while not done:
        cur_obs = np.asarray(obs[0])
        action = dpt_decode_action(model, discretizer, mean, std, buf, cur_obs, device, sample, rng)
        action = np.clip(action, low, high)
        next_obs, reward, done, info = env.step([list(action)])
        obs_hist.append(cur_obs); action_hist.append(action)
        reward_hist.append(reward[0]); next_obs_hist.append(np.asarray(next_obs[0]))
        buf.append(cur_obs, action, reward[0], np.asarray(next_obs[0]))
        obs = next_obs
    return (np.asarray(obs_hist, dtype=np.float64), np.asarray(action_hist, dtype=np.float64),
            np.asarray(reward_hist, dtype=np.float64), np.asarray(next_obs_hist, dtype=np.float64))


def main():
    task_idx = int(os.environ['SGE_TASK_ID']) - 1
    label, rel_path, subset, m = TRAIN_TASKS[task_idx]
    name = task_name(label, m)
    print(f"Task {task_idx}: {name} (family={label}, rel_path={rel_path}, subset={subset}, capacity={m})")

    t0 = time.perf_counter()
    o, a, r, no = rollout_chesca_labels(rel_path, subset, m)
    save_rollout(f'{name}__chesca_label', o, a, r, no, LABELS_DIR)
    sidecar = {'task_name': name, 'family': label, 'rel_path': rel_path, 'subset': subset,
               'capacity_multiplier': m, 'n_steps': int(o.shape[0]), 'citylearn_version': citylearn.__version__}
    with open(os.path.join(LABELS_DIR, f'{name}__chesca_label.json'), 'w') as f:
        json.dump(sidecar, f, indent=2)
    print(f"chesca_label: {o.shape[0]} steps in {time.perf_counter()-t0:.1f}s")

    t0 = time.perf_counter()
    obs_l, act_l, rew_l, nobs_l = [], [], [], []
    for i in range(N_RANDOM_SEEDS):
        o, a, r, no = rollout_random_one_seed(rel_path, subset, m, OUTAGE_SEED * 100 + i)
        obs_l.append(o); act_l.append(a); rew_l.append(r); nobs_l.append(no)
    save_rollout(f'{name}__random', np.concatenate(obs_l), np.concatenate(act_l),
                 np.concatenate(rew_l), np.concatenate(nobs_l), CONTEXT_DIR)
    print(f"random (7 seeds pooled): {sum(x.shape[0] for x in obs_l)} steps in {time.perf_counter()-t0:.1f}s")

    t0 = time.perf_counter()
    o, a, r, no = rollout_rbc(rel_path, subset, m)
    save_rollout(f'{name}__rbc', o, a, r, no, CONTEXT_DIR)
    print(f"rbc: {o.shape[0]} steps in {time.perf_counter()-t0:.1f}s")

    t0 = time.perf_counter()
    o, a, r, no = rollout_chesca_noisy(rel_path, subset, m)
    save_rollout(f'{name}__chesca_noisy', o, a, r, no, CONTEXT_DIR)
    print(f"chesca_noisy: {o.shape[0]} steps in {time.perf_counter()-t0:.1f}s")

    t0 = time.perf_counter()
    o, a, r, no = rollout_selfplay(rel_path, subset, m, sample=False, rng_seed=OUTAGE_SEED)
    save_rollout(f'{name}__selfplay_greedy', o, a, r, no, CONTEXT_DIR)
    print(f"selfplay_greedy (r3 model): {o.shape[0]} steps in {time.perf_counter()-t0:.1f}s")

    t0 = time.perf_counter()
    o, a, r, no = rollout_selfplay(rel_path, subset, m, sample=True, rng_seed=OUTAGE_SEED + 2_000_000)
    save_rollout(f'{name}__selfplay_sampled', o, a, r, no, CONTEXT_DIR)
    print(f"selfplay_sampled (r3 model): {o.shape[0]} steps in {time.perf_counter()-t0:.1f}s")


if __name__ == '__main__':
    main()
