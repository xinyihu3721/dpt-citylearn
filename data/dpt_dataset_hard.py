"""
Part 2, step 3: mixed-context DPT Dataset over the HARD (anchor-family x capacity) task
distribution. Same structure as dpt_dataset_mixed.py (three buckets: exploratory
[random+rbc], chesca_noisy, selfplay [greedy+sampled from r3]), configurable mix ratio,
explicit varied-order reshuffle every draw -- just keyed by (family, capacity) tasks instead
of capacity-only tasks, and pointed at data/{labels,context}_hard/.
"""
import os
import sys

import numpy as np
import torch
from torch.utils.data import Dataset

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from envs.combined_task_sampler import TRAIN_TASKS, task_name
from data.dpt_dataset import collate_dpt  # noqa: F401 (re-exported)

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
LABELS_DIR = os.path.join(DATA_DIR, 'labels_hard')
CONTEXT_DIR = os.path.join(DATA_DIR, 'context_hard')
NORMALIZER_PATH = os.path.join(DATA_DIR, 'normalizer_hard.npz')

DEFAULT_MIX_RATIOS = {'exploratory': 0.4, 'chesca_noisy': 0.3, 'selfplay': 0.3}


def _load_npz_fields(path):
    d = np.load(path)
    return d['obs'], d['action'], d['reward'], d['next_obs']


class HardTaskDPTDataset(Dataset):
    def __init__(self, tasks=None, h_max=256, epoch_size=1000, seed_rng=0,
                 normalizer_path=NORMALIZER_PATH, mix_ratios=None):
        self.tasks = list(tasks) if tasks is not None else list(TRAIN_TASKS)
        self.h_max = h_max
        self.epoch_size = epoch_size
        self.rng = np.random.default_rng(seed_rng)
        self.mix_ratios = dict(mix_ratios) if mix_ratios is not None else dict(DEFAULT_MIX_RATIOS)
        assert abs(sum(self.mix_ratios.values()) - 1.0) < 1e-6, \
            f"mix_ratios must sum to 1.0, got {self.mix_ratios}"

        norm = np.load(normalizer_path)
        self.obs_mean = norm['mean'].astype(np.float32)
        self.obs_std = norm['std'].astype(np.float32)
        self.obs_dim = self.obs_mean.shape[0]

        self.task_keys = []  # list of task identifiers used as dict keys
        self.buckets = {}
        self.label_pool = {}
        for (label, rel_path, subset, m) in self.tasks:
            name = task_name(label, m)
            key = name

            random_o, random_a, random_r, random_no = _load_npz_fields(
                os.path.join(CONTEXT_DIR, f'{name}__random.npz'))
            rbc_o, rbc_a, rbc_r, rbc_no = _load_npz_fields(
                os.path.join(CONTEXT_DIR, f'{name}__rbc.npz'))
            exploratory = dict(
                obs=np.concatenate([random_o, rbc_o], axis=0),
                action=np.concatenate([random_a, rbc_a], axis=0),
                reward=np.concatenate([random_r, rbc_r], axis=0),
                next_obs=np.concatenate([random_no, rbc_no], axis=0),
            )

            noisy_o, noisy_a, noisy_r, noisy_no = _load_npz_fields(
                os.path.join(CONTEXT_DIR, f'{name}__chesca_noisy.npz'))
            chesca_noisy = dict(obs=noisy_o, action=noisy_a, reward=noisy_r, next_obs=noisy_no)

            greedy_o, greedy_a, greedy_r, greedy_no = _load_npz_fields(
                os.path.join(CONTEXT_DIR, f'{name}__selfplay_greedy.npz'))
            sampled_o, sampled_a, sampled_r, sampled_no = _load_npz_fields(
                os.path.join(CONTEXT_DIR, f'{name}__selfplay_sampled.npz'))
            selfplay = dict(
                obs=np.concatenate([greedy_o, sampled_o], axis=0),
                action=np.concatenate([greedy_a, sampled_a], axis=0),
                reward=np.concatenate([greedy_r, sampled_r], axis=0),
                next_obs=np.concatenate([greedy_no, sampled_no], axis=0),
            )

            self.buckets[key] = {'exploratory': exploratory, 'chesca_noisy': chesca_noisy, 'selfplay': selfplay}

            label_d = np.load(os.path.join(LABELS_DIR, f'{name}__chesca_label.npz'))
            self.label_pool[key] = dict(obs=label_d['obs'], action=label_d['action'])
            self.task_keys.append(key)

        self.action_dim = self.buckets[self.task_keys[0]]['exploratory']['action'].shape[1]

    def __len__(self):
        return self.epoch_size

    def _normalize_obs(self, obs):
        return (obs - self.obs_mean) / self.obs_std

    def _bucket_counts(self, h):
        names = list(self.mix_ratios.keys())
        counts = {n: int(round(h * self.mix_ratios[n])) for n in names}
        diff = h - sum(counts.values())
        if diff != 0:
            biggest = max(names, key=lambda n: self.mix_ratios[n])
            counts[biggest] += diff
        return counts

    def __getitem__(self, idx):
        key = self.task_keys[self.rng.integers(len(self.task_keys))]
        buckets = self.buckets[key]

        h = int(self.rng.integers(1, self.h_max + 1))
        counts = self._bucket_counts(h)

        obs_parts, action_parts, reward_parts, next_obs_parts = [], [], [], []
        for bucket_name, n in counts.items():
            if n <= 0:
                continue
            pool = buckets[bucket_name]
            n_pool = pool['obs'].shape[0]
            n_eff = min(n, n_pool)
            idxs = self.rng.choice(n_pool, size=n_eff, replace=False)
            obs_parts.append(pool['obs'][idxs])
            action_parts.append(pool['action'][idxs])
            reward_parts.append(pool['reward'][idxs])
            next_obs_parts.append(pool['next_obs'][idxs])

        if obs_parts:
            ctx_obs = np.concatenate(obs_parts, axis=0)
            ctx_action = np.concatenate(action_parts, axis=0)
            ctx_reward = np.concatenate(reward_parts, axis=0)
            ctx_next_obs = np.concatenate(next_obs_parts, axis=0)

            perm = self.rng.permutation(ctx_obs.shape[0])
            ctx_obs, ctx_action = ctx_obs[perm], ctx_action[perm]
            ctx_reward, ctx_next_obs = ctx_reward[perm], ctx_next_obs[perm]
        else:
            ctx_obs = np.zeros((0, self.obs_dim)); ctx_action = np.zeros((0, self.action_dim))
            ctx_reward = np.zeros((0,)); ctx_next_obs = np.zeros((0, self.obs_dim))

        context_obs = self._normalize_obs(ctx_obs)
        context_next_obs = self._normalize_obs(ctx_next_obs)

        labels = self.label_pool[key]
        n_labels = labels['obs'].shape[0]
        q_idx = int(self.rng.integers(n_labels))
        query_obs = self._normalize_obs(labels['obs'][q_idx])
        action_label = labels['action'][q_idx]

        return {
            'seed': hash(key) % 100000,
            'task_name': key,
            'h': context_obs.shape[0],
            'context_obs': torch.from_numpy(context_obs.astype(np.float32)),
            'context_action': torch.from_numpy(ctx_action.astype(np.float32)),
            'context_reward': torch.from_numpy(ctx_reward.astype(np.float32)),
            'context_next_obs': torch.from_numpy(context_next_obs.astype(np.float32)),
            'query_obs': torch.from_numpy(query_obs.astype(np.float32)),
            'action_label': torch.from_numpy(action_label.astype(np.float32)),
        }
