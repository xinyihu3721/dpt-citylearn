"""
Gate 3 Step 4: DPT training-example Dataset/collator.

A "task instance" here is one outage-seed realization of the fixed schema
(citylearn_challenge_2023_phase_2_local_evaluation) -- see Gate 3 Step 1's finding that
CityLearnEnv(random_seed=X) alone does not vary the outage pattern; only explicitly overriding
the power-outage-model seed does, which is how the three task instances (73055/90000/12345)
were produced by gate3_harvest.py.

For each task instance we have:
  - a pooled CONTEXT bank: diverse (obs, action, reward, next_obs) transitions from random /
    RBC / CHESCA+noise rollouts (data/context/*__<policy>__seed<seed>.npz)
  - a LABEL bank: CHESCA on-policy (obs, action) pairs for that same task instance
    (data/labels/*__seed<seed>.npz -- note seed73055's file predates the "chesca_label" naming
    used for the two seeds added at Gate 3, see LABEL_FILES below)

Each training example: a random-size subset of CONTEXT transitions (size 1..H_max) drawn from
ONE task instance, plus one (query_obs, action_label) pair drawn from that SAME task instance's
CHESCA labels. obs/next_obs/query_obs are normalized with the Gate 3 Step 3 per-dim mean/std
normalizer (data/normalizer.npz); action/reward are left in their native raw units -- there is
no action or reward normalizer, only an observation normalizer.
"""
import os

import numpy as np
import torch
from torch.utils.data import Dataset

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
CONTEXT_DIR = os.path.join(DATA_DIR, 'context')
LABELS_DIR = os.path.join(DATA_DIR, 'labels')
NORMALIZER_PATH = os.path.join(DATA_DIR, 'normalizer.npz')

SCHEMA_NAME = 'citylearn_challenge_2023_phase_2_local_evaluation'
SEEDS = (73055, 90000, 12345)
CONTEXT_POLICIES = ('random', 'rbc', 'chesca_noisy')

LABEL_FILES = {
    73055: os.path.join(LABELS_DIR, f'{SCHEMA_NAME}__seed73055.npz'),
    90000: os.path.join(LABELS_DIR, f'{SCHEMA_NAME}__chesca_label__seed90000.npz'),
    12345: os.path.join(LABELS_DIR, f'{SCHEMA_NAME}__chesca_label__seed12345.npz'),
}


def context_file(seed, policy):
    return os.path.join(CONTEXT_DIR, f'{SCHEMA_NAME}__{policy}__seed{seed}.npz')


class DPTDataset(Dataset):
    def __init__(self, seeds=SEEDS, h_max=256, epoch_size=1000, seed_rng=0):
        self.seeds = list(seeds)
        self.h_max = h_max
        self.epoch_size = epoch_size
        self.rng = np.random.default_rng(seed_rng)

        norm = np.load(NORMALIZER_PATH)
        self.obs_mean = norm['mean'].astype(np.float32)
        self.obs_std = norm['std'].astype(np.float32)
        self.obs_dim = self.obs_mean.shape[0]

        self.context_pool = {}   # seed -> dict of obs/action/reward/next_obs arrays
        self.label_pool = {}     # seed -> dict of obs/action arrays
        for seed in self.seeds:
            obs_list, action_list, reward_list, next_obs_list = [], [], [], []
            for policy in CONTEXT_POLICIES:
                d = np.load(context_file(seed, policy))
                obs_list.append(d['obs'])
                action_list.append(d['action'])
                reward_list.append(d['reward'])
                next_obs_list.append(d['next_obs'])
            self.context_pool[seed] = dict(
                obs=np.concatenate(obs_list, axis=0),
                action=np.concatenate(action_list, axis=0),
                reward=np.concatenate(reward_list, axis=0),
                next_obs=np.concatenate(next_obs_list, axis=0),
            )

            d = np.load(LABEL_FILES[seed])
            self.label_pool[seed] = dict(obs=d['obs'], action=d['action'])

        self.action_dim = self.context_pool[self.seeds[0]]['action'].shape[1]

    def __len__(self):
        return self.epoch_size

    def _normalize_obs(self, obs):
        return (obs - self.obs_mean) / self.obs_std

    def __getitem__(self, idx):
        seed = self.seeds[self.rng.integers(len(self.seeds))]

        ctx = self.context_pool[seed]
        n_ctx = ctx['obs'].shape[0]
        h = int(self.rng.integers(1, self.h_max + 1))
        h = min(h, n_ctx)
        idxs = self.rng.choice(n_ctx, size=h, replace=False)

        context_obs = self._normalize_obs(ctx['obs'][idxs])
        context_action = ctx['action'][idxs]
        context_reward = ctx['reward'][idxs]
        context_next_obs = self._normalize_obs(ctx['next_obs'][idxs])

        labels = self.label_pool[seed]
        n_labels = labels['obs'].shape[0]
        q_idx = int(self.rng.integers(n_labels))
        query_obs = self._normalize_obs(labels['obs'][q_idx])
        action_label = labels['action'][q_idx]

        return {
            'seed': seed,
            'h': h,
            'context_obs': torch.from_numpy(context_obs.astype(np.float32)),
            'context_action': torch.from_numpy(context_action.astype(np.float32)),
            'context_reward': torch.from_numpy(context_reward.astype(np.float32)),
            'context_next_obs': torch.from_numpy(context_next_obs.astype(np.float32)),
            'query_obs': torch.from_numpy(query_obs.astype(np.float32)),
            'action_label': torch.from_numpy(action_label.astype(np.float32)),
        }


def collate_dpt(batch, h_max=None):
    """Pads variable-length context to a common length with a boolean validity mask."""
    obs_dim = batch[0]['context_obs'].shape[1]
    action_dim = batch[0]['context_action'].shape[1]
    pad_len = h_max if h_max is not None else max(ex['h'] for ex in batch)
    bsz = len(batch)

    context_obs = torch.zeros(bsz, pad_len, obs_dim, dtype=torch.float32)
    context_action = torch.zeros(bsz, pad_len, action_dim, dtype=torch.float32)
    context_reward = torch.zeros(bsz, pad_len, dtype=torch.float32)
    context_next_obs = torch.zeros(bsz, pad_len, obs_dim, dtype=torch.float32)
    context_mask = torch.zeros(bsz, pad_len, dtype=torch.bool)

    query_obs = torch.stack([ex['query_obs'] for ex in batch])
    action_label = torch.stack([ex['action_label'] for ex in batch])
    seeds = torch.tensor([ex['seed'] for ex in batch], dtype=torch.long)
    hs = torch.tensor([ex['h'] for ex in batch], dtype=torch.long)

    for i, ex in enumerate(batch):
        h = ex['h']
        context_obs[i, :h] = ex['context_obs']
        context_action[i, :h] = ex['context_action']
        context_reward[i, :h] = ex['context_reward']
        context_next_obs[i, :h] = ex['context_next_obs']
        context_mask[i, :h] = True

    return {
        'seed': seeds,
        'h': hs,
        'context_obs': context_obs,
        'context_action': context_action,
        'context_reward': context_reward,
        'context_next_obs': context_next_obs,
        'context_mask': context_mask,
        'query_obs': query_obs,
        'action_label': action_label,
    }
