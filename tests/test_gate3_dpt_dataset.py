"""
Gate 3 Step 4 test: DPTDataset + collate_dpt produce correctly-shaped, same-task,
normalized DPT training batches. Pure numpy/torch, no CHESCA/citylearn simulation --
run on a compute node anyway per instructions.

Note: DPTDataset.__getitem__ ignores its idx argument and draws from a single shared,
continuously-advancing RNG (each draw is a fresh random (context, query) pair, matching
the DPT training-example generation pattern, not a fixed per-index dataset). That means
calling ds[i] a second time does NOT reproduce an example already seen in a batch -- all
checks below work directly off the one batch object returned by the DataLoader.
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.dpt_dataset import (
    DPTDataset, collate_dpt, LABEL_FILES, NORMALIZER_PATH, context_file, CONTEXT_POLICIES,
)

OBS_DIM = 52
ACTION_DIM = 9
H_MAX = 256
BATCH_SIZE = 8


def main():
    ds = DPTDataset(h_max=H_MAX, epoch_size=1000, seed_rng=42)
    loader = DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=lambda b: collate_dpt(b, h_max=H_MAX),
    )
    batch = next(iter(loader))

    # --- shape/dim checks ---
    assert batch['context_obs'].shape == (BATCH_SIZE, H_MAX, OBS_DIM), batch['context_obs'].shape
    assert batch['context_action'].shape == (BATCH_SIZE, H_MAX, ACTION_DIM), batch['context_action'].shape
    assert batch['context_reward'].shape == (BATCH_SIZE, H_MAX)
    assert batch['context_next_obs'].shape == (BATCH_SIZE, H_MAX, OBS_DIM)
    assert batch['context_mask'].shape == (BATCH_SIZE, H_MAX)
    assert batch['query_obs'].shape == (BATCH_SIZE, OBS_DIM)
    assert batch['action_label'].shape == (BATCH_SIZE, ACTION_DIM)
    print("PASS: shapes/dims correct "
          f"(context_obs {batch['context_obs'].shape}, query_obs {batch['query_obs'].shape}, "
          f"action_label {batch['action_label'].shape})")

    # --- mask consistency: valid positions per example == that example's sampled context size h ---
    for i in range(BATCH_SIZE):
        assert batch['context_mask'][i].sum().item() == batch['h'][i].item()
        # padded region beyond h must be exactly zero
        h = batch['h'][i].item()
        assert torch.all(batch['context_obs'][i, h:] == 0)
        assert torch.all(batch['context_action'][i, h:] == 0)
    print("PASS: context_mask valid-count matches h, and padded region is zero")
    print(f"      sampled h values in this batch: {batch['h'].tolist()}")

    # --- same-task check: the query (obs, action) for each example is actually present in that
    # seed's CHESCA label file (a real (state, label) pair from that same task, not fabricated) ---
    norm = np.load(NORMALIZER_PATH)
    mean, std = norm['mean'], norm['std']
    label_obs_norm_cache = {}
    for i in range(BATCH_SIZE):
        seed = batch['seed'][i].item()
        if seed not in label_obs_norm_cache:
            labels = np.load(LABEL_FILES[seed])
            label_obs_norm_cache[seed] = ((labels['obs'] - mean) / std, labels['action'])
        label_obs_norm, label_actions = label_obs_norm_cache[seed]

        query_obs_np = batch['query_obs'][i].numpy()
        matches_obs = np.any(np.all(np.isclose(label_obs_norm, query_obs_np, atol=1e-4), axis=1))
        action_label_np = batch['action_label'][i].numpy()
        # action_label went through a float64 -> float32 cast in the dataset, so compare with
        # tolerance rather than exact equality (same reasoning as the obs isclose check above).
        matches_action = np.any(np.all(np.isclose(label_actions, action_label_np, atol=1e-4), axis=1))
        assert matches_obs, f"example {i}: query_obs not found in seed {seed}'s label file"
        assert matches_action, f"example {i}: action_label not found in seed {seed}'s label file"
    print("PASS: every query (obs, action) is a real pair drawn from its own task's CHESCA labels")

    # --- context-from-same-task check: each valid context_obs row is a real, normalized row from
    # THAT example's own seed's pooled context files (random/rbc/chesca_noisy), not another task's ---
    context_pool_norm_cache = {}
    for i in range(BATCH_SIZE):
        seed = batch['seed'][i].item()
        if seed not in context_pool_norm_cache:
            raw = np.concatenate([np.load(context_file(seed, p))['obs'] for p in CONTEXT_POLICIES], axis=0)
            context_pool_norm_cache[seed] = (raw - mean) / std
        pool_norm = context_pool_norm_cache[seed]

        h = batch['h'][i].item()
        for row in batch['context_obs'][i, :h].numpy():
            assert np.any(np.all(np.isclose(pool_norm, row, atol=1e-4), axis=1)), \
                f"example {i}: a context_obs row is not in seed {seed}'s own context pool"
    print("PASS: every valid context_obs row belongs to that example's own task's context pool")

    # --- normalization check: raw obs should clearly not already look normalized ---
    any_seed = batch['seed'][0].item()
    raw_pool = np.concatenate([np.load(context_file(any_seed, p))['obs'] for p in CONTEXT_POLICIES], axis=0)
    assert np.abs(raw_pool).max() > 10, "raw context obs looked already normalized -- suspicious"
    print("PASS: normalizer is actually doing something (raw obs are not already ~unit scale)")

    print("\nALL GATE 3 DATASET/COLLATOR TESTS PASSED")


if __name__ == '__main__':
    main()
