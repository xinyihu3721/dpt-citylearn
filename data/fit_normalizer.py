"""
Part 2, step 2: fit observation normalizer over pooled hard-task data (10 training tasks x
5 sources: chesca_label, random, rbc, chesca_noisy, selfplay_greedy, selfplay_sampled).
Pure numpy/file I/O, safe on login node.
"""
import json
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from envs.combined_task_sampler import TRAIN_TASKS, task_name

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
LABELS_DIR = os.path.join(DATA_DIR, 'labels_hard')
CONTEXT_DIR = os.path.join(DATA_DIR, 'context_hard')
OUT_PATH = os.path.join(DATA_DIR, 'normalizer_hard.npz')
STD_FLOOR = 1e-6


def main():
    all_obs = []
    per_file_counts = {}

    for label, rel_path, subset, m in TRAIN_TASKS:
        name = task_name(label, m)
        for policy, d in [('chesca_label', LABELS_DIR), ('random', CONTEXT_DIR), ('rbc', CONTEXT_DIR),
                           ('chesca_noisy', CONTEXT_DIR), ('selfplay_greedy', CONTEXT_DIR),
                           ('selfplay_sampled', CONTEXT_DIR)]:
            path = os.path.join(d, f'{name}__{policy}.npz')
            data = np.load(path)
            all_obs.append(data['obs'])
            per_file_counts[os.path.basename(path)] = int(data['obs'].shape[0])

    pooled = np.concatenate(all_obs, axis=0)
    mean = pooled.mean(axis=0)
    std = pooled.std(axis=0)
    n_floored = int((std < STD_FLOOR).sum())
    std = np.maximum(std, STD_FLOOR)

    np.savez(OUT_PATH, mean=mean, std=std)
    sidecar = {
        'obs_dim': int(pooled.shape[1]), 'n_pooled_transitions': int(pooled.shape[0]),
        'n_source_files': len(per_file_counts), 'n_dims_floored_std': n_floored,
        'n_train_tasks': len(TRAIN_TASKS), 'per_file_counts': per_file_counts,
    }
    with open(OUT_PATH.replace('.npz', '.json'), 'w') as f:
        json.dump(sidecar, f, indent=2)

    print(f"Pooled {pooled.shape[0]} observations from {len(per_file_counts)} files, obs_dim={pooled.shape[1]}")
    print(f"n_train_tasks: {len(TRAIN_TASKS)}")
    print(f"Saved: {OUT_PATH}")


if __name__ == '__main__':
    main()
