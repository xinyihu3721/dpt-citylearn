"""
Gate 4 Step 2: round-trip test for the action discretizer (B=21 bins per dim) against the
real Gate 2a CHESCA on-policy labels. Bin edges come from the env's ACTUAL action_space
low/high (not assumed [-1,1]).
"""
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHESCA_REPO = os.path.join(PROJECT_ROOT, 'oracle', 'chesca_repo')
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, CHESCA_REPO)
os.chdir(CHESCA_REPO)

from citylearn.citylearn import CityLearnEnv
from rewards.user_reward import SubmissionReward
from model.discretize import ActionDiscretizer

SCHEMA = os.path.join('data', 'schemas', 'citylearn_challenge_2023_phase_2_local_evaluation', 'schema.json')
N_BINS = 21
LABEL_PATH = os.path.join(PROJECT_ROOT, 'data', 'labels', 'citylearn_challenge_2023_phase_2_local_evaluation__seed73055.npz')


def main():
    env = CityLearnEnv(SCHEMA, reward_function=SubmissionReward)
    low = env.action_space[0].low
    high = env.action_space[0].high
    names = env.action_names[0]

    print("Action bounds used for discretization (NOT assumed [-1,1]):")
    for i in range(len(low)):
        print(f"  dim {i} ({names[i]}): low={low[i]:.7f} high={high[i]:.7f}")

    disc = ActionDiscretizer(low, high, N_BINS)
    print(f"\nmax half-bin-width across all dims: {disc.max_half_bin_width():.6f}")

    labels = np.load(LABEL_PATH)['action']
    bin_idx = disc.label_to_bin(labels)
    recon = disc.bin_to_action(bin_idx)

    assert bin_idx.shape == labels.shape
    assert bin_idx.min() >= 0 and bin_idx.max() <= N_BINS - 1

    abs_err = np.abs(recon - labels)
    max_err = abs_err.max()
    per_dim_max_err = abs_err.max(axis=0)
    tol = disc.max_half_bin_width() + 1e-9

    print(f"\nn_labels checked: {labels.shape[0]}")
    print(f"per-dim max reconstruction error: {per_dim_max_err}")
    print(f"overall max reconstruction error: {max_err:.6f} (tolerance: {tol:.6f})")

    assert max_err <= tol, f"round-trip error {max_err} exceeds half-bin-width tolerance {tol}"
    print("\nPASSED: label -> bin -> action round-trip within half-bin-width for all labels")


if __name__ == '__main__':
    main()
