"""
Per-action-dim discretization for the DPT action head.

Bin centers are built from the env's ACTUAL action_space low/high per dim (NOT assumed [-1,1] --
e.g. dim 1/4 electrical_storage is [-0.83, 0.83], dim 7 electrical_storage is [-0.4878788,
0.4878788], dims 2/5/8 cooling_device are [0, 1], see gate4_check_action_bounds.py output).
B bin centers are evenly spaced via linspace(low, high, B) inclusive of both endpoints.
"""
import numpy as np


class ActionDiscretizer:
    def __init__(self, low, high, n_bins):
        self.low = np.asarray(low, dtype=np.float64)
        self.high = np.asarray(high, dtype=np.float64)
        self.n_bins = n_bins
        self.action_dim = self.low.shape[0]
        # bin_centers[d] has shape (n_bins,), evenly spaced from low[d] to high[d] inclusive
        self.bin_centers = np.stack(
            [np.linspace(self.low[d], self.high[d], n_bins) for d in range(self.action_dim)],
            axis=0,
        )

    def label_to_bin(self, labels):
        """labels: array (..., action_dim) continuous actions -> (..., action_dim) int bin indices,
        nearest bin center per dim."""
        labels = np.asarray(labels, dtype=np.float64)
        # (..., action_dim, 1) - (action_dim, n_bins) broadcasts to (..., action_dim, n_bins)
        diffs = np.abs(labels[..., None] - self.bin_centers[None, ...] if labels.ndim > 1
                       else labels[:, None] - self.bin_centers)
        return np.argmin(diffs, axis=-1)

    def bin_to_action(self, bin_idx):
        """bin_idx: array (..., action_dim) int -> (..., action_dim) float bin-center actions."""
        bin_idx = np.asarray(bin_idx)
        out = np.empty(bin_idx.shape, dtype=np.float64)
        for d in range(self.action_dim):
            out[..., d] = self.bin_centers[d][bin_idx[..., d]]
        return out

    def max_half_bin_width(self):
        return float(np.max((self.high - self.low) / (2 * (self.n_bins - 1))))
