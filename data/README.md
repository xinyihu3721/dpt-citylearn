# data/

Dataset code + regenerable data caches for DPT training.

## Code (tracked in git)
- `dpt_dataset.py` — Gate-3 single-task `DPTDataset`/`collate_dpt` (used by `train/train_dpt.py`).
- `dpt_dataset_hard.py` — `HardTaskDPTDataset`, the mixed-context dataset over the current
  anchor-family x capacity task distribution (used by `train/train_mixratio_sweep.py` and
  `deploy/evaluate_and_report.py`).
- `fit_normalizer.py` — fits the observation normalizer used by the hard-task pipeline
  (`normalizer_hard.npz`/`.json`) by pooling observations across all training tasks and sources.
- `harvest_context_and_labels.py` — per-task harvester: rolls out CHESCA (clean labels),
  random/RBC (exploratory context), CHESCA+noise, and self-play from the locked r3 checkpoint,
  saving each as an `.npz` rollout. One task per SGE array index (`harvest_context_and_labels.qsub`).

## Data caches (gitignored, NOT in this repo — regenerate locally)
- `context/`, `labels/` — Gate-3 single-task context/label rollouts.
- `context_hard/`, `labels_hard/` — hard-task-distribution context/label rollouts.
- `normalizer*.npz`/`.json` — fitted observation normalizers (one per task-distribution version).

To regenerate: run `harvest_context_and_labels.py` (via `harvest_context_and_labels.qsub` as an
SGE array job, one task per anchor-family x capacity combination) to populate `context_hard/` and
`labels_hard/`, then `fit_normalizer.py` to produce `normalizer_hard.npz`. See the top-level
README.md for the full pipeline order.
