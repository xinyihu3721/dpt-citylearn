# data/

Dataset code + regenerable data caches for DPT training.

## Code (tracked in git)
- `dpt_dataset.py` — `collate_dpt`, the shared batch collator (pads variable-length context to a
  common length with a validity mask).
- `dpt_dataset_hard.py` — `HardTaskDPTDataset`, the mixed-context dataset over the current
  anchor-family x capacity task distribution (used by `train/train_mixratio_sweep.py` and
  `deploy/evaluate_and_report.py`).
- `fit_normalizer.py` — fits the observation normalizer used by the hard-task pipeline
  (`normalizer_hard.npz`/`.json`) by pooling observations across all training tasks and sources.
- `harvest_context_and_labels.py` — per-task harvester: rolls out CHESCA (clean labels),
  random/RBC (exploratory context), CHESCA+noise, and self-play from the locked r3 checkpoint,
  saving each as an `.npz` rollout. One task per SGE array index (`harvest_context_and_labels.qsub`).

## Data caches (gitignored, NOT in this repo — regenerate locally)
- `context_hard/`, `labels_hard/` — hard-task-distribution context/label rollouts.
- `normalizer_hard.npz`/`.json` — fitted observation normalizer.

To regenerate: run `harvest_context_and_labels.py` (via `harvest_context_and_labels.qsub` as an
SGE array job, one task per anchor-family x capacity combination) to populate `context_hard/` and
`labels_hard/`, then `fit_normalizer.py` to produce `normalizer_hard.npz`. See the top-level
README.md for the full pipeline order.
