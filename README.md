# CityLearn Universal DPT

An in-context-RL controller for centralized multi-building energy management on
[CityLearn 2023](https://www.citylearn.net/). A causal transformer (a Decision-Pretrained
Transformer, DPT, following Lee et al. 2023 and Berkes 2024's HVAC-DPT) is trained to predict
expert actions conditioned on a stream of interaction history, so that at deployment it adapts to
an unseen building/district **purely by conditioning on context — no gradient updates.**

## About this codebase

This repository was built with [Claude Code](https://claude.com/claude-code), Anthropic's agentic
coding CLI, working from a persistent project-instructions file, **[`CLAUDE.md`](CLAUDE.md)**.
That file is the actual specification Claude was given and held itself to throughout
development: the non-negotiable design decisions (e.g. CHESCA — not PPO — as the label oracle;
centralized control; discretized action head), the working agreement (validation gates as hard
blockers, verify-by-running rather than from memory), and a running log of what was tried, what
failed, and why, written as the project progressed rather than after the fact. It's kept in the
repo (lightly scrubbed of machine-specific paths/hostnames) as an honest record of the design
process, and it's worth reading if you want to understand *why* the code looks the way it does,
not just what it does.

This cleanup-and-release pass — reorganizing the repo, removing dead/exploratory code, scrubbing
identifying details, and writing this README — was itself done by Claude Code, working from a
reviewed and approved plan.

## Method summary

- **Labels** come from **CHESCA**, the winning classical-optimization algorithm from the 2023
  CityLearn Challenge (vendored under [`oracle/chesca_repo/`](oracle/chesca_repo/)) — never PPO,
  and never the diverse context-generating policies.
- **Context** (what the transformer conditions on) is deliberately diverse and kept separate from
  labels: random rollouts, a rule-based controller, CHESCA-with-noise, and the model's own
  self-play rollouts, mixed at a fixed ratio.
- **Model**: a causal transformer with a query-first token ordering, a discretized (binned)
  action head, and no positional encoding (context is treated as an order-robust set, and is
  reshuffled on every training example to enforce that).
- **Task distribution**: two distinct building/weather anchor families combined with five battery
  /DHW/heat-pump capacity multipliers (10 training tasks), evaluated zero-shot on a third,
  genuinely held-out anchor family at unseen capacities.
- **No gradient updates at deployment** — all adaptation happens by accumulating context and
  conditioning on it.

Measured results (8-KPI breakdown vs CHESCA and a rule-based baseline, the in-context learning
curve, and a runtime/latency comparison) are produced as JSON in `results/` by running the
`deploy/` scripts below — this repo ships the pipeline, not the pre-computed numbers.

## Repo structure

```
envs/        CityLearn schema/task sampling, conformed to a fixed (obs=52, action=9) layout
oracle/      vendored CHESCA (the expert label oracle) — see oracle/chesca_repo/LICENSE
model/       the DPT transformer + action discretizer
data/        dataset classes + context/label harvesting + normalizer fitting (see data/README.md)
train/       training loops (single-task reference trainer + the mix-ratio sweep that produced
             the deployed model)
deploy/      in-context evaluation, held-out KPI scoring vs CHESCA/RBC, runtime benchmarks
results/     (empty) where deploy/ scripts write measured KPI + runtime JSON results
figures/     (empty) where local figure-generation scripts would write plots
tests/       validation-gate sanity checks
configs/     yaml training configs
```

Everything under `checkpoints_locked/`, `checkpoints_hard_sweep/`, `data/`'s `.npz`/normalizer/
context/label caches, and `results/`/`figures/`'s generated contents is gitignored — only source
code, configs, and docs are tracked. Run the pipeline (see below) to regenerate all of it locally.

## Setup

```bash
conda create -n citylearn python=3.10 -y
conda activate citylearn
pip install -r requirements.txt
```

`citylearn` is pinned to `2.1b12` (not the latest release) because the vendored CHESCA code
targets citylearn's older pre-Gymnasium API — see `CLAUDE.md`'s Gate 1 notes for why. If you hit
`GLIBCXX`/`CXXABI` import errors from `pandas`/`xgboost` after activating the env, see the library
path fix documented in `CLAUDE.md`'s "Compute environment" section.

All scripts resolve paths relative to the repo root (`Path(__file__)`-based), so run them with the
repo root as your working directory, e.g. `python train/train_dpt.py`.

## Regenerating the gitignored artifacts

1. **Context + labels**: `python data/harvest_context_and_labels.py` (designed to run as an SGE
   array job, one task per anchor-family × capacity combination — see
   `data/harvest_context_and_labels.qsub`). Populates `data/context_hard/` and `data/labels_hard/`.
2. **Normalizer**: `python data/fit_normalizer.py` → `data/normalizer_hard.npz`.
3. **Checkpoints**: `python train/train_mixratio_sweep.py` trains the full r1–r4 mix-ratio family
   into `checkpoints_hard_sweep/`; `deploy/evaluate_and_report.py` extends/evaluates them and
   produces the results under `results/`.

## Known limitations

- **The in-context evaluation loop is duplicated**, not shared, across
  `deploy/evaluate_and_report.py`, `data/harvest_context_and_labels.py`, and
  `deploy/measure_runtime_dpt.py`. Consolidating it into one module is real refactor work with
  correctness risk (CHESCA rollouts are only bitwise-reproducible on the exact same compute-node
  hardware, per `CLAUDE.md`), so it was left duplicated-but-verified rather than refactored
  without re-verification. See `CLAUDE.md`'s "Known pitfalls" for detail.
- The composite `average_score` KPI can mask per-KPI tradeoffs — inspect the full 8-KPI breakdown
  that `deploy/evaluate_and_report.py` writes to `results/`, not just the headline average, before
  drawing conclusions from any given run.
- The two outage-conditional resilience KPIs (M, S) are only defined on episodes that actually
  contain an outage; with few eval seeds some may see zero outage steps, so interpret M/S with
  that sample-size caveat in mind.

## License

This project's own code is MIT-licensed — see [`LICENSE`](LICENSE). The vendored CHESCA code
under `oracle/chesca_repo/` is separately MIT-licensed by its original author; see
`oracle/chesca_repo/LICENSE`.
