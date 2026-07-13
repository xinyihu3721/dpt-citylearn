# CityLearn Universal DPT — Project Brief

> This file is the persistent project-instructions file used to steer Claude Code throughout this
> project's development. It defines the non-negotiable design decisions, working agreement, and
> validation-gate roadmap that guided every session. See `README.md` for how this fits into the
> repo as a whole; this file is kept (lightly edited to remove machine-specific/identifying details)
> as a record of the actual design process and decisions, not retrofitted after the fact.

## Goal
Build a **universal Decision-Pretrained Transformer (DPT)** that does in-context RL for
**centralized (single-agent) multi-building energy control** on the CityLearn 2023 environment.
At deployment it adapts to an unseen district **without any gradient updates** — purely by
conditioning on interaction history. Based on Lee et al. 2023 (DPT) and Berkes 2024 (HVAC-DPT).

## NON-NEGOTIABLE DESIGN DECISIONS (do not "simplify" these away)
1. **Two separate data streams — keep them distinct:**
   - *Context stream* `D = {(s,a,r,s')}`: read by the transformer to infer the task.
     Must be **diverse / exploratory** (random, RBC, CHESCA-with-noise, partial-SAC checkpoints).
   - *Label stream* `a*(s_query)`: what the model is trained to predict.
     Must come from the **CHESCA expert** (see below), NOT from the diverse agents.
2. **The optimal-policy oracle is CHESCA (winning 2023 CityLearn algorithm), NOT PPO.**
   - Primary labeler: **CHESCA** — hierarchical classical optimization (building-level opt, then a
     community-level controller for grid KPIs). Fast inference, arbitrary building count, strong
     generalization. Vendored in `oracle/chesca_repo/` (upstream: CHESCA, TheLeprechaun25).
   - Optional reference upper bound: perfect-foresight LP/QP over the full episode, only to measure
     CHESCA's optimality gap. Do NOT train the DPT on perfect-foresight labels.
   - Rationale: no top team used RL; winners used classical optimization. CHESCA is the proven expert.
   - **Never use PPO as the oracle.** SAC only as a diversity source for CONTEXT, never for labels.
3. **Centralized control:** always `central_agent=True` (one obs vector, one action vector,
   one reward for the whole district).
4. **Action head:** start with **discretized bins per action dim** (stable), Gaussian head later.
5. **Building metadata:** provide a per-building **descriptor token** of *physical, normalized*
   quantities (capacities, COP params, PV size, load scale, comfort band, climate embedding).
   NEVER building-identity one-hots. Apply **descriptor dropout** in pretraining so the model
   can still infer everything from context alone.

## CHESCA — the expert oracle for DPT labels
- What it is: Community-based Hierarchical Energy Systems Coordination Algorithm; winner of the
  2023 CityLearn Challenge (Garmendia, Morri, Cappart, Le Cadre, ECAI 2024). Classical
  optimization + heuristics, NOT RL. Interpretable, fast, generalizes with minimal data.
- Its role here: **generate the optimal-action labels** `a*` for the DPT.
- Labeling procedure: roll CHESCA out on each task; each visited `(state_t, CHESCA_action_t)`
  becomes a `(query_state, label)` pair. Collect the *context* stream separately from diverse
  policies. To broaden query coverage, also evaluate CHESCA's action on states drawn from the
  diverse rollouts (place the env in that state/time and ask CHESCA for the action).
- MUST resolve when reading the code: is CHESCA **Markov** (action = f(current obs)) or
  **receding-horizon / stateful** (keeps an internal plan)? If stateful, labeling an arbitrary
  query state requires reconstructing its internal state (time index + forecasts). Handle explicitly.
- Sanity check: run CHESCA unmodified and reproduce its reported winning-level score before trusting
  it as a labeler.

## Environment facts (verify against installed version at Gate 0)
- Package: `citylearn` (Farama Gymnasium interface). Confirm installed version and pin it.
- Construct: `CityLearnEnv(schema='citylearn_challenge_2023_phase_2', central_agent=True)`.
- Wrap with `NormalizedObservationWrapper`; add `StableBaselines3Wrapper` only for the SAC context source.
- Actions per building: **electrical (battery) storage, DHW storage, heat-pump power** (~[-1,1]).
- Observations: time encodings, weather + forecasts, carbon intensity, pricing,
  per-building load / PV / SoC / net consumption.
- Datasets as task anchors: `citylearn_challenge_2023_phase_1/2/3`, plus `2022`/`2021` families and
  `baeda_3dem`. Use `phase_2/3` for **power outages** (resilience KPIs).
- Control score = mean of 8 normalized KPIs (lower better), normalized vs a baseline RBC:
  carbon G, discomfort U, ramping R, 1-load-factor L, daily peak P_d, all-time peak P_n,
  1-thermal-resilience M, normalized unserved energy S.

## Compute environment
Developed on a remote SGE-scheduled HPC cluster via login/SSH, with a separate GPU-enabled compute
partition for training. NOTE: "CHESCA" is NOT the cluster — it is our expert algorithm (see above).
Python dependencies are pinned in `requirements.txt`; activate whatever conda/venv environment you
use for this project before running anything here.

**Library path note (found while checking CHESCA's deps):** on some systems, a conda env's own
newer `libstdc++.so.6` (required by `pandas`/`xgboost`) is not on the default library search
path, which can leave `pandas`/`xgboost` broken with `GLIBCXX_3.4.29 not found` /
`CXXABI_1.3.15 not found` errors if the system's own older `libstdc++.so.6` is found first.
Loading a newer `gcc` module does not reliably fix this (module search order can still win). Fix:
prepend your conda env's own `lib/` dir to `LD_LIBRARY_PATH` for any command that touches
pandas/xgboost (i.e. anything importing CHESCA).

General HPC lessons that shaped this project's structure (adapt to your own scheduler/cluster):
- **Login node = setup only. Compute node = the work.** Submit batch jobs for training and sweeps.
- **Compute nodes often have NO internet.** Do installs/downloads on the login node and pre-stage.
- **Label generation (running CHESCA over the task distribution) is embarrassingly parallel** → use a
  **job array** (one task per building-family/capacity combination). This is what makes label
  acquisition fast — see `data/harvest_context_and_labels.qsub` for the pattern used here.
- Transformer pretraining → GPU batch job, **resumable from checkpoint** (wall-time limits interrupt runs).
- Everything driven by a config + submit script committed to the repo; reproducible by re-submitting.

## Repo layout (as built)
```
envs/        # CityLearn setup, task/schema sampling, KPI scoring
oracle/      # vendored CHESCA (label generator); see oracle/chesca_repo/LICENSE
model/       # DPT transformer, tokenizer/embeddings, action head
data/        # dataset classes + rollout/label harvesting + normalizer fitting
train/       # pretraining loops + configs
deploy/      # in-context evaluation, held-out scoring vs CHESCA/RBC, runtime benchmarks
results/     # (empty, gitignored) where deploy/ scripts write measured KPI/runtime JSON results
figures/     # (empty, gitignored) where local figure-generation scripts would write plots
tests/       # sanity checks + validation-gate checks
configs/     # yaml configs; everything reproducible + seeded
checkpoints_locked/  # the one documented deployable checkpoint (gitignored, not in the repo)
```

## WORKING AGREEMENT (enforce every session)
- **Validation gates are hard blockers.** Do not build stage N+1 until stage N's gate passes.
- Work in **small, testable increments**; write a sanity test with each module.
- **Verify the CityLearn API and the CHESCA interface by running them**, not from memory.
- Pin dependencies; seed all RNGs; make every run reproducible from a config file.
- Prefer clear, typed, documented code. No silent fallbacks that mask failure.
- When a design choice conflicts with this file, STOP and ask rather than diverging.

## VALIDATION GATES (roadmap)
- **Gate 0 — Env spike:** print obs/action shapes for `central_agent=True` on 2023 phase_2;
  run a random policy for a full episode; compute the official 8-KPI score vs the built-in RBC.
- **Gate 0.5 — Cluster infra check:** detect scheduler, GPU partitions, modules, scratch paths,
  and compute-node internet (see Compute environment); confirm a trivial batch job and a job
  array both run.
- **Gate 1 — CHESCA online:** read the CHESCA repo; run it UNMODIFIED on
  citylearn_challenge_2023_phase_2 and reproduce its reported winning-level control score vs RBC.
  **PASSED (compute-node batch job, ~84s wall-clock):** on schema
  `citylearn_challenge_2023_phase_2_local_evaluation` (30-day/720-step local eval subset),
  CHESCA `average_score = 0.475` vs built-in `BasicRBC` baseline `average_score = 1.023`
  (lower is better) — CHESCA scores under half of RBC, consistent with winning-level.
  Required fixing two environment issues first, both resolved without installing anything new
  except the deliberate citylearn downgrade: (1) `LD_LIBRARY_PATH` needed the citylearn conda
  env's own `lib/` prepended for `pandas`/`xgboost` to import (see Compute environment section);
  (2) citylearn had to be downgraded from 2.3.1 to **2.1b12** (`pip install citylearn==2.1b12`,
  run on login node) because CHESCA's code (`reset()`/`step()` tuple shapes,
  `evaluate_citylearn_challenge()`) targets the old pre-Gymnasium API which 2.3.1 removed/changed.
  **This means citylearn is now pinned to 2.1b12 project-wide — Gate 0's results were against
  2.3.1 and have NOT been re-verified against 2.1b12.**
- **Gate 2 — CHESCA as labeler:** wrap CHESCA so (task + query state) -> expert action; emit
  (query_state, CHESCA_action) pairs. Resolve Markov vs receding-horizon and handle internal state.
- **Gate 3 — Data pipeline:** produce (diverse context, CHESCA labels) tensors for one district;
  verify shapes, normalization, and that stored labels match CHESCA actions on held-out query states.
- **Gate 4 — Single-task DPT:** train on one district; show in-context improvement over a
  random-context baseline on held-out rollouts.
- **Gate 5 — Task distribution:** procedural randomization + multi-dataset; descriptor tokens with
  dropout; zero-shot on held-out buildings/climates vs CHESCA and RBC.
- **Gate 6 — Resilience:** exercise outage windows (phase_2/3); report M, S.

