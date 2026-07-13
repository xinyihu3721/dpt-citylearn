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
analysis/    # figure generation from results/
results/     # measured KPI/runtime JSON results (small, tracked in git)
figures/     # generated figures (tracked in git)
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

## Known pitfalls
- Varying building COUNT breaks the fixed-size concatenated input. Start with **fixed N**
  (vary which buildings fill slots); move to a permutation-invariant per-building encoder later.
- If CHESCA is receding-horizon, labeling an arbitrary query state needs its internal plan state
  reconstructed — don't silently label with a stale/blank plan.
- Perfect-foresight LP (if used) leaks future info -> reference only, never a training label source.
- Don't reward-hack the KPI normalization: always score against the same RBC baseline the challenge uses.
- CHESCA is the labeler, NOT the deployed policy. The DPT must eventually match/beat it in-context
  WITHOUT calling it at deployment.
- **The login node and compute nodes produce different floating-point results for the same
  CHESCA rollout, same seed, same code (found at Gate 2a).** A determinism check (rerun CHESCA
  fresh for 50 steps, compare to saved actions) matched exactly for steps 0-14 then diverged at
  ~1e-7 magnitude from step 15 on when the check ran on the login node, but matched **exactly**
  (max abs diff 0.0) when rerun as a batch job on a compute node. Root cause is almost certainly
  CPU microarchitecture/SIMD codepath differences between the login node and the compute nodes,
  not any real nondeterminism in CHESCA itself. Implication: **always run/verify CHESCA rollouts
  on a compute node, including quick "smoke test" style checks** -- do not trust bit-exact
  comparisons made on the login node, even for a handful of steps.
- **Several "different" 2023-challenge anchor schemas share byte-identical building/weather CSVs
  (found at Gate 5a) -- checksum before trusting an anchor as a genuine holdout.**
  Verified via `md5sum`: `citylearn_challenge_2023_phase_1` and `..._phase_2_local_evaluation`
  share the same `Building_*.csv`/`weather.csv`; separately, `..._phase_2_online_evaluation_1/2/3`
  AND all three `..._phase_3_1/2/3` variants share a SECOND identical set for buildings 1-3
  (`weather.csv` and `Building_1/2/3.csv` all match) -- they differ only in a baked-in
  `stochastic_power_outage_model` seed, and in phase_3's case, three EXTRA buildings (4/5/6, which
  ARE genuinely distinct -- verified separately). Picking a "held-out" anchor's first 3 buildings
  without checking this silently reuses already-trained-on building data. Also: forcing the same
  `update_power_outage_random_seed` override across nominally-different anchors that share
  underlying data collapses them into byte-identical rollouts -- use distinct override seeds per
  anchor if they otherwise share data.
- **Gate 5b pre-registered "context ordering mismatch" hypothesis was tested and REFUTED as the
  (sole) cause of the inverted in-context curve.** Diagnosis chain: (1) pre-5b minimal-ambiguity
  test showed a genuinely-ambiguous task (train acc dropped to 97.1%) still produced an inverted
  self-play curve; (2) pre-architecture triage found the model DOES use context correctly OFFLINE
  via teacher-forcing on training-distributed context (accuracy 0.671->0.791 as h: 0->256), ruling
  out "model never learned to use context" -- but ALSO found training samples context as an
  unordered random SUBSET (`rng.choice(..., replace=False)`, no sort) while deploy's ContextBuffer
  is a chronological FIFO, and even with no positional encoding the causal mask leaked real
  order-dependence (same 24 tokens, chronological vs shuffled order, max logit diff 0.178 -- not
  noise). (3) THIS ORDERING-FIX TEST: patched ONLY the deploy loop to randomly permute context
  before each query (matching training's unordered-set sampling), reran the self-play curve on
  the SAME held-out task with matched eval seeds ([55555, 1020, 1025] -- confirmed nonzero
  outages) -- **the shuffled curve was statistically indistinguishable from the chronological
  one** (both climb ~+0.20 from h=0 to h=256; e.g. h=256: chronological 0.882 vs shuffled 0.886).
  Conclusion: order-dependence is real (measurable at the logit level) but is NOT the operative
  mechanism driving the self-play cliff. Leading remaining suspect (from the same triage, not yet
  tested): self-play feeds the model's OWN bin-quantized, compounding actions back into context
  every step after the first, a state/action distribution training never showed it (training
  context actions are always continuous, from random/RBC/CHESCA+noise policies, never "the model
  imitating itself") -- classic exposure-bias / DAgger-style distribution shift.
  **Two corrections applied at the retrain (architectural, not deploy-loop):**
  (a) put the query token FIRST in the sequence (`[query; context...]`), not last -- per the DPT
  paper (Lee et al. 2023), this is the standard construction and interacts better with causal
  masking than query-last.
  (b) expose the model to VARIED context orderings during training (not just one fixed
  random-subset draw per example) so it is trained to be genuinely order-robust/posterior-
  sampling-consistent.
- **Both corrections fixed the self-play cliff. Operating model = r3** (40% exploratory / 30%
  chesca_noisy / 30% self-generated context mix), query-first architecture, checkpoint locked at
  `checkpoints_locked/r3_operating_model.pt` (gitignored; regenerate via
  `train/train_mixratio_sweep.py`, do not overwrite once produced). Held-out capacity-variant
  (1.1x) self-play performance: h=0 average_score=0.708, best=0.662 @ h=72, a monotonic
  in-context gain (unlike the pre-retrain model, which cliffed +0.20 from h=0 to h=256). This is
  the current DEPLOYABLE model and fallback.
  Mix-ratio sweep (exploratory% in {60,50,40,25}, chesca_noisy:selfplay held at 1:1 of the
  remainder) on an easier, single-anchor capacity task distribution found h=0 and best self-play
  score both improve monotonically as exploratory% DECREASES (r1 60%: h0=0.777 -> r4 25%:
  h0=0.641), while random/RBC-context robustness at h=256 barely moved (~0.82-0.83 regardless of
  ratio) -- i.e. on that task distribution, context was largely REDUNDANT (r4, the most
  self-play-heavy, dominated on raw score). Flagged as likely because that task distribution was
  too easy (a single anchor's capacity variants, query state alone may already be near-sufficient)
  -- motivated building the genuinely harder multi-anchor + capacity-randomized distribution
  (`envs/combined_task_sampler.py`) that the final r1-r4 sweep (see `results/gate6b_group1_results.json`)
  was run against.
- **The in-context evaluation loop (`ContextBuffer` + greedy/sampled decoding) is duplicated
  near-identically across `deploy/evaluate_and_report.py`, `data/harvest_context_and_labels.py`,
  and `deploy/measure_runtime_dpt.py`, rather than factored into one shared module.** This is a
  known, deliberate-for-now piece of tech debt: consolidating it is real refactor work with
  correctness risk (would need re-verification on a compute node per the login/compute-node
  float-divergence pitfall above), so it was left duplicated-but-working rather than refactored
  without re-verification during the cleanup-for-release pass. Flagged here rather than silently
  left inconsistent.
