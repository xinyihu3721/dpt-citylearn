"""
Part 2: genuinely HARD task distribution -- combines anchor identity (real distinct
building/weather data across the checksum-verified Gate 5a families) WITH capacity
randomization (battery/DHW/heat-pump +/-30%, scaling nominal_power WITH capacity so action
bounds stay identical -- same fix as the pre-5b capacity test) on top, so tasks differ in BOTH
load/weather shape AND equipment sizing.

Anchor families (per Gate 5a's checksum verification, recorded in CLAUDE.md):
  Family A = citylearn_challenge_2023_phase_1 (== phase_2_local_evaluation, byte-identical data;
             using phase_1 as the one representative to avoid checksum-duplicate "tasks")
  Family B = citylearn_challenge_2023_phase_2_online_evaluation_1 (== _2, _3, and phase_3's
             buildings 1-3, byte-identical; using online_evaluation_1 as the one representative)
  Family C (HELD OUT, whole anchor) = citylearn_challenge_2023_phase_3_1, buildings 4/5/6
             (genuinely distinct, never used in A or B)

Training tasks = {A, B} x capacity multipliers {0.7, 0.85, 1.0, 1.15, 1.3} = 10 genuinely
distinct tasks (2 distinct anchors, each combined with 5 distinct physical parameterizations --
no checksum duplication possible since capacity scaling is a live schema mutation, not a raw
file reuse).

Held-out task = Family C x an UNSEEN capacity multiplier (1.075, interior to the training grid
but never trained on) -- tests generalization along BOTH the anchor axis and the capacity axis
simultaneously.
"""
import copy
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
CHESCA_REPO = os.path.join(PROJECT_ROOT, 'oracle', 'chesca_repo')

from envs.task_sampler import load_conformed_schema

FAMILY_A = ('A', 'data/schemas/citylearn_challenge_2023_phase_1/schema.json', None)
FAMILY_B = ('B', 'data/schemas/citylearn_challenge_2023_phase_2_online_evaluation_1/schema.json', None)
FAMILY_C_HELDOUT = ('C', 'data/schemas/citylearn_challenge_2023_phase_3_1/schema.json', (3, 4, 5))

TRAIN_FAMILIES = [FAMILY_A, FAMILY_B]
TRAIN_CAPACITY_MULTIPLIERS = [0.7, 0.85, 1.0, 1.15, 1.3]
HELD_OUT_CAPACITY_MULTIPLIER = 1.075  # interior to the training grid, never trained on

OUTAGE_SEED = 73055  # task identity = (anchor, capacity) only, not outage variation (as before)


def build_combined_schema(rel_path, building_subset, capacity_multiplier):
    schema = load_conformed_schema(rel_path, building_subset)
    for b in schema['buildings'].values():
        b['electrical_storage']['attributes']['capacity'] *= capacity_multiplier
        b['electrical_storage']['attributes']['nominal_power'] *= capacity_multiplier
        b['dhw_storage']['attributes']['capacity'] *= capacity_multiplier
        b['cooling_device']['attributes']['nominal_power'] *= capacity_multiplier
    return schema


def build_env(rel_path, building_subset, capacity_multiplier, reward_function, outage_seed=OUTAGE_SEED):
    sys.path.insert(0, CHESCA_REPO)
    from citylearn.citylearn import CityLearnEnv
    from local_evaluation import update_power_outage_random_seed
    schema = build_combined_schema(rel_path, building_subset, capacity_multiplier)
    env = CityLearnEnv(schema, reward_function=reward_function, random_seed=outage_seed)
    env = update_power_outage_random_seed(env, outage_seed)
    return env


def task_name(family_label, capacity_multiplier):
    return f'family{family_label}_cap{capacity_multiplier:.3f}x'


TRAIN_TASKS = [
    (label, rel_path, subset, m)
    for (label, rel_path, subset) in TRAIN_FAMILIES
    for m in TRAIN_CAPACITY_MULTIPLIERS
]  # 2 families x 5 capacities = 10 genuinely distinct training tasks

HELD_OUT_TASK = (FAMILY_C_HELDOUT[0], FAMILY_C_HELDOUT[1], FAMILY_C_HELDOUT[2], HELD_OUT_CAPACITY_MULTIPLIER)
