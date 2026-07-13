"""
Gate 2a determinism check: reload the harvested labels and re-run CHESCA fresh for the
first N_CHECK_STEPS steps, asserting the live actions exactly match the previously saved ones.

Must be run with cwd = oracle/chesca_repo (so relative schema/module paths resolve), e.g.:
  cd oracle/chesca_repo && python ../../tests/test_gate2a_determinism.py

Must be run on a COMPUTE NODE, not the login node -- see CLAUDE.md's "known pitfalls" section on
login-node vs compute-node floating-point divergence for CHESCA rollouts.
"""
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

from citylearn.citylearn import CityLearnEnv
from agents.user_agent import SubmissionAgent
from rewards.user_reward import SubmissionReward
from local_evaluation import WrapperEnv

SCHEMA_NAME = 'citylearn_challenge_2023_phase_2_local_evaluation'
SCHEMA_PATH = os.path.join('data', 'schemas', SCHEMA_NAME, 'schema.json')
SEED = 73055
N_CHECK_STEPS = 50
NPZ_PATH = os.path.join(PROJECT_ROOT, 'data', 'labels', f'{SCHEMA_NAME}__seed{SEED}.npz')


def build_env_and_agent():
    env = CityLearnEnv(SCHEMA_PATH, reward_function=SubmissionReward, random_seed=SEED)
    env_data = dict(
        observation_names=env.observation_names,
        action_names=env.action_names,
        observation_space=env.observation_space,
        action_space=env.action_space,
        time_steps=env.time_steps,
        random_seed=None,
        episode_tracker=None,
        seconds_per_time_step=None,
        buildings_metadata=env.get_metadata()['buildings'],
    )
    wrapper_env = WrapperEnv(env_data)
    agent = SubmissionAgent(wrapper_env)
    return env, agent


def main():
    saved = np.load(NPZ_PATH)
    saved_actions = saved['action'][:N_CHECK_STEPS]

    env, agent = build_env_and_agent()
    obs = env.reset()
    actions = agent.register_reset(obs)

    live_actions = []
    for _ in range(N_CHECK_STEPS):
        live_actions.append(list(actions[0]))
        obs, reward, done, info = env.step(actions)
        assert not done, "Episode ended before reaching N_CHECK_STEPS"
        actions = agent.predict(obs)

    live_actions = np.asarray(live_actions, dtype=np.float64)

    identical = np.array_equal(saved_actions, live_actions)
    max_abs_diff = np.max(np.abs(saved_actions - live_actions))

    print(f"Checked {N_CHECK_STEPS} steps.")
    print(f"Exact match: {identical}")
    print(f"Max abs diff: {max_abs_diff}")

    if not identical:
        mismatch_steps = np.where(np.any(saved_actions != live_actions, axis=1))[0]
        print(f"Mismatching steps ({len(mismatch_steps)}): {mismatch_steps.tolist()}")

    assert identical, "Determinism check FAILED: live CHESCA actions do not exactly match saved labels"
    print("PASSED: saved actions exactly match live CHESCA actions.")


if __name__ == '__main__':
    main()
