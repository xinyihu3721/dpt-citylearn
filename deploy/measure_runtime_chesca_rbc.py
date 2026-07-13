"""
Runtime-per-prediction measurement: CHESCA and RBC (CPU) on the SAME held-out task as the DPT
measurement (Family C, capacity=1.075x, seed=55555), one full episode each.

Per-step timing brackets ONLY the per-decision optimization/rule call (agent.predict / RBC.predict),
not env.step() -- consistent with "time each per-step optimization call" in the ask. First
WARMUP_STEPS steps are discarded from the reported mean/std (agent-internal lazy init / first-call
overhead) but the full per-step series is saved so discarded steps remain inspectable.

Must run on a COMPUTE NODE (per CLAUDE.md: login-node vs compute-node floating-point divergence for
CHESCA rollouts was found at Gate 2a -- timing on the login node would also be unrepresentative of real
deployment hardware). Submitted via qsub (see measure_runtime_chesca_rbc.qsub).
"""
import json
import os
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHESCA_REPO = os.path.join(PROJECT_ROOT, 'oracle', 'chesca_repo')
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, CHESCA_REPO)
os.chdir(CHESCA_REPO)

from envs.combined_task_sampler import FAMILY_C_HELDOUT, build_env

OUT_PATH = os.path.join(PROJECT_ROOT, 'results', 'results_runtime_chesca_rbc.json')
CAPACITY = 1.075
SEED = 55555
WARMUP_STEPS = 5


def run_chesca_timed():
    from rewards.user_reward import SubmissionReward
    from agents.user_agent import SubmissionAgent
    from local_evaluation import WrapperEnv
    _, rel_path, subset = FAMILY_C_HELDOUT
    env = build_env(rel_path, subset, CAPACITY, SubmissionReward, outage_seed=SEED)
    env_data = dict(observation_names=env.observation_names, action_names=env.action_names,
                     observation_space=env.observation_space, action_space=env.action_space,
                     time_steps=env.time_steps, random_seed=None, episode_tracker=None,
                     seconds_per_time_step=None, buildings_metadata=env.get_metadata()['buildings'])
    obs = env.reset()

    t0 = time.perf_counter()
    agent = SubmissionAgent(WrapperEnv(env_data))
    actions = agent.register_reset(obs)
    t1 = time.perf_counter()
    register_reset_s = t1 - t0

    step_times = []
    done = False
    while not done:
        obs, reward, done, info = env.step(actions)
        if not done:
            t0 = time.perf_counter()
            actions = agent.predict(obs)
            t1 = time.perf_counter()
            step_times.append(t1 - t0)
    return np.array(step_times), register_reset_s


def run_rbc_timed():
    from rewards.user_reward import SubmissionReward
    from citylearn.agents.rbc import BasicRBC
    _, rel_path, subset = FAMILY_C_HELDOUT
    env = build_env(rel_path, subset, CAPACITY, SubmissionReward, outage_seed=SEED)
    agent = BasicRBC(env)
    obs = env.reset()
    done = False
    step_times = []
    while not done:
        t0 = time.perf_counter()
        actions = agent.predict(obs)
        t1 = time.perf_counter()
        step_times.append(t1 - t0)
        obs, reward, done, info = env.step(actions)
    return np.array(step_times)


def summarize(name, step_times, extra=None):
    kept = step_times[WARMUP_STEPS:]
    out = {
        'controller': name,
        'n_steps': int(len(step_times)),
        'warmup_steps_discarded': WARMUP_STEPS,
        'mean_ms': float(kept.mean() * 1000),
        'std_ms': float(kept.std() * 1000),
        'total_episode_wallclock_s': float(step_times.sum()),
        'per_step_times_ms': (step_times * 1000).tolist(),
    }
    if extra:
        out.update(extra)
    print(f"[{name}] n_steps={out['n_steps']} mean={out['mean_ms']:.3f}ms std={out['std_ms']:.3f}ms "
          f"total={out['total_episode_wallclock_s']:.2f}s (first {WARMUP_STEPS} steps discarded)")
    return out


def main():
    nproc = os.cpu_count()
    print(f"os.cpu_count()={nproc}, NSLOTS={os.environ.get('NSLOTS')}")

    chesca_times, register_reset_s = run_chesca_timed()
    chesca_out = summarize('chesca', chesca_times, extra={'register_reset_s': register_reset_s})

    rbc_times = run_rbc_timed()
    rbc_out = summarize('rbc', rbc_times)

    results = {
        'task': 'familyC_cap1.075x',
        'seed': SEED,
        'cpu_count': nproc,
        'nslots': os.environ.get('NSLOTS'),
        'chesca': chesca_out,
        'rbc': rbc_out,
    }
    tmp = OUT_PATH + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(results, f, indent=2)
    os.replace(tmp, OUT_PATH)
    print(f"Saved: {OUT_PATH}")


if __name__ == '__main__':
    main()
