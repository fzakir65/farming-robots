"""Train the pickers' shared task-choice policy with MaskablePPO (multi-agent RL).

Usage (from ``src/``):
    python train_rl.py                       # 150k decisions, saves ../models/picker_policy.zip
    python train_rl.py --timesteps 20000     # quick run
    python train_rl.py --check               # only validate the PettingZoo / Gymnasium envs

The learning curve (mean episode reward per rollout) is written to
``../results/advanced/rl_training.csv``. See ``farm/rl.py`` for the observation,
actions and reward.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

from farm.constants import BROADCAST, EXTENDED
from farm.rl import make_aec_env, make_gym_env

ENV_KWARGS = dict(n_pickers=4, max_steps=600, mode=EXTENDED, comm_mode=BROADCAST)


def check_envs():
    """Run the official API checks of PettingZoo and Stable-Baselines3."""
    from pettingzoo.test import api_test
    from stable_baselines3.common.env_checker import check_env

    aec = make_aec_env(**dict(ENV_KWARGS, max_steps=150))
    api_test(aec, num_cycles=200)
    aec.close()
    gym_env = make_gym_env(**dict(ENV_KWARGS, max_steps=150))
    check_env(gym_env)
    gym_env.close()
    print("PettingZoo api_test and SB3 check_env passed")


def train(timesteps, out, seed=0):
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor

    class Curve(BaseCallback):
        def __init__(self):
            super().__init__()
            self.rows = []

        def _on_rollout_end(self):
            eps = [e["r"] for e in self.model.ep_info_buffer]
            if eps:
                self.rows.append({"timesteps": self.num_timesteps, "mean_episode_reward": float(np.mean(eps)),
                                  "episodes": len(eps)})

        def _on_step(self):
            return True

    env = Monitor(make_gym_env(**ENV_KWARGS))
    model = MaskablePPO(
        "MlpPolicy", env, seed=seed, verbose=0, n_steps=1024, batch_size=256, learning_rate=3e-4,
        gamma=0.99, ent_coef=0.01, policy_kwargs={"net_arch": [64, 64]}, device="cpu",
    )
    curve = Curve()
    t0 = time.time()
    model.learn(total_timesteps=timesteps, callback=curve)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    model.save(out)
    env.close()
    results = os.path.join("..", "results", "advanced")
    os.makedirs(results, exist_ok=True)
    pd.DataFrame(curve.rows).to_csv(os.path.join(results, "rl_training.csv"), index=False)
    print(f"trained {timesteps} decisions in {time.time() - t0:.0f}s -> {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--timesteps", type=int, default=150_000)
    parser.add_argument("--out", default=os.path.join("..", "models", "picker_policy.zip"))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    check_envs()
    if not args.check:
        train(args.timesteps, args.out)


if __name__ == "__main__":
    main()
