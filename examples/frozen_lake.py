# -*- coding: utf-8 -*-

import gymnasium as gym
from algorithms.rl import RL
from algorithms.planner import Planner
from examples.test_env import TestEnv
from examples.plots import Plots
import numpy as np
from gymnasium.envs.toy_text.frozen_lake import generate_random_map


class FrozenLake:
    def __init__(self, size):
        self.env = gym.make('FrozenLake-v1', render_mode=None, is_slippery=True, desc=generate_random_map(size=size))


if __name__ == "__main__":
    frozen_lake = FrozenLake(50)

    # VI/PI
    # V, V_track, pi = Planner(frozen_lake.env.P).value_iteration()
    # V, V_track, pi = Planner(frozen_lake.env.P).policy_iteration()

    # Q-learning
    for gamma in [0, 0.25, 0.5, 0.75, 1]:
        Q, V, pi, Q_track, V_track, pi_track = RL(frozen_lake.env).q_learning(gamma=gamma, n_episodes=300000)
        Plots.grid_world_policy_plot(
            np.array(list(map(lambda x: pi(x, None), range(frozen_lake.env.observation_space.n)))), gamma
        )

        test_scores = TestEnv.test_env(env=frozen_lake.env, render=True, user_input=False, pi=pi, n_iters=1)
