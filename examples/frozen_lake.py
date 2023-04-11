# -*- coding: utf-8 -*-

import gymnasium as gym
from algorithms.rl import RL
from algorithms.planner import Planner
from examples.test_env import TestEnv
from examples.plots import Plots
import numpy as np
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from itertools import repeat, product


class FrozenLake:
    def __init__(self, size):
        self.env = gym.make('FrozenLake-v1', render_mode=None, is_slippery=True, desc=generate_random_map(size=size))


def ql(frozen_lake, gamma):
    Q, V, pi, Q_track, V_track, pi_track = RL(frozen_lake.env).q_learning(
        gamma=gamma, n_episodes=300000  # , init_alpha=1., min_alpha=0.5, min_epsilon=0.8
    )
    new_pi = list(map(lambda x: pi(x, None), range(frozen_lake.env.observation_space.n)))
    Plots.grid_world_policy_plot(np.array(new_pi), gamma)
    Plots.grid_values_heat_map(V, gamma, False)
    Plots.v_iters_plot(np.amax(np.amax(Q_track, axis=2), axis=1), str(gamma), gamma)
    # test_scores = TestEnv.test_env(env=frozen_lake.env, render=True, user_input=False, pi=pi, n_iters=1)


if __name__ == "__main__":
    frozen_lake = FrozenLake(20)

    # VI/PI
    # V, V_track, pi = Planner(frozen_lake.env.P).value_iteration()
    # V, V_track, pi = Planner(frozen_lake.env.P).policy_iteration()

    # Q-learning
    gammas = [0, 0.25, 0.5, 0.75, 1]
    for g in gammas:
        ql(frozen_lake, g)
