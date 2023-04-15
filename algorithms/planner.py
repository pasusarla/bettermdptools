import numpy as np
import warnings


"""
Author: Miguel Morales
BSD 3-Clause License

Copyright (c) 2018, Miguel Morales
All rights reserved.
https://github.com/mimoralea/gdrl/blob/master/LICENSE
"""

"""
modified by: John Mansfield

documentation added by: Gagandeep Randhawa
"""

"""
Model-based learning algorithms: Value Iteration and Policy Iteration

Assumes prior knowledge of the type of reward available to the agent
for iterating to an optimal policy and reward value for a given MDP.
"""


class Planner:
    def __init__(self, P):
        self.P = P

    def value_iteration(self, gamma=1.0, n_iters=1000, theta=1e-10):
        """
        PARAMETERS:

        gamma {float}:
            Discount factor

        n_iters {int}:
            Number of iterations

        theta {float}:
            Convergence criterion for value iteration.
            State values are considered to be converged when the maximum difference between new and previous state
             values is less than theta.
            Stops at n_iters or theta convergence - whichever comes first.


        RETURNS:

        V {numpy array}, shape(possible states):
            State values array

        V_track {numpy array}, shape(n_episodes, nS):
            Log of V(s) for each iteration

        pi {lambda}, input state value, output action value:
            Policy mapping states to actions.
        """
        def pi(_s, _Q):
            policy = dict()
            for state, action in enumerate(np.argmax(_Q, axis=1)):
                policy[state] = action

            return policy[_s]

        V = np.zeros(len(self.P), dtype=np.float64)
        V_track = np.zeros((n_iters, len(self.P)), dtype=np.float64)
        pi_track = []

        i = 0
        converged = False
        while i < n_iters - 1 and not converged:
            i += 1
            Q = np.zeros((len(self.P), len(self.P[0])), dtype=np.float64)
            pi_track.append({s: pi(s, Q) for s in range(len(self.P))})
            for s in range(len(self.P)):
                for a in range(len(self.P[s])):
                    for prob, next_state, reward, done in self.P[s][a]:
                        Q[s][a] += prob * (reward + gamma * V[next_state] * (not done))

            if np.median(np.abs(V - np.max(Q, axis=1))) < theta:
                converged = True
                pi_track.append({s: pi(s, Q) for s in range(len(self.P))})

            V = np.max(Q, axis=1)
            V_track[i] = V

        if not converged:
            warnings.warn("Max iterations reached before convergence. Check theta and n_iters.")

        return Q, V, V_track, pi, pi_track

    def policy_iteration(self, gamma=1.0, n_iters=50, theta=1e-10):
        """
        PARAMETERS:

        gamma {float}:
            Discount factor

        n_iters {int}:
            Number of iterations

        theta {float}:
            Convergence criterion for policy evaluation.
            State values are considered to be converged when the maximum difference between new and previous state
            values is less than theta.


        RETURNS:

        V {numpy array}, shape(possible states):
            State values array

        V_track {numpy array}, shape(n_episodes, nS):
            Log of V(s) for each iteration

        pi {lambda}, input state value, output action value:
            Policy mapping states to actions.
        """
        random_actions = np.random.choice(tuple(self.P[0].keys()), len(self.P))

        def pi(s, _Q=None):
            policy = dict()
            for state, action in enumerate(random_actions):
                policy[state] = action

            return policy[s]

        # initial V to give to `policy_evaluation` for the first time
        V = np.zeros(len(self.P), dtype=np.float64)
        V_track = np.zeros((n_iters, len(self.P)), dtype=np.float64)
        pi_track = []

        i = 0
        converged = False
        while i < n_iters - 1 and not converged:
            i += 1
            old_pi = {s: pi(s) for s in range(len(self.P))}
            pi_track.append(old_pi)
            V = self.policy_evaluation(pi, V, gamma, theta)
            V_track[i] = V
            pi = self.policy_improvement(V, gamma)
            if old_pi == {s: pi(s) for s in range(len(self.P))}:
                converged = True
                pi_track.append(old_pi)

        if not converged:
            warnings.warn("Max iterations reached before convergence. Check n_iters.")

        return None, V, V_track, pi, pi_track

    def policy_evaluation(self, pi, prev_V, gamma=1.0, theta=1e-10):
        while True:
            V = np.zeros(len(self.P), dtype=np.float64)
            for s in range(len(self.P)):
                for prob, next_state, reward, done in self.P[s][pi(s)]:
                    V[s] += prob * (reward + gamma * prev_V[next_state] * (not done))

            if np.median(np.abs(prev_V - V)) < theta:
                break

            prev_V = V.copy()

        return V

    def policy_improvement(self, V, gamma=1.0):
        Q = np.zeros((len(self.P), len(self.P[0])), dtype=np.float64)
        for s in range(len(self.P)):
            for a in range(len(self.P[s])):
                for prob, next_state, reward, done in self.P[s][a]:
                    Q[s][a] += prob * (reward + gamma * V[next_state] * (not done))

        def new_pi(_s, _Q=None):
            policy = dict()
            for state, action in enumerate(np.argmax(Q, axis=1)):
                policy[state] = action

            return policy[_s]

        return new_pi
