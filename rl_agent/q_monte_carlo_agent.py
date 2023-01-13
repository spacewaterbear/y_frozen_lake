import random

import gym
import numpy as np

from models.rl_params import Params


np.random.seed(0)

class QMonteCarloAgent:
    def __init__(self, env: gym.Env, params: Params):
        self.env = env
        self.Q = np.zeros((env.observation_space.n, env.action_space.n))
        self.params = params
        self.history = []

    def policy(self, state, greedy=False):
        """greedy action always"""
        if np.random.random() < self.params.epsilon and not greedy:
            return self.env.action_space.sample()
        else:
            return np.random.choice(np.flatnonzero(self.Q[state] == self.Q[state].max()))

    def plot(self):
        V = np.max(self.Q, axis=1)
        print(V)

    def add_history(self, state, reward, action):
        self.history.append((state, reward, action))

    def update_q_value(self):
        G = 0
        for state, reward, action in reversed(self.history):
            G = self.params.gamma*G+reward
            self.Q[state, action] = self.Q[state, action] + self.params.alpha*(G-self.Q[state, action])
        self.history = []


