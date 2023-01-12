import random

import gym
import numpy as np

from models.rl_params import Params


class QTDAgent:
    def __init__(self, env: gym.Env, params: Params):
        self.env = env
        self.Q = np.zeros((env.observation_space.n, env.action_space.n))
        self.params = params

    def policy(self, state):
        """greedy action always"""
        if random.random() < self.params.epsilon:
            return self.env.action_space.sample()
        else:
            # return np.random.choice(np.flatnonzero(self.Q[state] == self.Q[state].max()))
            return np.argmax(self.Q[state])

    def update_value_function(self, state: int, reward: float, next_state: int, action: int):
        """update with TD value"""
        self.Q[state, action] = self.Q[state, action] + self.params.alpha * (reward + self.params.gamma * np.max(self.Q[next_state]) - self.Q[state, action])

    def plot(self):
        V = np.max(self.Q, axis=1)
        print(V)
