import random
import gym
from models.rl_params import Params
import numpy as np

class TDAgent:
    def __init__(self, env: gym.Env, params: Params):
        self.env = env
        self.V = np.zeros(self.env.observation_space.n)
        self.params = params

    def policy(self):
        """Random action always"""
        return self.env.action_space.sample()

    def update_value_function(self, state, reward, next_state):
        """update xwith TD value"""
        self.V[state] = self.V[state] + self.params.alpha * (reward + self.params.gamma*self.V[next_state] - self.V[state])


    def plot(self):
        V = self.V.copy().reshape(4,4)
        print(V)
