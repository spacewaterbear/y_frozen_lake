import random

import gym
import numpy as np
from gym import spaces
from stable_baselines3.common.env_checker import check_env


class BanditGym(gym.Env):
    """Custom Environment that follows gym interface."""

    # metadata = {"render.modes": ["human"]}

    def __init__(self, bandit_probas: list[float]):
        super().__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.bandit_probas = bandit_probas
        self.action_space = spaces.Discrete(len(self.bandit_probas))
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Discrete(1)

    def step(self, action):
        done = True
        info = {}
        # 1 en fonction de la proba du bandit
        rv = random.random()
        bandit_proba = self.bandit_probas[action]


        if rv<bandit_proba:
            reward = 1
        else:
            reward = 0
        # same as below
        # reward = 1 if rv<bandit_proba else 0
        observation = 0
        return observation, reward, done, info

    def reset(self):
        return 0  # reward, done, info can't be included



if __name__ == '__main__':
    bandit_probas = [0.1,0.2,0.8]
    env = BanditGym(bandit_probas=bandit_probas)
    env.step(action=2)
    check_env(env)
