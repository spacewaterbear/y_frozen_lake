import gym
from loguru import logger

from models.rl_params import Params
from rl_agent.q_monte_carlo_agent import QMonteCarloAgent
from utils.utils import HelperFunction

if __name__ == '__main__':
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False)
    params = Params(epsilon=0.1)
    nb_episodes = 1000
    agent = QMonteCarloAgent(env=env, params=params)
    trained_agent = HelperFunction.train(env=env, agent=agent, nb_episodes=nb_episodes)
    mean_r, std_r = HelperFunction.eval_trained_agent(env, trained_agent)
    logger.info(f"{mean_r=}, {std_r}")
