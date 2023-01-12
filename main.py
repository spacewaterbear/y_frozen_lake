import gym
import time
from rl_agent.td_agent import TDAgent
from models.rl_params import Params


if __name__ == '__main__':

    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False)
    params = Params()
    nb_episodes = 1000
    agent = TDAgent(env=env, params=params)
    for _ in range(nb_episodes):
        done = False
        state = env.reset()
        while not done:
            action = agent.policy()
            next_state, reward, done, info = env.step(action)
            agent.update_value_function(state, reward, next_state)
            state = next_state

    agent.plot()

