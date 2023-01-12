import gym
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from loguru import logger
from models.rl_params import Params
from rl_agent.q_monte_carlo_agent import QMonteCarloAgent
from rl_agent.q_td_agent import QTDAgent


def plot_V(V):
    data = np.array([V[i] for i in range(16)]).reshape(4, 4)
    plt.figure(figsize=(4, 4))
    ax = sns.heatmap(data, linewidth=1, annot=True)
    ax.set_xticklabels(['L', 'D', 'R', 'U'])
    ax.set_yticklabels(['L', 'D', 'R', 'U'])
    ax.set_xlabel('Action')
    ax.set_ylabel('State')

    plt.title('Value function')
    plt.show()


def eval(env: gym.Env, agent, nb_eval_episode: int=50):
    rewards = []
    for _ in range(nb_eval_episode):
        done = False
        state = env.reset()
        while not done:
            action = agent.policy(state, greedy=True)
            next_state, reward, done, info = env.step(action)
            state = next_state
        rewards.append(reward)
    mean_rewards = np.mean(rewards)
    std_rewards = np.std(rewards)
    return mean_rewards, std_rewards



if __name__ == '__main__':

    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False)
    params = Params(epsilon=0.1)
    nb_episodes = 1000
    agent = QMonteCarloAgent(env=env, params=params)
    for _ in range(nb_episodes):
        done = False
        state = env.reset()
        while not done:
            action = agent.policy(state)
            next_state, reward, done, info = env.step(action)
            agent.add_history(state=state, reward=reward, action=action) # store reward, state, action
            state = next_state
        agent.update_q_value()
    agent.plot()
    mean_r, std_r = eval(env, agent)
    logger.info(f"{mean_r=}, {std_r}")