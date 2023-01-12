import gym
import time

from rl_agent.q_td_agent import QTDAgent
from models.rl_params import Params
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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


if __name__ == '__main__':

    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False)
    params = Params(epsilon=0.8)
    nb_episodes = 1000
    agent = QTDAgent(env=env, params=params)
    for _ in range(nb_episodes):
        done = False
        state = env.reset()
        while not done:
            action = agent.policy(state)
            next_state, reward, done, info = env.step(action)
            agent.update_value_function(state, reward, next_state, action=action)
            state = next_state
    agent.plot()
    # plot_V(agent.Q)

