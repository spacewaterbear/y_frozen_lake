import gym
from loguru import logger
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from rl_env.bandit_gym_env import BanditGym

if __name__ == '__main__':
    bandit_probas = [0.1, 0.2, 0.8, 0.9]
    env = BanditGym(bandit_probas=bandit_probas)
    model = PPO("MlpPolicy", env, verbose=0)
    ts = 100
    model.learn(total_timesteps=ts)
    mean_reward, std_reward = evaluate_policy(model=model, env=model.get_env(), n_eval_episodes=1000)
    logger.info(f"{mean_reward=}, {std_reward} for timestep {ts}")
