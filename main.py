import gym
from loguru import logger
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

if __name__ == '__main__':
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False)
    tims_range = [10, 100, 10000]
    for ts in tims_range:
        model = PPO("MlpPolicy", env, verbose=0)
        model.learn(total_timesteps=ts)

        mean_reward, std_reward = evaluate_policy(model=model, env=model.get_env())
        logger.info(f"{mean_reward=}, {std_reward} for timestep {ts}")
