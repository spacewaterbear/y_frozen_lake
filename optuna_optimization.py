import gym
import optuna
import wandb
from loguru import logger
from optuna.integration.wandb import WeightsAndBiasesCallback
from models.rl_params import Params
from rl_agent.q_monte_carlo_agent import QMonteCarloAgent
from utils.utils import HelperFunction
from variables import WANDB_API_KEY

wandb_kwargs = {"project": "monte-carlo-hp-search"}
wandbc = WeightsAndBiasesCallback(
    metric_name=["mean_r", "nb_train_episode"],
    wandb_kwargs=wandb_kwargs, as_multirun=True
)

@wandbc.track_in_wandb()
def objective(trial):
    """
    hp to search :

    # train
    # eval -> mean_r
    :param trial:
    :return: mean_r
    """
    eps = trial.suggest_float("eps", 0.01, 0.1)
    alpa = trial.suggest_float("alpa", 0.1, 0.3)
    gamma = trial.suggest_float("gamma", 0.9, 0.99)
    nb_train_episode = trial.suggest_int("nb_train_episode", 10, 100)
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False)
    params = Params(epsilon=eps, alpha=alpa, gamma=gamma)
    agent = QMonteCarloAgent(env=env, params=params)
    trained_agent = HelperFunction.train(env=env, agent=agent, nb_episodes=nb_train_episode, plot=False)
    mean_r, _ = HelperFunction.eval_trained_agent(env, trained_agent)
    return mean_r, nb_train_episode


if __name__ == '__main__':

    wandb.login(key=WANDB_API_KEY)
    study = optuna.create_study(directions=["maximize", "minimize"])
    study.optimize(objective, n_trials=50, callbacks=[wandbc])
    logger.info(study.best_trials)
