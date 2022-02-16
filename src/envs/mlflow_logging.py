"""
Module contains a class to use MlFlow's built-in logging.
"""
from typing import Optional, List

import mlflow

from config.settings import MLFLOW_TRACKING, SAVE_MODEL
from datasets.datasets import DataSplit


class MlFlowLogger:
    """
    A class for the various MlFlow logging functionalities. It is only suited for the specific
    simulation and environment of this project.
    """

    def __init__(self) -> None:
        """
        Initializes a new instance.
        """
        self.logging_enabled = MLFLOW_TRACKING

    def log_config(self, config: dict) -> None:
        """
        Logs the specified config to MlFlow.

        :param config: The config to be logged.
        :return: None.
        """
        if not self.logging_enabled:
            return
        logging_params = {key: value for key, value in config['params'].items() if key != "qmixer"}
        mlflow.log_params(logging_params)

    def log_episode_reward(self,
                           data_split: DataSplit,
                           num_steps: int,
                           undiscounted_reward: int,
                           discounted_reward: int,) -> None:
        """
        Logs the specified episode reward to MlFlow.

        :param data_split: The used data split to be logged.
        :param num_steps: The number of steps to be logged.
        :param undiscounted_reward: The undiscounted reward to be logged.
        :param discounted_reward: The discounted reward to be logged.
        :return: None.
        """
        if not self.logging_enabled:
            return
        if data_split == DataSplit.TRAINING:
            mlflow.log_metric("episode_reward", undiscounted_reward, step=num_steps)
            mlflow.log_metric("discounted_reward", discounted_reward, step=num_steps)
        elif data_split == DataSplit.VALIDATION:
            mlflow.log_metric("validation_reward", undiscounted_reward, step=num_steps)
        elif data_split == DataSplit.TEST:
            mlflow.log_metric("test_reward", undiscounted_reward, step=num_steps)

    def log_episode_end(self, agents: List, average_reward: float) -> None:
        """
        Logs the python logs and the specified reward to MlFlow. If enabled also saves the
        specified agents.

        :param agents: The agents to be saved.
        :param average_reward: The reward to be logged.
        :return: None.
        """
        if not self.logging_enabled:
            return
        mlflow.log_artifact("info.log")
        mlflow.log_artifact("debug.log")
        mlflow.log_param("average_reward", average_reward)
        if SAVE_MODEL:
            for agent in agents:
                model_path = agent.save_model()
                if model_path is not None:
                    mlflow.log_artifact(model_path)

    def log_episode_number(self, episode: int) -> None:
        """
        Logs the current episode number to MlFlow.

        :param episode: The episode number to be logged.
        :return: None.
        """
        if not self.logging_enabled:
            return
        mlflow.log_metric("current_episode", episode, step=1)

    def log_reward_per_day(self, reward_per_day: List[int]) -> None:
        """
        Logs the reward per day to MlFlow.

        :param reward_per_day: The reward per day to be logged.
        :return: None.
        """
        if not self.logging_enabled:
            return
        # mlflow.log_metric("current_day", env.current_time.dayofyear, step=1)
        mlflow.log_metric("reward_per_day", reward_per_day[-1], step=1)
        mlflow.log_metric("avg_reward_per_day",
                          sum(reward_per_day) / len(reward_per_day), step=1)

    def log_agent_algorithm(self, agent_algorithm: str) -> None:
        """
        Logs the used agent algorithm to MlFlow.

        :param agent_algorithm: The agent algorithm to be logged.
        :return: None.
        """
        if not self.logging_enabled:
            return
        mlflow.log_param("agent", agent_algorithm)

    def log_best_config(self, best_config: Optional[dict]) -> None:
        """
        Logs the best config to MlFlow.

        :param best_config: The best config to be logged.
        :return: None.
        """
        if not self.logging_enabled:
            return
        mlflow.log_param("best_config", best_config)
