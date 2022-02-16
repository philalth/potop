"""
Module contains functionality for a random agent.
"""

from enum import Enum
from typing import Union

from gym.spaces import Discrete, Box
from numpy import ndarray

from agents.agent import Agent
from envs.enums import CustomAction


class Strategy(Enum):
    """
    Describes the strategy of the random agent.
        RANDOM_EDGE -> choose random edge from current position
        RANDOM_ROUTE -> choose a random route to an edge with a parking spot
    """
    RANDOM_EDGE = 0
    RANDOM_ROUTE = 1


class RandomAgent(Agent):
    """A baseline agent taking random actions."""

    def __init__(self, action_space: Discrete, observation_space: Box, params: dict) -> None:
        """
        Initializes a new instance.

        :param action_space: The action space of this agent.
        :param observation_space: The observation space of this agent (unused).
        :param params: The parameters for this agent (only strategy is necessary).
        """
        super().__init__(action_space, observation_space)
        self.strategy: Strategy = params["strategy"]

    def act(self, _state: ndarray) -> Union[int, CustomAction]:
        """
        Sample a random action from the action space.

        :param _state: The current state of the environment (unused).
        :return: The action to be executed.
        """
        if self.strategy is Strategy.RANDOM_ROUTE:
            action: int = self.action_space.sample()
        elif self.strategy is Strategy.RANDOM_EDGE:
            # Return random_edge, the env will handle the rest.
            action: CustomAction = CustomAction.RANDOM_EDGE
        else:
            raise NotImplementedError
        return action
