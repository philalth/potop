"""
Module contains the base class for creating agents.
"""
from abc import ABC


class Agent(ABC):
    """The base class for creating agents. Cannot be instantiated."""

    def __init__(self, action_space, observation_space) -> None:
        """
        Initializes a new instance.

        :param action_space: The agents action space.
        :param observation_space: The agents observation space.
        """
        self.action_space = action_space
        self.observation_space = observation_space
        self.test = False

    def act(self, state) -> int:
        """
        Returns an action based on the state of the environment.

        :param state: The current state of the environment.
        :return: The action to be executed.
        """

    def save_transition(self, next_state, action, reward, old_state, done) -> None:
        """
        Saves a transition to the agent's memory.

        :param next_state: List of next states.
        :param action: The executed action.
        :param reward: The obtained reward.
        :param old_state: List of old states.
        :param done: Whether the agent finished the action.
        :return: None.
        """

    def save_model(self) -> None:
        """
        Saves the current model of the agent.

        :return: None.
        """

    def load_model(self, filename) -> None:
        """
        Loads the pretrained model.

        :param filename: Filepath to the saved model.
        :return: None.
        """

    def set_test(self, test: bool) -> None:
        """
        Sets the test value of the agent. If true, the agent stops exploring and only exploits.

        :param test: Boolean value of the test parameter.
        :return: None.
        """
        self.test = test
