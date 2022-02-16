"""
Module contains functionality for a greedy agent.
"""

from typing import List, Union, Optional, Any, Tuple

import numpy as np
from numpy import ndarray

from agents.agent import Agent
from config.settings import END_HOUR, START_HOUR
from envs.enums import CustomAction
from envs.utils import IN_VIOLATION_ENCODING

SPOT_INDEX: int = 0
AGENT_INDEX: int = 1
HEURISTIC_INDEX: int = 2


class Greedy(Agent):
    """A baseline agent implementing the greedy algorithm."""

    def __init__(self, action_space: Any, observation_space: Any, params: dict):
        """
        Initializes a new instance.

        :param action_space: The agents action space.
        :param observation_space: The agents observation space.
        :param params: The necessary parameters.
        """
        super().__init__(action_space, observation_space)
        self.mapping: Any = params["spot_to_edge_mapping"]
        self.num_agents: int = params["num_agents"]
        # a list of all the spots that were taken by agents
        self.assigned_spots: list = [None] * self.num_agents

    # pylint: disable=W0221
    # arguments differ from overridden method
    def act(self, states: List[ndarray], available_agents: List[bool]) \
            -> List[Union[Optional[int], str]]:
        """
        Returns the action that the agents will perform, based on observed states.

        :param states: A list of length self.num_agents, containing all states as ndarrays
        :param available_agents: A list of booleans of length self.num_agents, indicating whether
        agent i is available.
        :return: a list of actions of length self.num_agents. The value -1 is used to signal the
        environment to choose an adjacent edge.
        """
        for agent_id, available in enumerate(available_agents):
            # if agent is available, no spot is assigned
            if available:
                self.assigned_spots[agent_id] = None

        actions: List[Optional[CustomAction]] = [
            CustomAction.SHORTEST_EDGE if available_agents[i] else None for i in
            range(self.num_agents)]

        violations: List[tuple] = get_violations(states, available_agents)
        violations = self._remove_assigned_spots(violations)
        # sort based on heuristic
        violations.sort(key=lambda x: x[HEURISTIC_INDEX], reverse=True)

        while len(violations) > 0 and any(available_agents):
            # get spot with best heuristic
            spot_id, agent_id, _ = violations.pop(0)
            # check if agent is still available
            assert available_agents[agent_id]
            # assign action
            actions[agent_id] = self.mapping[str(spot_id)]
            self.assigned_spots[agent_id] = spot_id
            # agent is no longer available
            available_agents[agent_id] = False

            # remove spots with this id and spots for this agent
            violations = [v for v in violations if
                          (v[SPOT_INDEX] != spot_id) and (v[AGENT_INDEX] != agent_id)]

        return actions

    def _remove_assigned_spots(self, violations: List[tuple]) -> List[tuple]:
        """
        Removes the violations with spots that are already assigned to some agents.

        :param violations: A list of tuples (spot_id, agent_id, heuristic)
        :return: A filtered list of violations.
        """
        violations = [v for v in violations if v[SPOT_INDEX] not in self.assigned_spots]
        return violations


def get_violations(states: List[np.ndarray], available_agents: List[bool]) -> List[tuple]:
    """
    Returns a list of all spots currently in violation. Consist of tuples with values spot_id,
    agent_id and the heuristic. The heuristic is the same as in the single agent case
    (found in greedy.py).

    :param states: A list of all observed states.
    :param available_agents: A list of booleans, indicating if an agent is currently available.
    :return: A list of tuples (spot_id, agent_id, heuristic).
    """
    violations: List[Tuple[int, int, float]] = []

    # consider the state of each agent because of a partial observability
    for agent_id, state in enumerate(states):
        # only look at states of available agents
        if available_agents[agent_id]:
            # each row in the state space represents a parking spot
            for spot_id, row in enumerate(state):
                # Find violations
                if (row[:4] == IN_VIOLATION_ENCODING).all():
                    heuristic: float = get_heuristic(row)
                    # append the tuple (spot_id, agent_id, heuristic)
                    violations.append((spot_id, agent_id, heuristic))
    return violations


def get_heuristic(spot: ndarray) -> float:
    """
    Calculate the greedy heuristic of the parking spot.

    :param spot: The spot to be checked.
    :return: The greedy heuristic (total violation time + distance).
    """
    distance: int = spot[4]
    current_time_in_seconds: int = spot[5]
    arrival_time_in_seconds: int = spot[6]

    try:
        allowed_time: int = spot[8] * ((END_HOUR - START_HOUR) * 3600)
    except IndexError as error:
        raise IndexError(
            "The observation array has to few columns (tried to access at position 8)."
            "Turn on the 'USE_NINTH_COLUMN_ALLOWED_PARKING_TIME' flag in settings.py to fix this "
            "error.") from error

    total_violation_time: int = current_time_in_seconds - (arrival_time_in_seconds + allowed_time)
    distance = distance * 3600  # in seconds

    # walking speed is already taken into account in distance
    return -(total_violation_time + distance)


def get_shortest_distance(state: Any) -> Any:
    """
    Returns the parking spot with the shortest walking distance.

    :param state: The current state.
    :return: The closest parking spot.
    """
    dist: np.int = np.inf
    spot: Any = None
    for row in state:
        if row[4] <= dist:
            dist = row[4]
            spot = row
    return spot
