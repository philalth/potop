"""
Module contains functions for the simulation environment.
"""
import datetime
import logging
import os
import random
import shutil
from datetime import timedelta
from typing import Tuple, List

import gym
import mlflow
import numpy as np
import pandas as pd
from gym import spaces
from numpy import ndarray

from config.settings import MLFLOW_TRACKING, SAVE_IMAGES, START_HOUR, END_HOUR, RENDER, \
    INITIAL_TIMESTAMP, FINAL_TIMESTAMP, AGENTS_SHARE_OBSERVATION, \
    USE_ALLOWED_PARKING_TIME, CACHED_SHORTEST_PATHS, PARTIAL_RENDERING, \
    OBSERVATION_MODE, AGENTS_VIEW, USE_SPOT_ASSIGNMENT_COLUMN, VIDEO_FOLDER
from datasets.datasets import DataSplit
from envs.enums import CustomAction, ObservationMode, ParkingStatus
from envs.env_constants import UNDISCOUNTED_REWARD, DISCOUNTED_REWARD
from envs.utils import EventType, discount_rewards, is_in_data_split, load_shortest_path_lookup, \
    TIME_COLUMN, TYPE_COLUMN, fine_spots, get_initial_position, get_spots, \
    load_graph_from_file, create_observation, get_num_spots, walking_dist_to_time, \
    get_edges_w_spots, update_spots, get_avg_walking_time, \
    get_route_to_edge, create_partial_observation

if RENDER:
    from envs.rendering import EnvironmentVisualization
else:
    class EnvironmentVisualization:
        """Do nothing if RENDER is set to False."""

        def render(self, graph, spots, agent_positions, current_time) -> None:
            """Do nothing if RENDER is set to False."""


class PotopEnv(gym.Env):
    """
    OpenAI Gym environment which models the TOP (Travelling Officer Problem)
    based on the Melbourne parking sensors dataset.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 event_log: pd.DataFrame,
                 graph_filename: str,
                 data_split: DataSplit,
                 num_agents: int = 1,
                 render: bool = False,
                 observation_mode: ObservationMode = ObservationMode.FULL) \
            -> None:
        logging.info('Creating simulation environment.')
        logging.info('Observation mode: %s', observation_mode)

        self.graph_filename = graph_filename
        self.graph = load_graph_from_file(graph_filename)

        # Maintain a copy of the full event log for resetting the environment
        self.full_event_log = event_log.copy()
        self.event_log = event_log
        self.data_split = data_split

        # initialise the environment at specific timestamp
        self.current_time = INITIAL_TIMESTAMP
        # initialise the environment at specific timestamp
        self.final_time = FINAL_TIMESTAMP

        self.starting_positions = [get_initial_position(self.graph) for _ in range(num_agents)]
        self.agent_positions = self.starting_positions

        self.num_agents = num_agents
        self.agent_log = []
        self.rendering = render

        self.observation_mode = observation_mode

        # True if the working day of all agents is complete
        self.day_finished = False

        num_spots: int = get_num_spots(self.graph)
        # The possible actions are all edges with parking spots
        self.actions: dict = get_edges_w_spots(self.graph)

        # List of all resources (node_one, node_two, spot)
        self.spots: list = get_spots(self.graph)

        # Attributes required by OpenAI Gym
        self.action_space = spaces.Discrete(len(self.actions))
        num_columns = 9 if USE_ALLOWED_PARKING_TIME else 8
        if USE_SPOT_ASSIGNMENT_COLUMN:
            # one column for each agent (one-hot encoding)
            num_columns += self.num_agents
        if OBSERVATION_MODE == ObservationMode.PARTIAL:
            num_columns += 2
        self.observation_space = spaces.Box(0, 1, shape=(num_spots, num_columns))

        # This stores intermediate rewards of agents that are still busy
        self.rewards = np.zeros((num_agents, 2))

        if observation_mode == ObservationMode.PARTIAL:
            # dict of the current view of the agents with partial observability
            # key is the 'id' of the parking spot
            # status is the known status of the parking spot
            # arrivalTime is the time when a car arrived there
            # observationTime is the time when the agent passed the edge and made this observation
            # if AGENTS_SHARE_OBSERVATION is True all observations are saved for agent 0
            if AGENTS_SHARE_OBSERVATION:
                self.agent_view = [{}]
            else:
                self.agent_view = [{} for _ in range(num_agents)]
            for spots in self.spots:
                for agent in range(num_agents):
                    if AGENTS_SHARE_OBSERVATION:
                        self.agent_view[0][spots[2]["id"]] = {"status": ParkingStatus.UNKNOWN,
                                                              "arrivalTime": 0,
                                                              "observationTime": 0}
                    else:
                        self.agent_view[agent][spots[2]["id"]] = {"status": ParkingStatus.UNKNOWN,
                                                                  "arrivalTime": 0,
                                                                  "observationTime": 0}

        # Lookup table for the shortest path between two nodes.
        self.shortest_path_lookup = None
        if CACHED_SHORTEST_PATHS:
            self.shortest_path_lookup = load_shortest_path_lookup(self.graph)

        self.visualization = EnvironmentVisualization()

        # set is_assigned variables of spot to false
        self.is_assigned = np.array([-1 for _ in range(len(self.spots))])
        self.action_to_spot = self.get_action_to_spot_mapping()

        # Check if the starting day is part of the selected data split
        if not is_in_data_split(self.current_time, self.data_split):
            self.next_day()

        num_violations: int = sum(self.event_log[:, TYPE_COLUMN] == EventType.VIOLATION)

        logging.debug('Action space: %s', self.action_space)
        logging.debug('Observation space: %s', self.observation_space)
        logging.debug('Number of parking spots: %i', len(self.spots))
        logging.debug('Number of events: %i', len(self.event_log))
        logging.debug('Number of violations: %i', num_violations)
        logging.debug('Average walking time per edge: %f',
                      get_avg_walking_time(self.graph))

        logging.info('Simulation environment created.')

    # pylint: disable=W0221
    # parameters differ from the overridden method to work for multiple agents
    def step(self, actions) -> Tuple[ndarray, ndarray, ndarray, List[dict]]:
        """
        Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        :param actions: The actions to be executed.
        :return: A tuple containing the following information: the states, the
            obtained rewards, whether the action is done and info, per agent
        """
        logging.info('Executing timestep.')
        logging.info(self.current_time)

        # At least one action should execute an action
        assert not all(action is None for action in actions)

        for i, action in enumerate(actions):
            if action is not None:
                self.execute_action(action, i)
                if not isinstance(action, CustomAction):
                    assert self.is_assigned[np.where(self.is_assigned == i)].size == 0
                    self.is_assigned[self.action_to_spot[action]] = i

        # as multiple agents access the agent_log, the sorting must be guaranteed
        self.agent_log.sort(key=lambda x: x["Time"])

        rewards = self.update_graph()
        self.rewards += rewards
        rewards = self._remove_rewards_for_busy_agents()
        assert not all(r is None for r in rewards)

        for agent_id, busy in enumerate(self._get_busy_agent_mask()):
            if busy == 0 or self.day_finished:
                for j in np.where(self.is_assigned == agent_id):
                    self.is_assigned[j] = -1

        states: List[ndarray] = []
        for index, agent_position in enumerate(self.agent_positions):
            if OBSERVATION_MODE == ObservationMode.FULL:
                state = create_observation(self.graph,
                                           self.spots,
                                           self.shortest_path_lookup,
                                           self.current_time,
                                           agent_position,
                                           self.is_assigned,
                                           self.num_agents)
            else:
                state = create_partial_observation(self.graph,
                                                   self.spots,
                                                   self.shortest_path_lookup,
                                                   self.current_time,
                                                   agent_position,
                                                   self.agent_view,
                                                   index,
                                                   self.is_assigned,
                                                   self.num_agents)
            states.append(state)

        dones: List[bool] = [len(self.event_log) == len(self.agent_log) == 0] * self.num_agents
        infos: List[dict] = [{}] * self.num_agents

        if self.day_finished:
            self.next_day()
            if self.rendering:
                self.render()

        logging.debug('Obtained rewards: %s', rewards)

        return np.array(states), np.array(rewards), np.array(dones), infos

    def execute_action(self, action, agent_id) -> None:
        """
        Executes an action within the environment.

        :param action: The action to be executed.
        :param agent_id: The integer id of the agent.
        :return: None
        """
        logging.info('Executing action: %s', action)

        if action is CustomAction.RANDOM_EDGE:
            edges = list(self.graph.out_edges(self.agent_positions[agent_id]))
            edge = random.choice(edges)
            route = edge
        elif action is CustomAction.SHORTEST_EDGE:
            edges = list(self.graph.out_edges(self.agent_positions[agent_id]))
            index = np.argmin([self.graph[u][v]["havlen"] for u, v in edges])
            edge = edges[index]
            route = edge
        else:
            edge = self.actions[str(action)]
            if CACHED_SHORTEST_PATHS:
                route = self.shortest_path_lookup[
                    self.agent_positions[agent_id]][edge[0]].route.copy()
            else:
                route = get_route_to_edge(self.graph, self.agent_positions[agent_id], edge).copy()
            route.append(edge[1])

        current_time = self.current_time

        for i, node in enumerate(route[:-1]):
            prev_node = node
            next_node = route[i + 1]

            walking_time: timedelta = timedelta(
                seconds=walking_dist_to_time(self.graph[prev_node][next_node]["havlen"]))
            arrival_time = current_time + walking_time
            current_time = arrival_time

            event = {"Time": arrival_time,
                     "Node": next_node,
                     "Agent": agent_id}

            self.agent_log.append(event)

        # last event should be arrival at the targeted edge
        assert self.agent_log[-1]["Node"] == edge[1]

    def move_agent(self, event, agent_id) -> Tuple[float, float]:
        """
        Changes the position of an agent based on an executed event.

        :param event: The executed event.
        :param agent_id: The integer id of the agent.
        :return: The obtained undiscounted reward and the discounted reward
        """
        logging.info('Changing the position of the agent.')

        new_position = event["Node"]
        reward, self.spots = fine_spots(self.spots, self.agent_positions[agent_id], new_position)

        undiscounted_reward = reward
        discounted_reward = discount_rewards(reward, self.current_time, event["Time"])
        assert undiscounted_reward >= discounted_reward

        # update partial view
        if self.observation_mode == ObservationMode.PARTIAL:
            if AGENTS_SHARE_OBSERVATION:
                self.update_agent_view(0, self.agent_positions[agent_id], new_position)
            else:
                self.update_agent_view(agent_id, self.agent_positions[agent_id], new_position)

        self.agent_positions[agent_id] = new_position
        self.current_time = event["Time"]

        return undiscounted_reward, discounted_reward

    def update_agent_view(self, agent_id, old_position, new_position):
        """
        updates the view of a agent when traveling along an edges from "old_position" to
        "new_position"
        """
        if "spots" in self.graph.edges[old_position, new_position]:
            for spot in self.graph.edges[old_position, new_position]["spots"]:
                self.agent_view[agent_id][spot["id"]]["status"] = spot["status"]
                self.agent_view[agent_id][spot["id"]]["arrivalTime"] = spot["arrivalTime"]
                self.agent_view[agent_id][spot["id"]]["observationTime"] = self.current_time

    def update_graph(self) -> np.array:
        """
        Updates the graph based on the current step.

        :return: The obtained rewards.
        """
        logging.debug('Updating graph.')

        rewards = np.zeros((self.num_agents, 2))
        events = []

        while self._all_agents_are_busy() and not self.day_finished:
            while len(self.event_log) > 0 and \
                    self.event_log[0][TIME_COLUMN] < self.agent_log[0]["Time"]:
                events.append(self.event_log[0])
                self.event_log = self.event_log[1:]

            if len(events) > 0:
                logging.debug('Updating %i spots.', len(events))
                # Set time to latest event
                self.current_time = events[-1][TIME_COLUMN]
                # Update spots based on events
                self.spots = update_spots(self.spots, events)
                # Events will be deleted while updating

            event = self.agent_log.pop(0)

            if event["Time"].hour >= END_HOUR:
                self.day_finished = True
                # All events after can be discarded
                self.agent_log = []
            else:
                agent_id = event["Agent"]

                undiscounted_reward, discounted_reward = self.move_agent(event, agent_id)
                rewards[agent_id][UNDISCOUNTED_REWARD] += undiscounted_reward
                rewards[agent_id][DISCOUNTED_REWARD] += discounted_reward

            if self.rendering:
                self.render()

        return rewards

    def next_day(self) -> None:
        """
        After the working time for the day is exceeded, set the time to 07:00
        on the next day and restart the agent positions.
        """
        # save images of the simulation as artifacts
        if SAVE_IMAGES and MLFLOW_TRACKING:
            self.save_images(VIDEO_FOLDER + "/" + str(self.current_time.date()))

        # get date of the next day
        next_date: datetime = self.current_time + pd.DateOffset(1)

        # Set new time
        self.current_time = self.current_time.replace(year=next_date.year,
                                                      month=next_date.month,
                                                      day=next_date.day,
                                                      hour=START_HOUR,
                                                      minute=0, second=0, microsecond=0)

        if not is_in_data_split(next_date, self.data_split):
            # Skip another day
            self.next_day()
            return

        # Reset agent positions
        self.agent_positions = self.starting_positions
        self.day_finished = False

        # All rewards should be 0.0 at the start of the new day
        assert np.all(self.rewards == [0.0, 0.0])
        if OBSERVATION_MODE == ObservationMode.PARTIAL:
            # clear agents view of environment
            for spots in self.spots:
                for agent in range(self.num_agents):
                    if AGENTS_SHARE_OBSERVATION:
                        self.agent_view[0][spots[2]["id"]] = {"status": ParkingStatus.UNKNOWN,
                                                              "arrivalTime": 0,
                                                              "observationTime": 0}
                    else:
                        self.agent_view[agent][spots[2]["id"]] = {"status": ParkingStatus.UNKNOWN,
                                                                  "arrivalTime": 0,
                                                                  "observationTime": 0}

    @staticmethod
    def save_images(path_to_save):
        """zip image-folder and log zip file via ml flow"""
        if os.path.exists(path_to_save):
            shutil.make_archive(path_to_save, 'zip', path_to_save)
            mlflow.log_artifact(path_to_save + ".zip", artifact_path="images")

    def reset(self) -> List[ndarray]:
        """Resets the state of the environment."""
        logging.info('Resetting simulation environment.')

        self.starting_positions = [get_initial_position(self.graph) for _ in
                                   range(self.num_agents)]

        self.agent_positions = self.starting_positions

        # Reset graph
        self.graph = load_graph_from_file(self.graph_filename)
        self.spots: list = get_spots(self.graph)

        # Reset event log
        self.event_log = self.full_event_log.copy()

        # Agent log should be empty
        assert len(self.agent_log) == 0

        # Start at the environment at specific timestamp
        self.current_time = INITIAL_TIMESTAMP

        self.day_finished = False
        self.rewards = np.zeros((self.num_agents, 2))

        # Check if the starting day is part of the selected data split
        if not is_in_data_split(self.current_time, self.data_split):
            self.next_day()
        state = []
        if OBSERVATION_MODE == ObservationMode.PARTIAL:
            for agent in range(self.num_agents):
                for spots in self.spots:
                    if AGENTS_SHARE_OBSERVATION:
                        self.agent_view[0][spots[2]["id"]] = {"status": ParkingStatus.UNKNOWN,
                                                              "arrivalTime": 0,
                                                              "observationTime": 0}
                    else:
                        self.agent_view[agent][spots[2]["id"]] = {"status": ParkingStatus.UNKNOWN,
                                                                  "arrivalTime": 0,
                                                                  "observationTime": 0}
                state.append(create_partial_observation(self.graph,
                                                        self.spots,
                                                        self.shortest_path_lookup,
                                                        self.current_time,
                                                        self.agent_positions[agent],
                                                        self.agent_view,
                                                        0,
                                                        self.is_assigned,
                                                        self.num_agents))
                # take zero(the first agent), as all agents
                # have no observations and thus all the same view

        else:
            for position in self.agent_positions:
                state.append(create_observation(self.graph,
                                                self.spots,
                                                self.shortest_path_lookup,
                                                self.current_time,
                                                position,
                                                self.is_assigned,
                                                self.num_agents))
        return state

    def set_data_split(self, data_split) -> None:
        """Changes the data split."""
        logging.debug("Set data split to: %s", data_split)
        self.data_split = data_split

    def render(self, mode='human'):
        """Wrapper function that either calls existing render function or empty dummy."""
        if PARTIAL_RENDERING and OBSERVATION_MODE == ObservationMode.PARTIAL:
            # render partial view
            if AGENTS_VIEW < self.num_agents and not AGENTS_SHARE_OBSERVATION:
                agent_id = AGENTS_VIEW
            else:
                agent_id = 0
            spots_status = [spot.get('status') for _, spot in self.agent_view[agent_id].items()]
        else:
            # render full view
            spots_status = [spot.get('status') for _, _, spot in self.spots]
        self.visualization.render(self.graph, spots_status, self.agent_positions,
                                  self.current_time)

    def close(self) -> None:
        """Closes the running environment instance."""
        self.graph = None
        logging.info('Simulation environment closed.')

    def _all_agents_are_busy(self):
        """
        Returns true if all agents are still executing actions.
        """
        busy_agents = self._get_busy_agents()
        return len(busy_agents) == self.num_agents and len(self.agent_log) > 0

    def _get_busy_agent_mask(self):
        busy_agents = self._get_busy_agents()
        return [1 if agent_id in busy_agents else 0 for agent_id in range(self.num_agents)]

    def _get_busy_agents(self) -> set:
        """
        Busy agents are agents that still have events left in the agent log.
        They should not execute new actions, since the old action still is not completed.
        """
        busy_agents: set = set()
        for event in self.agent_log:
            busy_agents.add(event['Agent'])
        return busy_agents

    def _get_agent_event(self, agent_id):
        for agent_event in self.agent_log:
            if agent_event['Agent'] == agent_id:
                return agent_event
        return None

    def _remove_rewards_for_busy_agents(self) -> np.array:
        """Stores rewards of busy agents and returns rewards of finished agents."""
        # 1 means busy, 0 means not busy
        busy_agents = self._get_busy_agent_mask()
        # return rewards of agents that are not busy
        return_rewards = [r if i == 0 else None for i, r in zip(busy_agents, self.rewards)]
        # store rewards of agents that are still busy
        self.rewards = np.array([r if i == 1 else [0.0, 0.0]
                                 for i, r in zip(busy_agents, self.rewards)])
        return np.array(return_rewards)

    def get_spot_to_edge_mapping(self) -> dict:
        """Returns the mapping from parking spots to edge ids."""
        mapping = {}
        i = 0
        for node_one, node_two, data in self.graph.edges(data=True):
            if "spots" in data:
                for _ in data["spots"]:
                    mapping[str(i)] = int(list(self.actions.keys())[
                        list(self.actions.values()).index(
                            [node_one, node_two])])
                    i += 1
        return mapping

    def get_action_to_spot_mapping(self) -> dict:
        """Returns the mapping from action ids to parking spots."""
        mapping = [[] for _ in range(len(self.actions))]
        i = 0
        for node_one, node_two, data in self.graph.edges(data=True):
            if "spots" in data:
                for _ in data["spots"]:
                    edge_id = int(list(self.actions.keys())[list(
                        self.actions.values()).index([node_one, node_two])])
                    mapping[edge_id].append(i)
                    i += 1
        return mapping
