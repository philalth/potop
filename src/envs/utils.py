"""
Module contains several utility methods for the simulation environment.
"""

import datetime as dt
import logging
import pickle
import platform
import subprocess
from collections import namedtuple
from random import choice
from typing import List, Tuple

import networkx as nx
import numpy as np
from numpy import ndarray
from shapely.geometry import LineString

from config.settings import CACHED_SHORTEST_PATHS, GAMMA, USE_CAR_ARRIVAL_TIME, \
    USE_SPOT_ASSIGNMENT_COLUMN, WALKING_SPEED, SHORTEST_PATHS_LOOKUP_PATH, \
    USE_ALLOWED_PARKING_TIME, START_HOUR, END_HOUR, \
    USE_NINTH_COLUMN_ALLOWED_PARKING_TIME
from datasets.datasets import DataSplit
from envs.enums import ParkingStatus, EventType
from envs.env_constants import FREE_ENCODING, IN_VIOLATION_ENCODING, FINED_ENCODING, \
    OCCUPIED_ENCODING, FREE_ENCODING_PARTIAL, OCCUPIED_ENCODING_PARTIAL, \
    IN_VIOLATION_ENCODING_PARTIAL, FINED_ENCODING_PARTIAL, UNKNOWN_ENCODING_PARTIAL, TYPE_COLUMN, \
    TIME_COLUMN, MAX_MINUTES_COLUMN, STREET_MARKER_COLUMN
from envs.exceptions import ServerNotRespondingException


# pylint: disable=(R0913, R0914)


def load_graph_from_file(filename: str) -> nx.DiGraph:
    """Loads graph with assigned parking spots from file."""
    graph: nx.DiGraph = nx.read_gpickle(filename)

    logging.info('Graph loaded from file: %s', filename)
    logging.info('Graph info: %s', nx.info(graph))

    return graph


def is_in_data_split(date: dt.datetime, data_split: DataSplit):
    """Returns true, if the selected date is included in the selected dataplit."""
    logging.debug(date.dayofyear)
    if data_split is DataSplit.TRAINING:
        return date.dayofyear % 13 > 1
    if data_split is DataSplit.VALIDATION:
        return date.dayofyear % 13 == 1
    if data_split is DataSplit.TEST:
        return date.dayofyear % 13 == 0
    raise AssertionError


def ensure_strongly_connected(graph):
    """Ensures that the graph is a single strongly connected component."""
    if nx.number_strongly_connected_components(graph) > 1:
        queue = sorted(nx.strongly_connected_components(graph),
                       key=len, reverse=True)[1:]
        while queue:
            max_scc = max(nx.strongly_connected_components(graph), key=len)
            component = queue.pop(0)
            connected = False
            for node in iter(component):
                # check neighbors that point to the node in this component
                nghbs = {n for n, _ in graph.in_edges(node)}
                intersection = nghbs.intersection(max_scc)
                if len(intersection) > 0:
                    add_reverted_edge(graph, list(intersection),
                                      node, node_to_ngh=True)
                    connected = True
                    break
                # check neighbors that are reachable from the node in this
                # component
                nghbs = {n for _, n in graph.out_edges(node)}
                intersection = nghbs.intersection(max_scc)
                if len(intersection) > 0:
                    add_reverted_edge(graph, list(
                        intersection), node, node_to_ngh=False)
                    connected = True
                    break
            if connected:
                continue
            queue.append(component)
    logging.info("Graph is now strongly connected.")
    return graph


def add_reverted_edge(graph, neighbors, node, node_to_ngh):
    """Adds a reverted edge."""
    neighbor = neighbors[np.random.randint(len(neighbors))]
    src = node if node_to_ngh else neighbor
    target = neighbor if node_to_ngh else node
    edge_data = {"id": graph[target][src]["id"] + "-r",
                 "havlen": graph[target][src]["havlen"]}
    # revert geometry if exists
    if 'geometry' in edge_data:
        geometry = np.flip(
            np.array(graph[target][src][0]["geometry"].xy), axis=1)
        edge_data["geometry"] = LineString(
            list(zip(geometry[0, :], geometry[1, :])))
    graph.add_edge(src, target, **edge_data)


# pylint: disable=R0913,R0914
def create_observation(graph: nx.DiGraph,
                       spots: list,
                       shortest_path_lookup: dict,
                       current_time: dt.datetime,
                       agent_position,
                       is_assigned: List[bool],
                       num_agents: int) -> ndarray:
    """Creates the observation based on the graph.

    Structure of an observation:

        index | Description
        ------|-----------------------------------------------------------
        0     | One-hot encoding for parking status (Free)
        1     | One-hot encoding for parking status (Occupied)
        2     | One-hot encoding for parking status (Violation)
        3     | One-hot encoding for parking status (Fined)
        4     | Walking time (distance of agent to parking spot)
        5     | The current date and time (normalized)
        6     | The time of arrival of the agent (normalized)
        7     | An indicator for free or violation, ranging from -1 to +2
        8     | The max allowed parking time for this spot [optional, depends on settings.py]
        9     | An indicator if the spot is already assigned to an agent [optional]
    """
    state: List[List[int]] = []

    for i, (node_one, node_two, spot) in enumerate(spots):
        edge_length = graph.get_edge_data(node_one, node_two)["havlen"]

        # distance = distance to start_node of edge + distance of edge itself
        if CACHED_SHORTEST_PATHS:
            walking_dist = shortest_path_lookup[agent_position][node_one].length + edge_length
        else:
            walking_dist = get_shortest_distance_to_edge(graph,
                                                         agent_position,
                                                         (node_one, node_two))
        walking_time = walking_dist_to_time(walking_dist) / 3600

        datetime = normalize_datetime(current_time)

        agent_arrival_time = datetime + walking_time / ((END_HOUR - START_HOUR) * 3600)

        # use maxMinutes, if the spot is occupied (arrival time not zero)
        allowed_time = spot["maxSeconds"] if spot["arrivalTime"] > 0 else 0

        # encoded_status = [x, x, x, x]
        observation = [
            *encode_status(spot["status"]),
            walking_time,
            datetime_to_sec(current_time) if USE_CAR_ARRIVAL_TIME else datetime,
            spot["arrivalTime"] if USE_CAR_ARRIVAL_TIME else agent_arrival_time,
            time_left(datetime_to_sec(current_time), spot["arrivalTime"],
                      allowed_time)
        ]

        if USE_ALLOWED_PARKING_TIME:
            observation.append(allowed_time)
        if USE_NINTH_COLUMN_ALLOWED_PARKING_TIME:
            observation.append(allowed_time / ((END_HOUR - START_HOUR) * 3600))

        if USE_SPOT_ASSIGNMENT_COLUMN:
            for j in encode_spot_assignment(is_assigned[i], num_agents):
                observation.append(j)

        state.append(observation)

    return np.array(state)


def create_partial_observation(graph: nx.DiGraph,
                               spots: list,
                               shortest_path_lookup: dict,
                               current_time: dt.datetime,
                               agent_position,
                               agent_view: [dict],
                               agent_number: int,
                               is_assigned: List[bool],
                               num_agents: int) -> ndarray:
    """Creates the observation based on the graph.

    Structure of an partial observation:

        index | Description
        ------|-----------------------------------------------------------
        0     | One-hot encoding for parking status (Free)
        1     | One-hot encoding for parking status (Occupied)
        2     | One-hot encoding for parking status (Violation)
        3     | One-hot encoding for parking status (Fined)
        4     | One-hot encoding for parking status (Unknown)
        5     | Walking time (distance of agent to parking spot)
        6     | The current date and time (normalized)
        7     | The time of arrival of the agent (normalized)
        8     | An indicator for free or violation, ranging from -1 to +2
        9     | Time of observation (normalized)
        10     | The max allowed parking time for this spot [optional, depends on settings.py]
    """
    state: List[List[int]] = []

    for i, node_one, node_two, spot in enumerate(spots):
        edge_length = graph.get_edge_data(node_one, node_two)["havlen"]

        # distance = distance to start_node of edge + distance of edge itself
        if CACHED_SHORTEST_PATHS:
            walking_dist = shortest_path_lookup[agent_position][node_one].length + edge_length
        else:
            walking_dist = get_shortest_distance_to_edge(graph,
                                                         agent_position,
                                                         (node_one, node_two))
        walking_time = walking_dist_to_time(walking_dist) / 3600

        datetime = normalize_datetime(current_time)
        # car_arrival_time = agent_view[0][spot["id"]]["arrivalTime"]
        agent_arrival_time = datetime + walking_time / ((END_HOUR - START_HOUR) * 3600)

        # use maxMinutes, if the spot is occupied (arrival time not zero)
        allowed_time = spot["maxSeconds"] if spot["arrivalTime"] > 0 else 0
        time_since_observation = 1  # 1 means an observation is 1 day old (the maximum
        # observation period)
        max_seconds = (END_HOUR - START_HOUR) * 3600

        observation_time = agent_view[agent_number][spot["id"]]["observationTime"]
        if observation_time != 0:
            observation_time = datetime_to_sec(observation_time)
            time_since_observation = (datetime_to_sec(
                current_time) - observation_time) / max_seconds

        # encoded_partial_status = [x, x, x, x, x]
        observation = [
            *encode_partial_status(agent_view[agent_number][spot["id"]]["status"]),
            walking_time,
            datetime,
            agent_arrival_time,
            time_left(datetime_to_sec(current_time), agent_view[0][spot["id"]]["arrivalTime"],
                      allowed_time),
            observation_time,
            time_since_observation
        ]

        if USE_NINTH_COLUMN_ALLOWED_PARKING_TIME:
            observation.append(allowed_time / ((END_HOUR - START_HOUR) * 3600))

        if USE_SPOT_ASSIGNMENT_COLUMN:
            for j in encode_spot_assignment(is_assigned[i], num_agents):
                observation.append(j)

        state.append(observation)

    return np.array(state)


def walking_dist_to_time(walking_dist: float, time_unit="seconds") -> float:
    """Calculates the walking time (in seconds) based on the distance (in meters)."""
    # walking speed is set to km/h
    # calculate walking_dist * min/meter
    # time (in minutes) = m * (min / m)
    walking_time: float = walking_dist * (60 / (WALKING_SPEED * 1000))
    # convert to seconds if needed
    if time_unit == "seconds":
        walking_time = walking_time * 60
    return walking_time


def time_left(datetime, car_arrival_time, allowed_time, cutoff=3_600 * 4) -> float:
    """ Calculate the coding between -1 and +2 that indicates if
        and how long a car is already in violation.
       -1: Car just arrived
        0: Violation is about to start (car has reached maximum allowed parking time)
        2: Long violation

        Default cutoff value: 4 hours
    """

    parking_duration = datetime - car_arrival_time
    assert parking_duration >= 0.0

    if allowed_time == 0:
        return -1.0

    if parking_duration <= allowed_time:
        #  0: car stayed for full allowed time, parking time == allowed time
        # -1: car just arrived, parking_duration == 0
        coding = parking_duration / allowed_time - 1.0
    else:
        # 0: car is about to enter violation time
        # 2: car stayed for allowed time + cutoff OR LONGER
        if parking_duration - allowed_time > cutoff:
            coding = 2.0
        else:
            coding = 2.0 * ((parking_duration - allowed_time) / cutoff)
    return coding


def discount_rewards(rewards: float, start_time: dt.datetime, end_time: dt.datetime):
    """Discount the rewards based on the duration of the edge."""
    time_delta = (end_time - start_time).total_seconds()
    assert time_delta >= 0
    discounted_rewards = rewards * (GAMMA ** time_delta)
    return discounted_rewards


def normalize_datetime(datetime):
    """Converts datetime to seconds since the daily working start of the agent"""
    time = datetime.time()
    time_in_seconds = (time.hour * 60 + time.minute) * 60 + time.second
    time_normalized = (time_in_seconds - START_HOUR * 3600) / ((END_HOUR - START_HOUR) * 3600)
    return time_normalized


def datetime_to_sec(datetime):
    """Converts datetime to seconds since 01.01.2017"""
    time = (datetime - dt.datetime(2017, 1, 1)).total_seconds()
    return time


def get_initial_position(graph: nx.Graph):
    """Returns the randomly chosen initial position of an agent."""
    random_node = choice(list(graph.nodes))
    logging.info("Starting position: %s", str(random_node))
    return random_node


def encode_status(status: ParkingStatus) -> List[int]:
    """Returns the one-hot encoding of a parking spot status."""
    if status == ParkingStatus.FREE:
        encoding = FREE_ENCODING
    elif status == ParkingStatus.OCCUPIED:
        encoding = OCCUPIED_ENCODING
    elif status == ParkingStatus.IN_VIOLATION:
        encoding = IN_VIOLATION_ENCODING
    elif status == ParkingStatus.FINED:
        encoding = FINED_ENCODING
    else:
        raise AssertionError
    return encoding


def encode_spot_assignment(is_assigned: int, num_agents: int) -> List[int]:
    """
    Encodes the spot assignment value as a one-hot encoding, where 1 means that
    a spot is booked by an agent. The encoding has one column for each agent, and
    the columns respond to the agent ids.
    """
    encoding = [0] * num_agents
    # -1 means not assigned
    if is_assigned != -1:
        encoding[is_assigned] = 1
    return encoding


def encode_partial_status(status: ParkingStatus) -> List[int]:
    """Returns the one-hot encoding of a parking spot status in with partial observability"""
    if status == ParkingStatus.FREE:
        encoding = FREE_ENCODING_PARTIAL
    elif status == ParkingStatus.OCCUPIED:
        encoding = OCCUPIED_ENCODING_PARTIAL
    elif status == ParkingStatus.IN_VIOLATION:
        encoding = IN_VIOLATION_ENCODING_PARTIAL
    elif status == ParkingStatus.FINED:
        encoding = FINED_ENCODING_PARTIAL
    elif status == ParkingStatus.UNKNOWN:
        encoding = UNKNOWN_ENCODING_PARTIAL
    else:
        raise AssertionError
    return encoding


def prune_subgraphs(graph):
    """"Removes (weekly connected) subgraphs without any parking spots."""
    # extract subgraphs
    sub_graphs = [graph.subgraph(c).copy()
                  for c in nx.weakly_connected_components(graph)]
    # delete all subgraphs except the biggest one
    for subgraph in sub_graphs[1:]:
        # remove subgraphs without parking spots
        if get_num_spots(subgraph) == 0:
            graph.remove_nodes_from(subgraph)
        else:
            # either way remove the subgraph
            logging.info("Deleting graph with %i spots.", get_num_spots(graph))
            graph.remove_nodes_from(subgraph)
    return graph


def get_subgraphs(graph):
    """Returns the (weekly connected) subgraphs and some info."""
    # extract subgraphs
    sub_graphs = [graph.subgraph(c).copy()
                  for c in nx.weakly_connected_components(graph)]
    for i, subgraph in enumerate(sub_graphs):
        logging.info("Subgraph %i has:", i)
        logging.info("\tNodes: %i", len(subgraph.nodes(data=True)))
        logging.info("\tEdges: %i", len(subgraph.edges()))
        logging.info("\tSpots: %i", get_num_spots(subgraph))


def find_spot(graph, marker):
    """
    Finds the two nodes of an edge that contain the parking spot
    with the selected street marker.
    """
    for node_one, node_two, data in graph.edges(data=True):
        if "spots" in data:
            for spot in data["spots"]:
                if spot["id"] == marker:
                    return node_one, node_two
    raise ValueError("Unknown id.")


def update_spot(spot, event):
    """Updates a parking spot based on an event."""
    if event[TYPE_COLUMN] == EventType.ARRIVAL:
        spot["status"] = ParkingStatus.OCCUPIED
        spot["arrivalTime"] = datetime_to_sec(event[TIME_COLUMN])
        spot["maxSeconds"] = event[MAX_MINUTES_COLUMN] * 60
    elif event[TYPE_COLUMN] == EventType.DEPARTURE:
        spot["status"] = ParkingStatus.FREE
        spot["arrivalTime"] = 0
    elif event[TYPE_COLUMN] == EventType.VIOLATION:
        spot["status"] = ParkingStatus.IN_VIOLATION
    return spot


def update_spots(spots: list, events: list):
    """Updates the parking spots based on a number of events"""
    new_spots = []
    for node_one, node_two, spot in spots:
        new_spot = spot
        i = 0
        while i < len(events):
            if events[i][STREET_MARKER_COLUMN] == spot["id"]:
                new_spot = update_spot(new_spot, events[i])
                events.pop(i)
            else:
                i += 1
        new_spots.append((node_one, node_two, new_spot))
    return new_spots


def fine_spots(spots, node_from, node_to) -> Tuple[float, list]:
    """Collects all violations and fines them. Returns the reward and updated spots."""
    num_violations = 0
    new_spots = []
    for node_one, node_two, spot in spots:
        new_spot = spot
        if node_one == node_from and \
                node_two == node_to and \
                spot["status"] == ParkingStatus.IN_VIOLATION:
            num_violations += 1
            new_spot["status"] = ParkingStatus.FINED
        new_spots.append((node_one, node_two, new_spot))
    assert num_violations >= 0
    return float(num_violations), new_spots


def get_num_spots(graph):
    """Returns the number of parking spots in a graph."""
    spots = 0
    for _, _, data in graph.edges(data=True):
        if "spots" in data:
            for _ in data["spots"]:
                spots += 1
    return spots


def get_spots(graph):
    """Returns all parking spots"""
    spots = []
    for start_node, end_node, data in graph.edges(data=True):
        if "spots" in data:
            for spot in data["spots"]:
                spots.append((start_node, end_node, spot))
    return spots


def get_avg_walking_time(graph):
    """Returns the average walking time of all edges."""
    time = 0
    i = 0
    for _, _, data in graph.edges(data=True):
        time += data["havlen"]
        i += 1
    return time / i


def get_edges_w_spots(graph):
    """Returns all edges that contain parking spots in a graph."""
    edges = {}
    i = 0
    for node_one, node_two, data in graph.edges(data=True):
        if "spots" in data:
            edges[str(i)] = [node_one, node_two]
            i += 1
    return edges


ShortestPath = namedtuple('ShortestPath', ('length', 'route'))


def precompute_shortest_paths(graph: nx.DiGraph) -> dict:
    """Compute the shortest path from each node in the graph to each other node in the graph

    Returns:
        lookup, a dictionary of dictionaries of paths

    Usage:
        lookup[4][7] returns the shortest path from node 4 to node 7

        If you look up the path from a node to an edge, do the following:
            1. Loop up route from node to edge[0] (=start of edge)
                `path = lookup[node][edge[0]]`
            2. Append edge[1] (=end of edge) to the resulting list
                `path.append(edge[1])`

        If you search for the route from node to edge[1] directly you might get a route that doesn't
            pass the desired edge!
    """
    all_path_pairs = dict(nx.all_pairs_dijkstra_path(graph, weight="havlen"))

    lookup = dict()
    for start_node, all_single_paths in all_path_pairs.items():
        lookup[start_node] = dict()
        for end_node, route in all_single_paths.items():
            if len(route) == 1:
                lookup[start_node][end_node] = ShortestPath(length=0.0, route=route)
            else:
                length = 0.0
                for i in range(len(route) - 1):
                    prev_node = route[i]
                    next_node = route[i + 1]
                    length += graph[prev_node][next_node]["havlen"]

                lookup[start_node][end_node] = ShortestPath(length=length, route=route)

    return lookup


# pylint: disable=E1121
# pylint: disable=E1123
def get_route_to_edge(graph, node, edge):
    """Returns the shortest path that traverses a certain edge."""
    route = nx.shortest_path(graph, node, edge[0], weight="havlen")
    # route.append(edge[1])
    return route


def get_shortest_distance_to_edge(graph, node, edge):
    """Returns the length of the shortest path that traverses a certain edge.

    TODO: Use precomputed distances!
    """
    try:
        dist = nx.shortest_path_length(graph, node, edge[0], weight="havlen")
    except nx.exception.NetworkXNoPath:
        print(f"Not reachable: {node}, {edge[0]}, {edge}, {type(edge)}")
        return 1000000000
    return dist


def get_distance_matrix(graph) -> ndarray:
    """
    returns dist matrix
    shape: edge of action x resource
    """
    distance_matrix = []
    for edge in get_edges_w_spots(graph).values():
        dist = []
        for start_node, end_node, _ in get_spots(graph):
            if edge[0] == start_node and edge[1] == end_node:
                dist.append(0.0)
            else:
                dist.append(get_shortest_distance_to_edge(
                    graph, end_node, edge))
        distance_matrix.append(dist)

    return np.array(distance_matrix)


def get_min_max(graph):
    """Returns the minimum and maximum coordinates of the graph"""
    min_x = list(graph.edges)[0][0][0]
    max_x = list(graph.edges)[0][0][0]
    min_y = list(graph.edges)[0][0][1]
    max_y = list(graph.edges)[0][0][1]

    for source_node, destination_node in graph.edges:
        if max_x < source_node[0]:
            max_x = source_node[0]
        if source_node[0] < min_x:
            min_x = source_node[0]
        if max_y < source_node[1]:
            max_y = source_node[1]
        if source_node[1] < min_y:
            min_y = source_node[1]
        if max_x < destination_node[0]:
            max_x = destination_node[0]
        if destination_node[0] < min_x:
            min_x = destination_node[0]
        if max_y < destination_node[1]:
            max_y = destination_node[1]
        if destination_node[1] < min_y:
            min_y = destination_node[1]

    return min_x, max_x, min_y, max_y


def ping(host):
    """
    Raises an error if the host does not respond to a ping request.
    Else, nothing happens.
    """
    # Ping parameters as function of OS
    ping_str = "-n 1" if platform.system().lower() == "windows" else "-c 1"
    args = "ping" + " " + ping_str + " " + host
    need_sh = platform.system().lower() != "windows"

    # Check if ping was successful
    return_code = subprocess.call(args, shell=need_sh,
                                  stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL)
    # Either ping was successful or ping utility is not installed
    if return_code not in (0, 127):
        raise ServerNotRespondingException(host)


def load_shortest_path_lookup(graph):
    """
    Try to load a pickled version from SHORTEST_PATHS_LOOKUP_PATH (takes ~1 minute),
    if that fails, compute it from scratch (takes ~5 minutes) and cache a pickled version
    to the file system (size of the pickled file is around 1.6 GB)
    """
    shortest_path_lookup = None
    try:
        with open(SHORTEST_PATHS_LOOKUP_PATH, "rb") as file:
            logging.info("Loading cached lookup table for for all shortest paths from %s, \
this will take up to two minutes...", SHORTEST_PATHS_LOOKUP_PATH)
            shortest_path_lookup = pickle.load(file)
    except FileNotFoundError:
        logging.warning(
            "Creating lookup table for all shortest paths as no cached version was found."
            "This will take approx. 5-10 minutes...")
        shortest_path_lookup = precompute_shortest_paths(graph)
        with open(SHORTEST_PATHS_LOOKUP_PATH, "wb") as outfile:
            pickle.dump(shortest_path_lookup, outfile)
        logging.info(
            "Shortest paths lookup table successfully created and cached for future use at %s.",
            SHORTEST_PATHS_LOOKUP_PATH)
    return shortest_path_lookup
