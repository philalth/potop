"""
Stub that imports the osm graph and adds parking spots to its edges
"""
import networkx as nx
import numpy as np
from graph import shrink_graph as shrink
from graph import import_osm_graph as osmp  # Import this module
from graph import dist2street as d2s
from datasets.parking_bays import ParkingBays
from envs.utils import ParkingStatus, prune_subgraphs, ensure_strongly_connected, get_subgraphs

OSM_FILE = "../data/planet_144.86,-37.852_145.033,-37.767.osm"
PARKING_SPOTS_FILE = "../data/Off-street_car_parking_2017_map.csv"

# pylint: disable=C0103


def assing_parking_spots(graph, spots):
    """Assigns parking spots to closest edges"""

    # iterate over all monitored parking spots
    for _, row in spots.iterrows():
        # check if parking spot has location information
        if np.isnan(row["y"]) or np.isnan(row["x"]):
            continue
        CLOSEST_DIST = 10000
        closest_edge = (0, 0)
        # search for the closest edge for each parking spot
        for u, v in graph.edges():
            dist = d2s.pnt2line((row["y"], row["x"]),
                                (graph.nodes[u]['lat'],
                                graph.nodes[u]['lon']),
                                (graph.nodes[v]['lat'],
                                graph.nodes[v]['lon']))[0]
            if dist < CLOSEST_DIST:
                CLOSEST_DIST = dist
                closest_edge = (u, v)

        # get the start and end node, for the closest edge
        u, v = closest_edge

        spot_object = {
            "id": row["marker_id"],
            "status": ParkingStatus.FREE,
            "arrivalTime": 0
        }

        # if its the first parking spot at this edge, add spots attribute
        # else add spot object to spot list
        if 'spots' in graph.edges[u, v]:
            spots = graph.edges[u, v]["spots"].copy()
            spots.append(spot_object)
            graph.edges[u, v].update({"spots": spots})
        else:
            graph.edges[u, v].update({"spots": [spot_object]})

    return graph


def fix_dead_end_nodes(graph):
    """
    Adds an out_edge to each node with degree 0 in order to avoid dead ends.
    """
    for node in graph.nodes:
        if graph.out_degree(node) == 0:
            for node_one, node_two in graph.in_edges(node):
                graph.add_weighted_edges_from(
                    [(node_two,
                      node_one,
                      graph[node_one][node_two]["havlen"])],
                    weight="havlen")
    return graph


def create_graph(disctricts, filename):
    """Creates a graph using a OpenStreetMaps file and sensor data."""
    print("Creating graph for districts: ", disctricts)

    with open(OSM_FILE, mode='r', encoding='utf-8') as f:
        osm_data = f.read()

    # Might take around two minutes
    graph = osmp.read_osm(osm_data)

    # load parking spots into data frame
    spots = ParkingBays().dataframe
    spots = spots[spots["Area"].isin(disctricts)]
    print("Number of parking spots in selected district: ", len(spots))

    # only look at the important part of the graph
    # spot is the data frame with the parking spots
    # if one wants to use a subset e.g. downtown
    # just just create a dataframe withe only those spots
    graph = shrink.shrink_by_df(spots, graph)
    graph = assing_parking_spots(graph, spots)
    graph = fix_dead_end_nodes(graph)
    get_subgraphs(graph)
    graph = prune_subgraphs(graph)
    nx.write_gpickle(graph, filename)
    graph = ensure_strongly_connected(graph)

    # save graph as .gpickle
    nx.write_gpickle(graph, filename)
