"""
Used to shrink a networkx graph to the size of a convex hull defined by
the coordinates of the parking spots
"""

import scipy.spatial.qhull
import numpy as np
import networkx as nx


def shrink_by_df(spots_df, graph):
    """
    spots: dataframe of parking spot
    graph: graph to shrink to the size of the convex hull
    """
    # get list of tuples with parking spot coordinates
    spot_coordinates = [tuple(x) for x in
                        spots_df[['x', 'y']].to_numpy()]
    # remove spots without coordinates
    spot_coordinates = [x for x in spot_coordinates if str(x[0]) != 'nan']

    spot_coordinates = [list(i) for i in spot_coordinates]
    spot_coordinates_array = np.asarray(spot_coordinates)
    return shrink(spot_coordinates_array, graph, margin=0.001)


def shrink(points_to_generate_hull, graph, margin=0.0):
    """
    hull_points: points that define a convex hull as array
    graph: graph to shrink to the size of the convex hull
    margin: margin to increase the convex hull, 0 for original
    """

    # convex hull of all the parking spots
    spot_hull = scipy.spatial.qhull.ConvexHull(points_to_generate_hull)

    # get points to generate a new (greater) convex hull
    # for each point defining the original convex hull add one point to the
    # top-right, top-left, bottom-right and bottom left
    points_to_generate_increased_hull = []
    for vertex in points_to_generate_hull[spot_hull.vertices]:
        points_to_generate_increased_hull.append([vertex[0] + margin, vertex[1] + margin])
        points_to_generate_increased_hull.append([vertex[0] + margin, vertex[1] - margin])
        points_to_generate_increased_hull.append([vertex[0] - margin, vertex[1] + margin])
        points_to_generate_increased_hull.append([vertex[0] - margin, vertex[1] - margin])

    # generate the increased convex hull for all of those points
    increased_hull = scipy.spatial.qhull.ConvexHull(points_to_generate_increased_hull)
    points_to_generate_increased_hull = np.array(points_to_generate_increased_hull)

    # create delaunay triangulation of the hull
    hull_delaunay = scipy.spatial.qhull.Delaunay(
        points_to_generate_increased_hull[increased_hull.vertices])

    edges_outside = []  # edges we do not need
    # iterate over all the edges in the graph and
    # decide if we keep or remove them
    for source_node, destination_node in graph.edges():
        if hull_delaunay.find_simplex(source_node) < 0 and \
                hull_delaunay.find_simplex(destination_node) < 0:
            # remove edge
            # store the edges we do not need anymore
            edges_outside.append([source_node, destination_node])
        else:
            # keep edge
            pass

    # remove edges with no node inside the convex hull
    for source_node, destination_node in edges_outside:
        graph.remove_edge(source_node, destination_node)

    # remove nodes without an edge
    graph.remove_nodes_from(list(nx.isolates(graph)))

    return graph
