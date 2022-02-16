"""
Convert a Open Street Maps `.map` format file into a networkx directional graph
This parser is based on the osm to networkx tool from aflaxman:
https://gist.github.com/aflaxman/287370/

Added :
- python3.6 compatibility
- networkx v2 compatibility
- cache to avoid downloading the same osm tiles again and again
- distance computation to estimate length of each ways
    (useful to compute the shortest path)
Copyright (C) 2017 Loïc Messal (github : Tofull)

Edited for the PBDS course, 2021.
Refer to the jupyter notebook for additional information on the graph,
nodes and edges.


Example usage:
```
import import_osm_graph as osmp    # Import this module

osmFile = "./path/to/file"
with open(osmFile, mode='r', encoding='utf-8') as f:
    content = f.read()

# Might take some time
graph = osmp.read_osm(content)

print(type(graph))
```
"""

import copy
import xml.sax
from math import radians, cos, sin, asin, sqrt

import networkx


def haversine_distance(lon1, lat1, lon2, lat2, unit_m=True):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    default unit : km
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    temp_1 = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    temp_2 = 2 * asin(sqrt(temp_1))
    radius = 6371  # Radius of the Earth in kilometers. Use 3956 for miles
    if unit_m:
        radius *= 1000
    return temp_2 * radius


def read_osm(osm_xml_data, is_xml_string=True, only_roads=True):
    """Read graph in OSM format from file specified by name or by stream object.
    Parameters
    ----------
    filename_or_stream : filename or stream object
    Returns
    -------
    G : Graph
    Examples
    --------
    >>> G=nx.read_osm(nx.download_osm(-122.33,47.60,-122.31,47.61))
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(   [G.node[n]['lat'] for n in G],
                    [G.node[n]['lon'] for n in G],
                    'o',
                    color='k')
    >>> plt.show()
    """
    osm = OSM(osm_xml_data, is_xml_string=is_xml_string)
    graph = networkx.DiGraph()

    # Add ways
    for way in osm.ways.values():
        if only_roads and 'highway' not in way.tags:
            continue

        if 'oneway' in way.tags:
            if way.tags['oneway'] == 'yes':
                # ONLY ONE DIRECTION
                networkx.add_path(graph, way.nds, id=way.way_id)
            else:
                # BOTH DIRECTION
                networkx.add_path(graph, way.nds, id=way.way_id)
                networkx.add_path(graph, way.nds[::-1], id=way.way_id)
        else:
            # BOTH DIRECTION
            networkx.add_path(graph, way.nds, id=way.way_id)
            networkx.add_path(graph, way.nds[::-1], id=way.way_id)

    # Complete the used nodes' information
    coordinates_map = {}
    for n_id in graph.nodes():
        node = osm.nodes[n_id]
        graph.nodes[n_id]['lat'] = node.lat
        graph.nodes[n_id]['lon'] = node.lon
        graph.nodes[n_id]['id'] = node.node_id
        coordinates_map[n_id] = (node.lon, node.lat)

    # Estimate the length of each way
    for start, end, _dist in graph.edges(data=True):
        # Give a realistic distance estimation
        # (neither EPSG nor projection nor reference system are specified)
        distance = haversine_distance(graph.nodes[start]['lon'],
                                      graph.nodes[start]['lat'],
                                      graph.nodes[end]['lon'],
                                      graph.nodes[end]['lat'],
                                      unit_m=True)

        graph.add_weighted_edges_from([(start, end, distance)],
                                      weight='havlen')

    graph = networkx.relabel_nodes(graph, coordinates_map)
    return graph


# pylint: disable=R0903
class Node:
    """ Temporary class for modelling nodes during import """
    def __init__(self, node_id, lon, lat):
        self.node_id = node_id
        self.lon = lon
        self.lat = lat
        self.tags = {}

    def __str__(self):
        return "Node (id : %s) lon : %s, lat : %s " % \
            (self.node_id, self.lon, self.lat)


# pylint: disable=R0903
class Way:
    """ Temporary class for modelling edges during import """
    def __init__(self, way_id, osm):
        self.osm = osm
        self.way_id = way_id
        self.nds = []
        self.tags = {}

    def split(self, dividers):
        """ Split a way """
        def slice_array(array, dividers):
            """ Slice the node-array using this nifty recursive function """
            for i in range(1, len(array)-1):
                if dividers[array[i]] > 1:
                    left = array[:i+1]
                    right = array[i:]

                    rightsliced = slice_array(right, dividers)

                    return [left]+rightsliced
            return [array]

        slices = slice_array(self.nds, dividers)

        # create a way object for each node-array slice
        ret = []
        i = 0
        for way_slice in slices:
            littleway = copy.copy(self)
            littleway.way_id += "-%d" % i
            littleway.nds = way_slice
            ret.append(littleway)
            i += 1

        return ret


# pylint: disable=R0903
class OSM:
    """ Parser for the osm_xml file """
    def __init__(self, osm_xml_data, is_xml_string=True):
        """ File can be either a filename or stream/file object.
        set `is_xml_string=False` if osm_xml_data is a filename or
            a file stream.
        """
        nodes = {}
        ways = {}

        superself = self

        class OSMHandler(xml.sax.ContentHandler):
            """ Helper class to handle XML parsing """
            # pylint: disable=W0221
            @classmethod
            def setDocumentLocator(cls, loc):
                pass

            @classmethod
            def startDocument(cls):
                pass

            @classmethod
            def endDocument(cls):
                pass

            @classmethod
            def startElement(cls, name, attrs):
                if name == 'node':
                    cls.currElem = Node(attrs['id'], float(
                        attrs['lon']), float(attrs['lat']))
                elif name == 'way':
                    cls.currElem = Way(attrs['id'], superself)
                elif name == 'tag':
                    cls.currElem.tags[attrs['k']] = attrs['v']
                elif name == 'nd':
                    cls.currElem.nds.append(attrs['ref'])

            @classmethod
            def endElement(cls, name):
                if name == 'node':
                    nodes[cls.currElem.node_id] = cls.currElem
                elif name == 'way':
                    ways[cls.currElem.way_id] = cls.currElem

            # pylint: disable=W0221
            @classmethod
            def characters(cls, chars):
                pass

        if is_xml_string:
            xml.sax.parseString(osm_xml_data, OSMHandler)
        else:
            with open(osm_xml_data, mode='r') as file:
                xml.sax.parse(file, OSMHandler)

        self.nodes = nodes
        self.ways = ways

        # count times each node is used
        node_histogram = dict.fromkeys(self.nodes.keys(), 0)
        # Line below was changed to iterate over a copy, prevents runtime error
        for way in self.ways.copy().values():
            # if a way has only one node, delete it out of the osm collection
            if len(way.nds) < 2:
                del self.ways[way.way_id]
            else:
                for node in way.nds:
                    node_histogram[node] += 1

        # use that histogram to split all ways, replacing the
        # member set of ways
        new_ways = {}
        for _, way in self.ways.items():
            split_ways = way.split(node_histogram)
            for split_way in split_ways:
                new_ways[split_way.way_id] = split_way
        self.ways = new_ways
