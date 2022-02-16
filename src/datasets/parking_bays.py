"""
Module contains data loading functionality for the Melbourne On-street
Parking Bays dataset.
"""

import pandas as pd
import shapely.wkt
from datasets.datasets import MelbourneData

PATH_TO_DATA = '../data/parking_bays.csv'

# pylint: disable=R0903


class ParkingBays:
    """The Melbourne On-street Parking Bays dataset."""

    def __init__(self):
        self.dataframe = pd.read_csv(PATH_TO_DATA)
        self.dataframe.dropna(subset=["marker_id"], inplace=True)
        self.dataframe = _calculate_centroids(self.dataframe)
        self.dataframe = _add_area(self.dataframe)


def _add_area(dataframe):
    """Add area of sensor based on other datasets."""
    melbourne = MelbourneData(
    ).dataframe[["StreetMarker", "Area"]].drop_duplicates()
    return pd.merge(dataframe, melbourne, left_on="marker_id",
                    right_on="StreetMarker", how='left')


def _calculate_centroids(dataframe):
    """Calculates x and y coordinates of multipolygons."""
    dataframe["x"] = dataframe["the_geom"].apply(
        lambda x: shapely.wkt.loads(x).centroid.x)
    dataframe["y"] = dataframe["the_geom"].apply(
        lambda x: shapely.wkt.loads(x).centroid.y)
    return dataframe
