"""
Module contains data loading functionality for the Melbourne On-street Car
Parking Sensor Data - 2017 dataset.
"""
import datetime as dt
import logging
from abc import ABC
from enum import Enum

import pandas as pd
from pandas.core.frame import DataFrame

PATH_TO_DATA = '../data/On-street_Car_Parking_Sensor_Data_-_2017.csv'
PATH_TO_DOCKLANDS_DATA = '../data/Docklands.csv'
PATH_TO_QUEENSBERRY_DATA = '../data/Queensberry.csv'


class DataSplit(Enum):
    """
    Describes the different splits of the data.
    """
    TRAINING = 0
    TEST = 1
    VALIDATION = 2


class Data(ABC):
    """
    Base class for data loading. Shouldn't be instantiated alone.
    """

    def __init__(self):
        self.dataframe = None

    def get_data(self):
        """
        Returns the given data split.
        """
        return self.dataframe


class MelbourneData(Data):
    """
    This class provides data loading functionality for the entire dataset.
    """

    def __init__(self):
        super().__init__()
        self.dataframe = _load_data(PATH_TO_DATA)


class QueensberryData(Data):
    """
    This class provides data loading functionality for the Queensberry subset
    of the dataset.
    """

    def __init__(self):
        super().__init__()
        self.dataframe = _load_data(PATH_TO_QUEENSBERRY_DATA)
        self.dataframe = self.dataframe.loc[self.dataframe['Area'] == 'Queensberry']


class DocklandsData(Data):
    """
    This class provides data loading functionality for the Docklands subset
    of the dataset.
    """

    def __init__(self):
        super().__init__()
        self.dataframe = _load_data(PATH_TO_DOCKLANDS_DATA)
        self.dataframe = self.dataframe.loc[self.dataframe['Area'] == 'Docklands']


class DowntownData(Data):
    """
    This class provides data loading functionality for the Downtown subset
    of the dataset. Downtown consists of several smaller areas.
    """

    def __init__(self):
        super().__init__()
        self.dataframe = _load_data('')  # change when used
        downtown_areas = ['Magistrates', 'Titles', 'The Mac', 'Regency',
                          'County', 'Supreme', 'Hardware', 'Chinatown',
                          'Princes Theatre', 'Spencer', 'Rialto', 'RACV',
                          'Tavistock', 'Banks', 'City Square', 'Hyatt']
        self.dataframe = \
            self.dataframe.loc[self.dataframe['Area'].isin(downtown_areas)]


def _load_data(filepath: str) -> pd.DataFrame:
    logging.info("Loading dataset from path: %s", filepath)
    dataframe = pd.read_csv(filepath,
                            dtype={
                                "DeviceId": object,
                                "In Violation": int,
                                "Vehicle Present": int})
    dataframe['ArrivalTime'] = pd.to_datetime(dataframe['ArrivalTime'])
    dataframe['DepartureTime'] = pd.to_datetime(dataframe['DepartureTime'])
    dataframe = dataframe[dataframe["DurationSeconds"] > 0]
    return dataframe


def convert_datetime_to_ms(dataframe: DataFrame):
    """Converts datetime columns to milliseconds since 01.01.2017"""
    for col in ["ArrivalTime", "DepartureTime"]:
        dataframe[col] = (pd.to_datetime(dataframe[col]) -
                          dt.datetime(2017, 1, 1)).dt.total_seconds()
    dataframe.sort_values(by="ArrivalTime", inplace=True)
    return dataframe
