"""
Module contains the functionality to convert a parking dataset into an event log.
"""
import logging
from os import path

import networkx as nx
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame

from envs.enums import EventType
from envs.utils import TYPE_COLUMN

EVENT_TYPE_CONVERSION = {"Arrival": EventType.ARRIVAL,
                         "Departure": EventType.DEPARTURE,
                         "Violation": EventType.VIOLATION}


class EventLog:
    """
    Wrapper class which converts a dataset to an event log.
    """

    def __init__(self, dataframe: DataFrame, filename: str, graph_filename: str) -> None:
        """
        Initializes a new instance. If filename is not None, the event log will be loaded from the
        filesystem, otherwise it will be newly created with the specified dataframe.

        :param dataframe: The data on which the event log is based on.
        :param filename: The filepath to an already created event log.
        :param graph_filename: The filepath to the underlying graph on which the data is based.
        """
        if filename is not None and path.exists(filename):
            # Read event log from file
            self.event_log = np.load(filename, allow_pickle=True)
        elif dataframe is not None:
            # Create new event log
            self.dataframe = dataframe.dropna(
                subset=["ArrivalTime", "DepartureTime", "StreetMarker"])
            self._remove_data_with_negative_duration(graph_filename)
            # Save to file
            np.save(filename, self.event_log)
        else:
            # dataframe not provided or event log missing
            raise ValueError("Dataframe missing.")

    def calculate_max_hours(self) -> None:
        """
        Calculates the max hours a car can stay in a parking spot
        based on the parking sign.
        """
        self.dataframe["Sign"] = self.dataframe["Sign"].replace({np.nan: "None"})
        self.dataframe["MaxMinutes"] = np.select(
            [
                self.dataframe["Sign"].str.startswith("None"),
                self.dataframe["Sign"].str.startswith("P5"),
                self.dataframe["Sign"].str.startswith("P10"),
                self.dataframe["Sign"].str.startswith("P/10"),
                self.dataframe["Sign"].str.startswith("1/4P"),
                self.dataframe["Sign"].str.startswith("1/2P"),
                self.dataframe["Sign"].str.startswith("1/2"),
                self.dataframe["Sign"].str.startswith("1P"),
                self.dataframe["Sign"].str.startswith("2P"),
                self.dataframe["Sign"].str.startswith("3P"),
                self.dataframe["Sign"].str.startswith("4P"),
                self.dataframe["Sign"].str.startswith("LZ 15M"),
                self.dataframe["Sign"].str.startswith("LZ 30M"),
                self.dataframe["Sign"].str.contains("60mins"),
                self.dataframe["Sign"].str.contains("30MINS"),
                self.dataframe["Sign"].str.contains("15mins"),
                self.dataframe["Sign"].str.contains("15Mins"),
                self.dataframe["Sign"].str.contains("1PMTR"),
            ],
            [
                24 * 60,
                5,
                10,
                10,
                15,
                30,
                30,
                60,
                120,
                180,
                240,
                15,
                30,
                60,
                30,
                15,
                15,
                60
            ],
            default="Unknown"
        )
        # Drop rows with unknown parking
        self.dataframe = self.dataframe[self.dataframe["MaxMinutes"] != "Unknown"]
        self.dataframe['MaxMinutes'] = self.dataframe['MaxMinutes'].astype(float)

    def create_event_log(self) -> DataFrame:
        """
        Creates the event log created from a parking dataset.

        :return: The newly created event log.
        """
        logging.info("Creating event log from dataset.")

        # Remove events without cars
        self.dataframe = self.dataframe[self.dataframe["Vehicle Present"] == 1]

        # create individual event logs
        arrivals = self._create_arrivals_dataframe()
        departures = self._create_departures_dataframe()
        violations = self._create_violations_dataframe()

        # Combine
        event_log = arrivals.append(departures).append(violations)
        event_log.sort_values(by=["Time"], inplace=True)
        event_log.reset_index(drop=True, inplace=True)

        return event_log

    def _remove_data_with_negative_duration(self, graph_filename) -> None:
        """
        Remove data with negative duration (falsely recorded).

        :param graph_filename: The filepath to the underlying graph on which the data is based.
        :return: None.
        """
        self.dataframe = self.dataframe[self.dataframe["DurationSeconds"] >= 0]
        self.dataframe = _remove_redundant_events(self.dataframe, graph_filename)
        self.calculate_max_hours()
        self.event_log = self.create_event_log()
        self.event_log = _to_numpy(self.event_log)

    def _create_arrivals_dataframe(self) -> DataFrame:
        """
        Creates a dataframe for arrival events.

        :return: The newly created arrival event log.
        """
        arrivals = self.dataframe[["ArrivalTime", "StreetMarker", "MaxMinutes"]].copy()
        arrivals.rename({"ArrivalTime": "Time"}, axis=1, inplace=True)
        arrivals["Type"] = "Arrival"
        return arrivals

    def _create_departures_dataframe(self) -> DataFrame:
        """
        Creates a dataframe for departure events.

        :return: The newly created departure event log.
        """
        departures = self.dataframe[['DepartureTime', 'StreetMarker']].copy()
        departures.rename({"DepartureTime": "Time"}, axis=1, inplace=True)
        departures["Type"] = "Departure"
        departures["MaxMinutes"] = 0
        return departures

    def _create_violations_dataframe(self) -> DataFrame:
        """
        Creates a dataframe for violation events.

        :return: The newly created violations event log.
        """
        violations = self.dataframe[self.dataframe["In Violation"] == 1.0][
            ['ArrivalTime', 'StreetMarker', "MaxMinutes"]].copy()
        violations["Time"] = violations.apply(
            lambda x: x["ArrivalTime"] + pd.DateOffset(minutes=x["MaxMinutes"]), axis=1)
        violations.drop(['ArrivalTime'], axis=1, inplace=True)
        violations["Type"] = "Violation"
        return violations


def _to_numpy(event_log: pd.DataFrame) -> np.array:
    """
    Converts the event log dataframe to a numpy array and removes redundant spots.

    :param event_log: The event log to be converted.
    :return: The event log as numpy.array.
    """
    # Time, StreetMarker,   MaxMinutes,     Type
    # 0,    1,              2,              3
    event_log = np.array(event_log.values.tolist())
    event_log[:, TYPE_COLUMN] = [EVENT_TYPE_CONVERSION[t] for t in event_log[:, TYPE_COLUMN]]
    return event_log


def _remove_redundant_events(event_log: np.array, graph_filename: str) -> np.array:
    """
    Removes the events from the event_log where the StreetMarker does not match any parking spots
    from the graph.

    :param event_log: The event log to be checked.
    :param graph_filename: The filepath to the underlying graph on which the data is based.
    :return: The checked event log without redundant events.
    """
    graph: nx.DiGraph = nx.read_gpickle(graph_filename)
    spots: list = []

    for _, _, data in graph.edges(data=True):
        if "spots" in data:
            for spot in data["spots"]:
                spots.append(spot["id"])

    event_log = event_log[event_log["StreetMarker"].isin(spots)]
    return event_log
