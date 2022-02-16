"""
Module contains custom enums used by the environment.
"""

from enum import Enum


class ObservationMode(Enum):
    """Describes the observation mode of the environment."""
    FULL = 0
    PARTIAL = 1


class ParkingStatus(Enum):
    """Describes the status of the parking space."""
    FREE = 0
    OCCUPIED = 1
    IN_VIOLATION = 2
    FINED = 3
    UNKNOWN = 4


class EventType(Enum):
    """Describes the type of the agent event."""
    ARRIVAL = 0
    DEPARTURE = 1
    VIOLATION = 2


class CustomAction(Enum):
    """Signals the environment to take a custom action instead of going to a specific edge."""
    RANDOM_EDGE = 0
    SHORTEST_EDGE = 1
