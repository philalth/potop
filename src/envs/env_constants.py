"""
Environment constants.
"""

from typing import List, Dict

from envs.enums import EventType
from envs.mlflow_logging import MlFlowLogger

# general constants
MLFLOW_LOGGER: MlFlowLogger = MlFlowLogger()
UNDISCOUNTED_REWARD: int = 0
DISCOUNTED_REWARD: int = 1

# one-hot encoding of the parking status
FREE_ENCODING: List[int] = [1, 0, 0, 0]
OCCUPIED_ENCODING: List[int] = [0, 1, 0, 0]
IN_VIOLATION_ENCODING: List[int] = [0, 0, 1, 0]
FINED_ENCODING: List[int] = [0, 0, 0, 1]

# one-hot encoding of the parking status for partial observability
FREE_ENCODING_PARTIAL: List[int] = [1, 0, 0, 0, 0]
OCCUPIED_ENCODING_PARTIAL: List[int] = [0, 1, 0, 0, 0]
IN_VIOLATION_ENCODING_PARTIAL: List[int] = [0, 0, 1, 0, 0]
FINED_ENCODING_PARTIAL: List[int] = [0, 0, 0, 1, 0]
UNKNOWN_ENCODING_PARTIAL: List[int] = [0, 0, 0, 0, 1]

# event log columns
TIME_COLUMN: int = 0
STREET_MARKER_COLUMN: int = 1
MAX_MINUTES_COLUMN: int = 2
TYPE_COLUMN: int = 3

EVENT_TYPE_CONVERSION: Dict[str, EventType] = {"Arrival": EventType.ARRIVAL,
                                               "Departure": EventType.DEPARTURE,
                                               "Violation": EventType.VIOLATION}
