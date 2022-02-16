"""Module contains various settings."""
import logging
from typing import List, Any, Dict, Union

import pandas as pd
import torch
from ray.tune.schedulers.median_stopping_rule import MedianStoppingRule
from ray.tune.schedulers.trial_scheduler import TrialScheduler

from envs.enums import ObservationMode

# Local testing setting.
USE_LOCAL_SETTINGS: bool = False
SMALL_EVENT_LOG: bool = False  # use only events in the first week

# Environment settings.
SEED: int = 1789
NUM_AGENTS: int = 1
NUM_EPISODES: int = 25
WALKING_SPEED: int = 5
START_HOUR: int = 7
END_HOUR: int = 19
INITIAL_TIMESTAMP: pd.Timestamp = pd.Timestamp(2017, 1, 1, 7)
FINAL_TIMESTAMP: pd.Timestamp = pd.Timestamp(2018, 1, 1, 7)
OBSERVATION_MODE: ObservationMode = ObservationMode.FULL
CACHED_SHORTEST_PATHS: bool = not USE_LOCAL_SETTINGS
DISTRICTS: List[str] = ["Docklands"]
NUM_EPISODES_BEFORE_VALIDATION: int = 100
USE_NINTH_COLUMN_ALLOWED_PARKING_TIME: bool = False
SAVE_MODEL: bool = True
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GAMMA: float = 0.999  # reward discount factor

# Observation settings
AGENTS_SHARE_OBSERVATION: bool = False  # if True all agents share their observation among them
USE_ALLOWED_PARKING_TIME: bool = False
USE_SPOT_ASSIGNMENT_COLUMN: bool = False  # needed for DDQN single shared policy
USE_CAR_ARRIVAL_TIME: bool = False  # needed for the greedy agent

# Pre-trained models.
TEST_MODEL: bool = False  # If True, only test episodes will be run.
LOAD_MODEL: bool = False
DDQN_MODEL: Any = None
PPO_MODEL: Any = None

# Tuning.
TUNE: bool = (not USE_LOCAL_SETTINGS) and (not TEST_MODEL)
TUNE_LOCAL: bool = False
TUNE_NAME: str = "test"
TUNE_DIR: str = "../tune-results"
NUM_SAMPLES: int = 40
RESOURCES: Dict[str, Union[int, float]] = {"cpu": 1, "gpu": 0.2}
SCHEDULER: TrialScheduler = MedianStoppingRule(
    time_attr="episode",
    metric="episode_reward",
    mode="max",
    grace_period=10,
    min_samples_required=10
)

# ML Flow.
MLFLOW_TRACKING: bool = not USE_LOCAL_SETTINGS
MLFLOW_TRACKING_URI: str = "http://10.195.5.17:8001"

# Logging.
LOG_LEVEL: int = logging.DEBUG

# Files.
GRAPH_FILENAME: str = "../data/graphs/" + DISTRICTS[0].lower() + ".gpickle"
EVENT_LOG_PATH: str = "../data/event_logs/" + DISTRICTS[0].lower() + ".npy"
SHORTEST_PATHS_LOOKUP_PATH: str = "../data/shortest_paths_lookup_" + DISTRICTS[0] + ".pickle"
SAVE_MODEL_PATH: str = "../data/models/"

# Rendering
RENDER: bool = False
SCREEN_WIDTH: int = 600
SCREEN_HEIGHT: int = 400
PARTIAL_RENDERING: bool = False
AGENTS_VIEW: int = 0  # ID of the agent whose partial view you want to observe
SAVE_IMAGES: bool = False
VIDEO_FOLDER: str = "../videos"
SAVE_INTERVAL: int = 70  # save every nth day to video file, 1 for whole simulation
