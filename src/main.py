"""The main simulation loop.

Runs the simulation loop, using the specified agent to make decisions.
Detailed settings can be set in the file config/settings.py
Hyperparameters for the agents can be set in the file config/params.py

Usage:
    python3 main.py AGENT_NAME EXPERIMENT_NAME YOUR_NAME

Required arguments:
    AGENT_NAME: the type of agent you want to use.
                Valid names are: random, greedy, aco, ddqn, ppo
    EXPERIMENT_NAME: the MLFlow experiment that you want your runs to be logged as.
                If you are only testing, pass e.g. "test" as experiment name.
    YOUR_NAME: your own name (used for attributing who started a remote run)
"""

import argparse
import logging
import os
import platform
import random
import sys
from typing import List, Tuple, Any

import mlflow
import numpy as np
import ray
import torch
from numpy import ndarray
from ray import tune
from ray.tune.integration.mlflow import mlflow_mixin

from agents.aco import ACO
from agents.agent import Agent
from agents.ddqn.ddqn import DDQN
from agents.greedy import Greedy
from agents.multi.coma import COMA
from agents.ppo import PPO
from agents.random import RandomAgent, Strategy
from config.models import get_model
from config.params import ppo_params, ppo_params_tune, ddqn_params, ddqn_params_tune, \
    aco_params, coma_params
from config.settings import LOAD_MODEL, LOG_LEVEL, NUM_EPISODES, \
    TUNE, GRAPH_FILENAME, MLFLOW_TRACKING, MLFLOW_TRACKING_URI, TUNE_DIR, TUNE_LOCAL, TUNE_NAME, \
    OBSERVATION_MODE, SEED, RENDER, SAVE_MODEL_PATH, SCHEDULER, TEST_MODEL, \
    NUM_EPISODES_BEFORE_VALIDATION, NUM_SAMPLES, CACHED_SHORTEST_PATHS, USE_ALLOWED_PARKING_TIME, \
    USE_LOCAL_SETTINGS, DISTRICTS, EVENT_LOG_PATH, RESOURCES, SMALL_EVENT_LOG, \
    USE_SPOT_ASSIGNMENT_COLUMN, NUM_AGENTS
from datasets.datasets import DataSplit, DocklandsData, QueensberryData
from datasets.event_log import EventLog
from envs.env_constants import MLFLOW_LOGGER, DISCOUNTED_REWARD, UNDISCOUNTED_REWARD
from envs.exceptions import InvalidAgentAlgorithm, InvalidCommandLineArgsException
from envs.potop_env import PotopEnv
from envs.utils import TIME_COLUMN, ping
from graph.main_graph import create_graph


@mlflow_mixin
def simulate(config: dict) -> None:
    """Performs a simulation of the environment."""
    logging.info('Starting simulation.')

    _set_seeds()
    env: PotopEnv = _initialize_environment()

    MLFLOW_LOGGER.log_config(config)

    # This mapping is needed by the greedy agent
    config["params"]["spot_to_edge_mapping"] = env.get_spot_to_edge_mapping()
    # This reference is needed by the DDQN agent
    config["params"]["graph"] = env.graph

    use_shared_policy: bool = config["params"].get("shared_policy") is True

    agent: Any = None
    agents: List[Agent] = []

    if use_shared_policy:
        agent = config["agent"](env.action_space,
                                env.observation_space,
                                config["params"])
        load_model(agent, config["models"])
    else:
        for index in range(NUM_AGENTS):
            config["params"]["agent_number"] = index
            agents.append(config["agent"](
                env.action_space,
                env.observation_space,
                config["params"]))
        load_model(agents, config["models"])
    total_reward: int = 0

    # due to the modulo operation the counting has to start at 1
    # add artificial last episode for the test set
    for episode in range(1, NUM_EPISODES + 1):
        if TEST_MODEL:
            is_training_episode: bool = False
            is_test_episode: bool = True
        else:
            is_training_episode: bool = episode % (NUM_EPISODES_BEFORE_VALIDATION + 1) != 0
            is_test_episode: bool = episode == NUM_EPISODES + 2

        if use_shared_policy:
            agent.set_test(not is_training_episode)
        else:
            for agent in agents:
                agent.set_test(not is_training_episode)

        data_split: DataSplit = _set_data_split(env, is_test_episode, is_training_episode)

        num_steps, undiscounted_reward, discounted_reward = \
            _run_episode(env, agent, agents, episode, is_training_episode, use_shared_policy)

        total_reward: float = total_reward + undiscounted_reward
        logging.debug('Episode finished after %i steps', num_steps)
        logging.debug('Undiscounted reward: %i', undiscounted_reward)
        logging.debug('Discounted reward: %i', discounted_reward)

        MLFLOW_LOGGER.log_episode_reward(data_split, num_steps, undiscounted_reward,
                                         discounted_reward)

        if TUNE:
            tune.report(episode_reward=undiscounted_reward, episode=episode)

    average_reward: float = total_reward / NUM_EPISODES
    logging.debug('Average undiscounted reward: %i', average_reward)
    logging.debug("Simulation finished.")
    env.close()

    MLFLOW_LOGGER.log_episode_end(agents, average_reward)


def _set_seeds() -> None:
    logging.info('Setting seed: "%s"', SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)


def _set_data_split(env: PotopEnv, is_test_episode: bool, is_training_episode: bool) -> DataSplit:
    if is_training_episode:
        data_split = DataSplit.TRAINING
        env.set_data_split(DataSplit.TRAINING)
    elif is_test_episode:
        data_split = DataSplit.TEST
        env.set_data_split(DataSplit.TEST)
    else:
        data_split = DataSplit.VALIDATION
        env.set_data_split(DataSplit.VALIDATION)
    return data_split


def _run_episode(env: PotopEnv, shared_policy_agent, agents, episode: int, is_training: bool,
                 use_shared_policy: bool) -> Tuple[int, int, int]:
    MLFLOW_LOGGER.log_episode_number(episode)

    states: ndarray = env.reset()

    undiscounted_reward: int = 0
    discounted_reward: int = 0
    num_steps: int = 0
    current_day: int = env.current_time.day
    reward_per_day: list = [0]

    if use_shared_policy:
        actions = shared_policy_agent.act(states, [True] * NUM_AGENTS)
    else:
        actions = []
        old_actions = []
        for i, agent in enumerate(agents):
            actions.append(agent.act(states[i]))
            old_actions.append(actions[i])
            try:
                agent.memory.signal_starting_episode()
            except AttributeError:
                pass

    episode_ongoing: bool = True

    while episode_ongoing:
        logging.debug(env.current_time)
        next_states, rewards, dones, _ = env.step(actions)

        if use_shared_policy:
            if is_training:
                for i, reward in enumerate(rewards):
                    if reward is not None:
                        shared_policy_agent.save_transition(next_states[i],
                                                            actions[i],
                                                            rewards[i][DISCOUNTED_REWARD],
                                                            states[i],
                                                            dones[i])

            available_agents = [(rewards[i] is not None) for i in range(NUM_AGENTS)]
            actions = shared_policy_agent.act(next_states, available_agents)

            for i in range(NUM_AGENTS):
                if rewards[i] is None:
                    continue
                undiscounted_reward += rewards[i][UNDISCOUNTED_REWARD]
                discounted_reward += rewards[i][DISCOUNTED_REWARD]
                reward_per_day[-1] += rewards[i][UNDISCOUNTED_REWARD]

        else:
            # Save old actions, old states and resulting new states and rewards
            if is_training:
                for i, agent in enumerate(agents):
                    if rewards[i] is not None:
                        agent.save_transition(
                            next_states[i],
                            old_actions[i],
                            rewards[i][DISCOUNTED_REWARD],
                            states[i],
                            dones[i])

            # Generate new actions
            actions = []
            for i, agent in enumerate(agents):
                if rewards[i] is None:
                    actions.append(None)
                else:
                    actions.append(agent.act(next_states[i]))
                    old_actions[i] = actions[i]
                    undiscounted_reward += rewards[i][UNDISCOUNTED_REWARD]
                    discounted_reward += rewards[i][DISCOUNTED_REWARD]
                    reward_per_day[-1] += rewards[i][UNDISCOUNTED_REWARD]

        num_steps += 1
        for i, reward in enumerate(rewards):
            if reward is not None:
                states[i] = next_states[i]
        episode_ongoing = not dones[0]

        if env.current_time.day != current_day:
            MLFLOW_LOGGER.log_reward_per_day(reward_per_day)
            reward_per_day.append(0)
            current_day = env.current_time.day
    return num_steps, undiscounted_reward, discounted_reward


def _initialize_environment():
    logging.info('Initializing environment.')

    # This is needed on the slurm server
    if os.getcwd() == '/':
        os.chdir('/repo/potop/src')
    # This is needed if we run the simulation using ray tune
    if TUNE:
        # Use this for local tuning
        if not USE_LOCAL_SETTINGS:
            os.chdir("../../../repo/potop/src")

    if not os.path.exists(GRAPH_FILENAME):
        # Creates a new graph (takes approx. 15-30 minutes)
        create_graph(disctricts=DISTRICTS, filename=GRAPH_FILENAME)

    event_log = get_event_log()

    env = PotopEnv(
        event_log=event_log,
        graph_filename=GRAPH_FILENAME,
        num_agents=NUM_AGENTS,
        render=RENDER,
        data_split=DataSplit.TRAINING,
        observation_mode=OBSERVATION_MODE)

    return env


def get_event_log():
    """Returns the event log."""
    logging.info("Loading event log.")

    if os.path.exists(EVENT_LOG_PATH):
        # Event logs exists, no need to create it
        dataset = None
    else:
        # Load dataset in order to create new event log
        if DISTRICTS == ["Docklands"]:
            dataset = DocklandsData().get_data()
        elif DISTRICTS == ["Queensberry"]:
            dataset = QueensberryData().get_data()
        else:
            raise ValueError("Unknown district selected.")

    event_log = EventLog(dataset, EVENT_LOG_PATH, GRAPH_FILENAME).event_log

    if SMALL_EVENT_LOG:
        event_log = np.array([e for e in event_log if e[TIME_COLUMN].week < 2])

    return event_log


def setup_agent(agent_algorithm: str) -> Tuple[Agent, dict, bool]:
    """Sets up the agent and agent parameters based on the algorithm name."""
    model = None
    if agent_algorithm == "random_edge":
        agent = RandomAgent
        agent_params = {"strategy": Strategy.RANDOM_EDGE}
    elif agent_algorithm == "random_route":
        agent = RandomAgent
        agent_params = {"strategy": Strategy.RANDOM_ROUTE}
    elif agent_algorithm == "greedy":
        agent = Greedy
        agent_params = {"num_agents": NUM_AGENTS, "shared_policy": True}
        # the greedy agent needs the allowed for the heuristic
        assert USE_ALLOWED_PARKING_TIME
    elif agent_algorithm == "aco":
        agent = ACO
        agent_params = aco_params
    elif agent_algorithm == "ddqn":
        agent = DDQN
        if TUNE:
            agent_params = ddqn_params_tune
        else:
            agent_params = ddqn_params

        agent_params["num_agents"] = NUM_AGENTS
        shared_policy: bool = NUM_AGENTS > 1 and params["shared_policy"]
        model = get_model(agent_algorithm, NUM_AGENTS, shared_policy)
        # the spot assignemnt encoding is needed for shared policy ddqn
        if shared_policy:
            assert USE_SPOT_ASSIGNMENT_COLUMN
    elif agent_algorithm == "ppo":
        agent = PPO
        model = get_model(agent_algorithm, NUM_AGENTS, False)
        if TUNE:
            agent_params = ppo_params_tune
        else:
            agent_params = ppo_params
    elif agent_algorithm == "coma":
        agent = COMA
        agent_params = coma_params
        agent_params["shared_policy"] = True
        # COMA always needs the spot assignment
        assert USE_SPOT_ASSIGNMENT_COLUMN
    else:
        raise InvalidAgentAlgorithm

    return agent, agent_params, model


def setup_logging():
    """
    Sets up the logging for the whole project. The logs are saved to two separate files (info.log
    and debug.log) and also the console.
    """
    logging.root.handlers = []
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(LOG_LEVEL)
    info_handler = logging.FileHandler("info.log", mode='w')
    info_handler.setLevel(logging.INFO)
    debug_handler = logging.FileHandler("debug.log", mode='w')
    debug_handler.setLevel(logging.DEBUG)

    logging.basicConfig(
        level=LOG_LEVEL,
        handlers=[
            stream_handler,
            info_handler,
            debug_handler
        ]
    )


def setup_mlflow(experiment_name: str) -> object:
    """If mlflow is enabled, set it up and return the config object for ray-tune."""
    if MLFLOW_TRACKING:
        # Check if mlflow server can be reached
        mlflow_tracking_ip = MLFLOW_TRACKING_URI.replace("http://", "").split(":")[0]
        ping(mlflow_tracking_ip)  # Raises error if ping is unsuccessful

        # Set connection details
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        # Set experiment and run details
        mlflow.set_experiment(experiment_name)
        mlflow.set_tag("username", your_name)
        mlflow.log_param("Operating System", f"{os.name}-{platform.system()}-{platform.release()}")
        mlflow.log_param("Python Version", sys.version)

        # Log params from settings.py
        mlflow.log_params({
            "SEED": SEED,
            "NUM_AGENTS": NUM_AGENTS,
            "NUM_EPISODES": NUM_EPISODES,
            "GRAPH_FILENAME": GRAPH_FILENAME,
            "OBSERVATION_MODE": OBSERVATION_MODE,
            "CACHED_SHORTEST_PATHS": CACHED_SHORTEST_PATHS,
            "TUNE": TUNE,
            "RENDER": RENDER,
            "USE_ALLOWED_PARKING_TIME": USE_ALLOWED_PARKING_TIME,
            "USE_SPOT_ASSIGNMENT": USE_SPOT_ASSIGNMENT_COLUMN,
        })

        mlflow_config = {
            "experiment_name": experiment_name,
            "tracking_uri": mlflow.get_tracking_uri()
        }
    else:
        logging.info("MLFlow tracking is disabled.")
        mlflow_config = None
    return mlflow_config


def load_model(agents: List[Agent], pretrained_models: List[str]) -> None:
    """
    Loads the pre-trained models, if loading models is enabled and if there is a model for the agent
    """
    if LOAD_MODEL:
        if pretrained_models is None:
            logging.debug("Pre-trained model is not provided.")
        else:
            logging.debug("Loading pre-trained model.")

            # Check if all agents have a provided model
            assert len(agents) == len(pretrained_models)

            for agent, model in zip(agents, pretrained_models):
                filename = SAVE_MODEL_PATH + model
                try:
                    os.path.isfile(filename)
                except FileNotFoundError:
                    logging.warning("Model does not exist.")
                else:
                    agent.load_model(filename)
    else:
        logging.debug("Continuing without pre-trained model.")


if __name__ == '__main__':
    setup_logging()

    logging.debug(torch.cuda.is_available())

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('config', metavar='N', nargs=3, type=str)
    parser.add_argument('--num_agents', dest='num_agents', type=int, required=False,
                        help="Overwrites the NUM_AGENTS parameter in config.settings.")
    args = parser.parse_args()

    if args.num_agents is not None:
        NUM_AGENTS = args.num_agents  # noqa: F811
    try:
        parsed_agent_algorithm = sys.argv[1]
        parsed_experiment_name = sys.argv[2]
        your_name = sys.argv[3]
    except IndexError as index_error:
        raise InvalidCommandLineArgsException from index_error

    Agent, params, models = setup_agent(parsed_agent_algorithm)
    mlflow_configuration = setup_mlflow(parsed_experiment_name)

    configuration = {
        "agent": Agent,
        "params": params,
        "models": models,
        "mlflow": mlflow_configuration,
    }

    MLFLOW_LOGGER.log_agent_algorithm(parsed_agent_algorithm)

    if TUNE and parsed_agent_algorithm in ["ppo", "ddqn"]:

        # Local mode just for debugging
        ray.init(local_mode=TUNE_LOCAL)

        simulation = tune.run(
            simulate,
            name=TUNE_NAME,
            local_dir=TUNE_DIR,
            config=configuration,
            num_samples=NUM_SAMPLES,
            scheduler=SCHEDULER,
            resources_per_trial=RESOURCES
        )
        best_config = simulation.get_best_config(metric="average_reward", mode="max")
        MLFLOW_LOGGER.log_best_config(best_config)
    else:
        simulate(configuration)
