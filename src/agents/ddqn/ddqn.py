"""
Module contains functionality for a double DQN agent.
"""
import copy
import logging
import pickle
from typing import List

import mlflow
import numpy as np
import torch

from agents.agent import Agent
from agents.ddqn.ddqn_model import DDQNModel, SimpleDDQNModel
from agents.ddqn.prioritized_replay_memory import ReplayMemory
from config.settings import MLFLOW_TRACKING, DEVICE
from envs.utils import get_distance_matrix

DISTANCE_MATRIX = "../data/distance_matrix.pickle"


class DDQN(Agent):
    """The class for creating ddqn agents."""

    # pylint: disable=(R0902, R0914, R0915)
    def __init__(self, action_space, observation_space, params):
        super().__init__(action_space, observation_space)
        self.action_space = action_space
        self.observation_space = observation_space

        self.resource_input_size = observation_space.shape[1]

        # Learning hyperparameters
        learning_rate = params["learning_rate"]
        self.minibatch_size = params["minibatch_size"]

        # Epsilon decay for exploration
        self.epsilon = params["epsilon"]["start"]
        self.epsilon_min = params["epsilon"]["min"]
        self.epsilon_decay = params["epsilon"]["decay"]

        self.target_update_interval = params["target_update_interval"]
        self.agent_number = params["agent_number"] if "agent_number" in params else 0

        # Model hyperparameters
        n_hidden_distance_mlp = params["n_hidden_distance_mlp"]
        n_in_resource_mlp = self.resource_input_size
        n_hidden_resource_mlp = params["n_hidden_resource_mlp"]
        n_out_resource_mlp = params["n_out_resource_mlp"]
        n_hidden_final_mlp = params["n_hidden_final_mlp"]

        self.warmup_phase = params["warmup_phase"]
        self.update_step = params["update_step"]
        self.memory = ReplayMemory(size=params["memory_size"])
        self.training_count = 0

        logging.info("CPU/GPU: torch.device is set to %s.", str(DEVICE))
        if MLFLOW_TRACKING:
            mlflow.log_param("torch.device", DEVICE)

        self.simple_model = params["simple_model"]
        self.gradient_clipping = params["gradient_clipping"]
        self.reward_clipping = params["reward_clipping"]

        if self.simple_model:
            action_dim: int = action_space.n
            state_dim: int = observation_space.shape[0] * observation_space.shape[1]
            self.policy_model = SimpleDDQNModel(state_dim, action_dim).to(DEVICE)
            self.target_model = SimpleDDQNModel(state_dim, action_dim).to(DEVICE)
        else:
            self.n_out = params["num_agents"] if params["shared_policy"] else 1
            self.policy_model = DDQNModel(n_hidden_distance_mlp,
                                          n_in_resource_mlp,
                                          n_hidden_resource_mlp,
                                          n_out_resource_mlp,
                                          n_hidden_final_mlp).to(DEVICE)

            self.target_model = DDQNModel(n_hidden_distance_mlp,
                                          n_in_resource_mlp,
                                          n_hidden_resource_mlp,
                                          n_out_resource_mlp,
                                          n_hidden_final_mlp).to(DEVICE)

        if params["loss"] == "mse":
            self.loss = torch.nn.MSELoss()
        elif params["loss"] == "l1":
            self.loss = torch.nn.SmoothL1Loss()

        self.mixer = None
        if params["use_qmix"]:
            self.mixer = params["qmixer"]
            self.target_mixer = copy.deepcopy(self.mixer)

        parameters = self.policy_model.parameters()
        if self.mixer is not None:
            parameters = list(parameters) + list(self.mixer.parameters())

        if params["optimizer"] == "adam":
            self.optimizer = torch.optim.Adam(parameters, learning_rate)
        elif params["optimizer"] == "rmsprop":
            self.optimizer = torch.optim.RMSprop(parameters, learning_rate)

        graph = params["graph"]

        # Try loading the pickled distance matrix.
        # If it doesn't exist, create it and cache it for future use
        # WARNING: Takes ~10 minutes to create from scratch
        try:
            with open(DISTANCE_MATRIX, "rb") as file:
                self.distance_matrix = torch.from_numpy(pickle.load(file)).float().to(DEVICE)
        except FileNotFoundError:
            logging.warning(
                "Creating distance matrix as no cached version was found. "
                "This could take up to 10 minutes...")
            self.distance_matrix = torch.from_numpy(get_distance_matrix(graph)) \
                .float().to(DEVICE)
            with open(DISTANCE_MATRIX, "wb") as outfile:
                pickle.dump(self.distance_matrix, outfile)
            logging.info(
                "Distance matrix successfully created and cached for future use as %s.",
                DISTANCE_MATRIX)

    # pylint: disable=(W0221, W0102)
    def act(self, state, available_agents: List[bool] = [True]):
        """Returns an action based on the state of the environment."""
        rand = np.random.random()
        if rand < self.epsilon and not self.test:
            # random action
            action = [self.action_space.sample() if i else None for i in available_agents]
        else:
            action = []
            for i, available in enumerate(available_agents):
                if available:
                    # best action
                    # shared_policy setting
                    if self.n_out > 1:
                        resource_matrix = torch.from_numpy(
                            state[i][:, :self.resource_input_size]).float().to(DEVICE)
                    else:
                        resource_matrix = torch.from_numpy(
                            state[:, :self.resource_input_size]).float().to(DEVICE)

                    distance_matrix = self.distance_matrix

                    if self.simple_model:
                        resource_matrix = resource_matrix.unsqueeze(0)
                        q_values = self.policy_model(resource_matrix)
                        q_values = q_values.squeeze(0)
                    else:
                        distance_matrix = self.distance_matrix
                        q_values = self.policy_model(resource_matrix, distance_matrix)

                    assert self.action_space.n == len(q_values)
                    action.append(q_values.argmax().item())
                else:
                    action.append(None)

        self._epsilon_decay()

        if self.n_out == 1:
            action = action[0]

        return action

    def _epsilon_decay(self):
        if (self.epsilon - self.epsilon_decay) >= self.epsilon_min:
            self.epsilon -= self.epsilon_decay
        logging.debug("Epsilon %s", self.epsilon)

    # pylint: disable=R0902, E1102, R0914
    # Why is torch.tensor not callable?
    def update(self):
        """Updates the weights to optimize a ddqn loss function."""
        if self.warmup_phase > 0:
            self.warmup_phase -= 1
            return
        if self.training_count == 0:
            logging.info("Warmup phase ended!")
        self.training_count += 1

        # Update only each update_step steps
        if self.training_count % self.update_step != 0:
            return

        minibatch = self.memory.sample_batch(self.minibatch_size)

        states, actions, rewards, next_states, non_final_mask = get_tensors(
            minibatch)

        if self.reward_clipping:
            rewards = np.clip(rewards, 0.0, 1.0)

        if self.simple_model:
            on_policy_q_values = self.policy_model(states.to(DEVICE)) \
                .gather(1, actions.unsqueeze(1).to(DEVICE)).squeeze()
            next_actions = self.policy_model(next_states.to(DEVICE)).max(dim=-1)[1]
        else:
            on_policy_q_values = self.policy_model(states.to(DEVICE), self.distance_matrix) \
                .gather(1, actions.unsqueeze(1).to(DEVICE)).squeeze()
            # Get indices of the action that is proposed by the policy network for the NEXT states
            next_actions = \
                self.policy_model(next_states.to(DEVICE), self.distance_matrix).max(dim=-1)[1]

        # Calculate the q-values for those proposed actions via the target network
        off_policy_q_values = torch.zeros(self.minibatch_size, device=DEVICE)

        if self.simple_model:
            unmasked_q = self.target_model(next_states.to(DEVICE)) \
                .gather(1, next_actions.unsqueeze(1)).squeeze().detach()
        else:
            unmasked_q = self.target_model(next_states.to(DEVICE), self.distance_matrix) \
                .gather(1, next_actions.unsqueeze(1)).squeeze().detach()

        off_policy_q_values[non_final_mask] = unmasked_q

        expected_off_policy_action_q_values = rewards.to(DEVICE) + off_policy_q_values

        if self.mixer is not None:
            indices = torch.tensor([0]).to(DEVICE)  # no clue if this is correct
            on_policy_q_values = self.mixer(on_policy_q_values,
                                            torch.index_select(next_states, 0, indices).to(
                                                DEVICE))
            expected_off_policy_action_q_values = self.target_mixer(
                expected_off_policy_action_q_values,
                torch.index_select(next_states, 0, indices).to(DEVICE))

        self.perform_gradient_step(on_policy_q_values, expected_off_policy_action_q_values)

        # Every few steps, update target network
        if self.training_count % self.target_update_interval == 0:
            logging.debug("frozen weights are updated")
            self.target_model.load_state_dict(self.policy_model.state_dict())
            self.target_model.eval()
            if self.mixer is not None:
                self.target_mixer.load_state_dict(self.mixer.state_dict())

    def perform_gradient_step(self, on_policy_q_values, expected_off_policy_action_q_values):
        """Calculates the loss and performs a gradient step."""
        loss = self.loss(on_policy_q_values, expected_off_policy_action_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        if self.gradient_clipping:
            for param in self.policy_model.parameters():
                param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    # pylint: disable=R0902, R0913
    def save_transition(self, next_state, action, reward, old_state, done):
        """Saves a transition to the agent's memory.

        Columns of the state vector:
        0.-3. one-hot encoded:
            - free
            - occupied,
            - violation
            - fined
        4. walking time to resource
        5. current time
        6. time of arrival
        7. -1 to +2 indicator of violation time
        [8. allowed parking time]
        """

        old_state = torch.FloatTensor(
            old_state[:, :self.resource_input_size])  # (531, x), x={8, 9}
        next_state = torch.FloatTensor(
            next_state[:, :self.resource_input_size])  # (531, x), x={8, 9}
        action = torch.tensor([action])  # 1 (int)
        reward = torch.FloatTensor([reward])  # 1 (float)
        # done doesn't have to be converted

        self.memory.save((old_state, action, reward, next_state, done))

        self.update()

    def save_model(self) -> str:
        """Saves the current model of the agent."""
        # torch.save(self.policy_model.state_dict(),
        #           "../data/ddqn/ddqn_policy_model" + self.agent_number + ".pth")
        # model_path: str = SAVE_MODEL_PATH + "ddqn_target_model" + str(self.agent_number) + ".pth"
        model_path: str = "ddqn_target_model" + str(self.agent_number) + ".pth"
        torch.save(self.target_model.state_dict(), model_path)
        return model_path

    def load_model(self, filename: str) -> None:
        """Loads the pretrained model."""
        self.policy_model.load_state_dict(torch.load(filename))
        self.target_model.load_state_dict(torch.load(filename))


def get_tensors(minibatch):
    """Returns the tensors based on the sampled minibatch of transitions."""

    states, actions, rewards, next_states, dones = tuple(zip(*minibatch))
    non_final_mask = torch.tensor([not done for done in dones], dtype=torch.bool,
                                  device=DEVICE).detach()

    # values are already tensors, we just have to stack them
    states = torch.stack(states)  # (64, 531, x), x={8, 9}
    rewards = torch.stack(rewards).squeeze()  # 64
    actions = torch.stack(actions).squeeze()  # 64
    next_states = [s for s, done in zip(next_states, dones) if not done]
    next_states = torch.stack(next_states)  # (64-x, 531, x), x={8, 9}

    return states, actions, rewards, next_states, non_final_mask
