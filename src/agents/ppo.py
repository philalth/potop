"""
Module contains implementation of the Proximal Policy Optimization (PPO)
algorithm including the replay memory and neural network.
"""
import logging
from typing import List, Any

import numpy as np
import torch
import torch.nn as nn
from numpy import ndarray
from torch import FloatTensor, Tensor
from torch.distributions import Categorical
from torch.nn import MSELoss, Sequential
from torch.optim import Adam

from agents.agent import Agent
from config.settings import DEVICE


class PPO(Agent):
    """
    Agent implementing the Proximal Policy Optimization (PPO) algorithm.

    See also: https://arxiv.org/pdf/1707.06347.pdf
    """

    def __init__(self, action_space: Any, observation_space: Any, params: dict) -> None:
        """
        Initializes a new instance.

        :param action_space: The agents action space.
        :param observation_space: The agents observation space.
        """
        super().__init__(action_space, observation_space)
        self.params: dict = params
        self.memory: Memory = Memory()

        action_dim: int = action_space.n
        state_dim: int = observation_space.shape[0] * observation_space.shape[1]

        self.policy: ActorCritic = ActorCritic(action_dim,
                                               state_dim,
                                               params["n_hidden_actor"],
                                               params["n_hidden_critic"]).to(DEVICE)
        self.policy_old: ActorCritic = ActorCritic(action_dim,
                                                   state_dim,
                                                   params["n_hidden_actor"],
                                                   params["n_hidden_critic"]).to(DEVICE)

        self.actor_optimizer: Adam = Adam(self.policy.actor.parameters(),
                                          lr=params["lr_actor"],
                                          betas=params["betas"],
                                          weight_decay=params["weight_decay"])
        self.value_optimizer: Adam = Adam(self.policy.critic.parameters(),
                                          lr=params["lr_critic"],
                                          betas=params["betas"],
                                          weight_decay=params["weight_decay"])

        self.loss: MSELoss = MSELoss()

    def save_transition(self, next_state: Any, action: Any, reward: float, _, done) -> None:
        """Saves a transition to the agent's memory."""
        # Check if transitions are stored correctly.
        if self.params["reward_clipping"]:
            reward = np.clip(reward, 0.0, 1.0)
        self.memory.add_transition(next_state, action, reward, done)
        if len(self.memory) == self.params["horizon"]:
            self.update()

    def act(self, state: ndarray) -> int:
        """
        Returns an action based on the state of the environment.

        :param state: The current state of the environment.
        :return: The action to be executed.
        """
        state: FloatTensor = FloatTensor(state).to(DEVICE)
        action, action_logprob = self.policy_old.act(state)
        # Store action log probability now, other info will be stored in main.py
        self.memory.logprobs.append(FloatTensor([action_logprob]))

        return action.item()

    def update(self):
        """Updates the weights of the agent."""
        logging.info("Updating PPO agent.")

        discounted_rewards: FloatTensor = _standardize_rewards(self.memory.rewards)

        # Optimize policy for K epochs:
        for _ in range(self.params["epochs"]):
            # Iterate over mini-batches
            for i in range(int(self.params["horizon"] / self.params["batch_size"])):
                batch_slice = slice(i * self.params["batch_size"],
                                    (i + 1) * self.params["batch_size"])

                # old information
                old_states: FloatTensor = _convert_to_tensor(self.memory.states, batch_slice)
                old_actions: FloatTensor = _convert_to_tensor(self.memory.actions, batch_slice)
                old_logprobs: FloatTensor = _convert_to_tensor(self.memory.logprobs, batch_slice)

                # Evaluate old actions and values :
                logprobs, state_values, _ = self.policy.evaluate(old_states, old_actions)

                # find the ratio (pi_theta / pi_theta_old)
                ratio: Tensor = torch.exp(logprobs - old_logprobs.detach())
                # find surrogate loss
                advantage: Tensor = discounted_rewards[batch_slice] - state_values.detach()
                # ratio_adv = ratio * advantage
                clip: Tensor = torch.clamp(ratio, 1 - self.params["eps_clip"],
                                           1 + self.params["eps_clip"]) * advantage

                action_loss: Tensor = -torch.min(ratio * advantage, clip).mean()
                value_loss: MSELoss = self.loss(state_values, discounted_rewards[batch_slice])

                self._perform_gradient_step(action_loss, value_loss)

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        # Clear memory
        self.memory.clear()

    def save_model(self) -> str:
        """
        Saves the current model of the agent to 'ppo_model.pth'.

        :return: The file path where the model has been saved.
        """
        model_path = "ppo_model.pth"
        torch.save(self.policy.state_dict(), model_path)
        return model_path

    def load_model(self, filename: str) -> None:
        """
        Loads the specified (pretrained) model.

        :param filename: The file location of the model to be loaded.
        :return: None.
        """
        self.policy.load_state_dict(torch.load(filename))
        self.policy_old.load_state_dict(self.policy.state_dict())

    def _perform_gradient_step(self, action_loss, value_loss):
        # actor network
        self.actor_optimizer.zero_grad()
        action_loss.mean().backward()
        if self.params["gradient_clipping"]:
            for param in self.policy.actor.parameters():
                param.grad.data.clamp_(-1, 1)
        self.actor_optimizer.step()

        # value network
        self.value_optimizer.zero_grad()
        value_loss.mean().backward()
        if self.params["gradient_clipping"]:
            for param in self.policy.critic.parameters():
                param.grad.data.clamp_(-1, 1)
        self.value_optimizer.step()


class ActorCritic(nn.Module):
    """The ActorCritic model of the PPO Agent."""

    def __init__(self, action_space, observation_space, n_hidden_actor=32,
                 n_hidden_critic=32) -> None:
        super().__init__()
        self.device = torch.device("cpu")
        self.actor: Sequential = Sequential(
            nn.Linear(observation_space, n_hidden_actor),
            nn.Tanh(),
            nn.Linear(n_hidden_actor, n_hidden_actor),
            nn.Tanh(),
            nn.Linear(n_hidden_actor, action_space),
            nn.Softmax()
        )
        self.critic: Sequential = Sequential(
            nn.Linear(observation_space, n_hidden_critic),
            nn.Tanh(),
            nn.Linear(n_hidden_critic, n_hidden_critic),
            nn.Tanh(),
            nn.Linear(n_hidden_critic, 1),
        )

    def act(self, state: object) -> object:
        """
        Chooses an action based on the current state of the environment.

        :param state: the agents observation
        :return: an action
        """
        state: Tensor = torch.flatten(state)
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach()

    def forward(self, input_tensor):
        """
        Defines the computation performed at every call.

        :param input_tensor: None
        :return: None
        """
        # not implemented
        return self, input_tensor

    def evaluate(self, state, action):
        """Evaluates the old actions based on the critic."""
        state = torch.flatten(state, 1)
        action_probs = self.actor(state)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        state_values = self.critic(state)
        dist_entropy = dist.entropy()

        return action_logprobs, state_values, dist_entropy


class Memory:
    """Stores the agents transitions."""

    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def __len__(self):
        """Returns the length of memory (the number of saved transitions)."""
        return len(self.states)

    def add_transition(self, state, action, reward, done):
        """Adds a transitions to the agents memory."""
        self.states.append(torch.FloatTensor(state))
        self.actions.append(torch.FloatTensor([action]))
        self.rewards.append(torch.FloatTensor([reward]))
        self.is_terminals.append(done)

    def clear(self):
        """Deletes all saved transitions."""
        self.__init__()


def _convert_to_tensor(array: list, indices: slice) -> torch.FloatTensor:
    return torch.squeeze(torch.stack(array[indices]), 1).to(DEVICE).detach()


def _standardize_rewards(rewards: List[int]) -> torch.FloatTensor:
    rewards = torch.FloatTensor(rewards).to(DEVICE)
    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
    return rewards
