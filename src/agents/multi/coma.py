""""
Module contains implementation of Counterfactual Multi-Agent Policy Gradients
Original paper: https://arxiv.org/pdf/1705.08926.pdf
"""

from typing import List
from torch.distributions.categorical import Categorical
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from agents.agent import Agent
from config.settings import GAMMA


# pylint: disable=R0902
class COMA(Agent):
    """
    Agent implementing the Counterfactual Multi-Agent Policy Gradients
    (COMA) Algorithm
    """

    def __init__(self, action_space, observation_space, params):
        super().__init__(action_space, observation_space)

        # only for debugging
        torch.autograd.set_detect_anomaly(True)

        action_space: int = action_space.n
        observation_space: int = observation_space.shape[0] * observation_space.shape[1]

        self.observation_space = observation_space

        # Parameters
        self.gamma = GAMMA
        self.batch_size = params["batch_size"]
        self.num_agents = 2
        self.params = params

        self.actors = []
        self.a_optimizers = []

        for i in range(self.num_agents):
            self.actors.append(Actor(action_space, observation_space))
            self.a_optimizers.append(
                torch.optim.Adam(self.actors[i].parameters(), lr=params["lr_actor"])
            )

        self.critic = Critic(action_space, observation_space, self.num_agents)
        self.c_optimizer = torch.optim.Adam(self.critic.parameters(), lr=params["lr_critic"])

        self.c_loss_fn = torch.nn.MSELoss()

        self.memory = Memory()

    # pylint: disable=W0221
    # parameters differ from overriden act method
    def act(self, state, available_agents: List[bool]) -> List[int]:
        """Returns an action based on the state of the environment."""

        actions = []
        pis = []
        for i, available in enumerate(available_agents):
            if available:
                dist = self.actors[i](torch.FloatTensor(state[i]))
                action = Categorical(dist).sample().item()
            else:
                # Fake dist
                dist = torch.rand(188)
                action = Categorical(dist).sample().item()
                # dist, action = None, None
            pis.append(dist)
            actions.append(action)

        self.memory.pis.append(pis)

        return actions

    def save_transition(self, next_state, action, reward, _, done):
        """Saves a transition to the agent's memory."""
        action = np.array(action)
        reward = np.array(reward)

        for i, act in enumerate(action):
            if act is None:
                # Fake action
                action[i] = self.action_space.n-1
                # Fake reward
                reward[i] = 0.0

        self.memory.add_transition(next_state, action, reward, done)

        if len(self.memory) == self.batch_size:
            self.update()

    # pylint: disable=R0914
    # too many local variables
    def update(self):
        """Updates the actors and critic based on previous transitions."""
        a_optimizers = self.a_optimizers

        states = np.array(self.memory.states)
        rewards = np.array(self.memory.rewards)
        dones = self.memory.is_terminals

        actions = np.array(self.memory.actions, dtype=float)
        actions = torch.FloatTensor(actions)

        pis = []
        for i in range(self.num_agents):
            pis.append(torch.cat(self.memory.pis[i]).view(
                len(self.memory.pis[i]), self.action_space.n))

        for i in range(self.num_agents):

            # train actor
            input_critic = self.build_input_critic(i, states, actions)
            q_target = self.critic(input_critic).detach()

            action_taken = actions.type(torch.long)[:, i].reshape(-1, 1)

            baseline = torch.sum(pis[i] * q_target, dim=1).detach()
            q_taken_target = torch.gather(q_target, dim=1, index=action_taken).squeeze()
            advantage = q_taken_target - baseline

            log_pi = torch.log(torch.gather(pis[i], dim=1, index=action_taken).squeeze())

            actor_loss = - torch.mean(advantage * log_pi)

            a_optimizers[i].zero_grad()
            actor_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), 5)
            a_optimizers[i].step()

            self.train_critic(i, actions, rewards, dones, q_taken_target, input_critic)

        self.memory.clear()

    # pylint: disable=R0913
    # too many arguments
    def train_critic(self, i, actions, rewards, dones, q_taken_target, input_critic):
        """Updates the network of the critic."""
        # train critic
        q_critic = self.critic(input_critic)

        action_taken = actions.type(torch.long)[:, i].reshape(-1, 1)
        q_taken = torch.gather(q_critic, dim=1, index=action_taken).squeeze()

        # TD(0)
        discounted_reward = torch.zeros(len(rewards[:, i]))
        for trans in range(len(rewards[:, i])-1):
            if dones[i][trans]:
                discounted_reward[trans] = rewards[:, i][trans]
            else:
                discounted_reward[trans] = rewards[:, i][trans] + \
                    self.gamma * q_taken_target[trans + 1]

        critic_loss = torch.mean((discounted_reward - q_taken) ** 2)

        self.c_optimizer.zero_grad()
        critic_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 5)
        self.c_optimizer.step()

    def build_input_critic(self, agent_id, observations, actions):
        """
        Concatenates the states, agent id and actions. This is used as the input for the critic.
        """

        ids = (torch.ones(self.batch_size) * agent_id).view(-1, 1)

        observations = torch.FloatTensor(observations).view(
            self.batch_size, self.observation_space * 2)
        input_critic = torch.cat([observations.type(torch.float32),
                                  actions.type(torch.float32)], dim=-1)
        input_critic = torch.cat([ids, input_critic], dim=-1)

        return input_critic


class Actor(nn.Module):
    """The actor model of a single COMA agent."""

    def __init__(self, action_space, observation_space):
        super().__init__()
        self.fc1 = nn.Linear(observation_space, 32)
        self.fc2 = nn.Linear(32, action_space)

    def forward(self, state):
        """Returns an action based on the state of the environment."""
        state = torch.flatten(state)
        out = F.relu(self.fc1(state))
        out = F.softmax(self.fc2(out), dim=0)
        # out = Categorical(out.squeeze(0))
        return out


class Critic(nn.Module):
    """The critic model used to evaluate states."""

    def __init__(self, action_space, observation_space, num_agents):
        super().__init__()

        input_dim = 1 + observation_space * num_agents + num_agents

        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.out = nn.Linear(32, action_space)

    def forward(self, state):
        """Returns a value based on the state observed by an agent."""
        # state_one = torch.flatten(state_one).unsqueeze(0)
        out = F.relu(self.fc1(state))
        out = F.relu(self.fc2(out))
        out = self.out(out)
        return out


class Memory:
    """Stores the agents transitions."""

    def __init__(self):
        self.actions = []
        self.states = []
        self.rewards = []
        self.pis = []
        self.is_terminals = []

    def __len__(self):
        """Returns the length of memory (the number of saved transitions)."""
        return len(self.rewards)

    def add_transition(self, states, actions, reward, done):
        """Adds a transitions to the agents memory."""
        self.states.append(states)
        self.actions.append(actions)
        self.rewards.append(reward)
        self.is_terminals.append(done)

    def clear(self):
        """Deletes all saved transitions."""
        self.__init__()
