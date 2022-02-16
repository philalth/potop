"""
Module contains the mixing network for the QMIX architecture.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config.settings import DEVICE


class MixingNetwork(nn.Module):
    """
    The mixing network for the QMIX architecture.

    See also: https://arxiv.org/pdf/1803.11485v2.pdf
    """

    def __init__(self, params):
        super().__init__()
        self.state_dim = int(np.prod(params["state_shape"]))

        embed_dim = params["mixing_embed_dim"]
        hyper_hidden_dim = params["hyper_hidden_dim"]

        # first hypernet weights
        self.hyper_w1 = nn.Sequential(nn.Linear(self.state_dim, hyper_hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(hyper_hidden_dim, embed_dim)).to(DEVICE)

        # second hypernet weights
        self.hyper_w2 = nn.Sequential(nn.Linear(self.state_dim, hyper_hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(hyper_hidden_dim, embed_dim)).to(DEVICE)

        # first hypernet biases
        self.hyper_b1 = nn.Linear(self.state_dim, embed_dim).to(DEVICE)

        # second hypernet biases
        self.hyper_b2 = nn.Sequential(nn.Linear(self.state_dim, embed_dim),
                                      nn.ReLU(),
                                      nn.Linear(embed_dim, 1)).to(DEVICE)

    def forward(self, q_values, state):
        """
        Defines the computation performed at every call.

        :param q_values: The q-values of the agents.
        :param states: The state from the agent.
        :return: The q_total value.
        """
        q_values = q_values.unsqueeze(0).unsqueeze(2).to(DEVICE)  # [1, q_values=188, 1]
        states = state.reshape(-1, self.state_dim)  # [1, state_dim=4248]

        # absolute activation function for the first hypernet
        first_hypernet_weights = torch.abs(self.hyper_w1(states)).unsqueeze(0)
        # [1, 1, embed_dim=64]
        first_hypernet_biases = self.hyper_b1(states).unsqueeze(0)  # [1, 1, embed_dim=64]

        hidden = F.elu(torch.bmm(q_values, first_hypernet_weights) + first_hypernet_biases)
        # [1, q_values=188, embed_dim=64]

        # absolute activation function for the second hypernet
        second_hypernet_weights = torch.abs(self.hyper_w2(states))
        second_hypernet_biases = self.hyper_b2(states)

        # transform into correct shape
        second_hypernet_weights = second_hypernet_weights.unsqueeze(2)  # [1, embed_dim=64, 1]
        second_hypernet_biases = second_hypernet_biases.view(-1, 1, 1)

        # final q_tot calculation
        q_total = torch.bmm(hidden, second_hypernet_weights) + second_hypernet_biases

        q_total = q_total.view(-1, 1, 1)  # [q_values=188, 1, 1] could also be squeezed

        return q_total
