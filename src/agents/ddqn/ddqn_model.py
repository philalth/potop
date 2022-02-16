"""
Module defines the PyTorch model for the Double-Deep-Q-Network (DDQN).
"""

import torch
from torch import nn
from torch.nn.functional import relu


# pylint: disable=R0913


class DDQNModel(nn.Module):
    """This class defines the complete DDQN neural network.

    It closely follows the network architecture by Schmoll, Schubert (2020) and also uses a similar
    input encoding.

    The network takes two different inputs:
        - a distance matrix (dimensions: n_actions x n_resource), which stays the same for all
          evaluations on the same street network graph and can therefore be precomputed
        - a resource representation matrix (dimensions: n_resource * encoding_size), which contains
          the encoded states of all resources in the environment

    It instantiates three submodules:
        - The distance module
        - The resource module
        - The final layer module

    The distance and resource module are computed independently and their results are matrix
    multiplied. After that, the final layer module is applied to the result.

    For a visual depiction of the model's architecture, see Schmoll, Schubert (2020), Figure 2 (p.5)

    Literature:
        - Schmoll, Sebastian; Schubert, Matthias (2020): "Semi-Markov Reinforcement Learning for
            Stochastic Resource Collection", in: Proceedings of the Twenty-Ninth International Joint
            Conference on Artificial Intelligence (IJCAI-20)
    """

    def __init__(self,
                 n_hidden_distance_mlp=20,
                 n_in_resource_mlp=8,
                 n_hidden_resource_mlp=20,
                 n_out_resource_mlp=8,
                 n_hidden_final_mlp=20):
        super().__init__()
        self.distance_module = DDQNDistance(n_hidden=n_hidden_distance_mlp)
        self.resource_repr_module = DDQNResourceRepr(n_in=n_in_resource_mlp,
                                                     n_hidden=n_hidden_resource_mlp,
                                                     n_out=n_out_resource_mlp)
        self.final_mlp = DDQNFinalLayerMLP(n_in=n_out_resource_mlp, n_hidden=n_hidden_final_mlp)

    def forward(self, state_mat, dist_mat):
        """Forward function over the whole net."""
        resource_mat = self.resource_repr_module(state_mat.float())
        similarity_mat = self.distance_module(dist_mat.unsqueeze(-1)).squeeze(-1)
        similarity_mat = similarity_mat / similarity_mat.sum(1).unsqueeze(1)

        matrix = torch.matmul(similarity_mat, resource_mat)
        q_values = self.final_mlp(matrix).squeeze(-1)

        return q_values


class DDQNDistance(nn.Module):
    """
    Takes distance matrix between resources and edges of the action space
    perform fully connected sigmoid functions.
    """

    def __init__(self, n_hidden):
        super().__init__()
        self.fc1 = nn.Linear(1, n_hidden)
        self.fc2 = nn.Linear(n_hidden, 1)

        # Initialization of weights
        # nn.init.normal_(self.fc1.weight, mean=0.0, std=0.1)
        # self.fc1.bias.data.fill_(0.0)
        # nn.init.normal_(self.fc2.weight, mean=0.0, std=0.1)
        # self.fc2.bias.data.fill_(0.0)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, dist_mat):
        """Forward function."""
        matrix = torch.sigmoid(self.fc1(dist_mat)).to(self.device)
        matrix = torch.sigmoid(self.fc2(matrix)).to(self.device)

        return matrix


class DDQNResourceRepr(nn.Module):
    """
    Takes distance matrix: resources x attributes
    perform fully connected relu.
    """

    def __init__(self, n_in, n_hidden, n_out):
        super().__init__()
        self.n_out = n_out
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_out)

        # Initialization of weights
        # nn.init.normal_(self.fc1.weight, mean=0.0, std=0.1)
        # self.fc1.bias.data.fill_(0.0)
        # nn.init.normal_(self.fc2.weight, mean=0.0, std=0.1)
        # self.fc2.bias.data.fill_(0.0)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, resource_mat):
        """Forward function."""
        matrix = relu(self.fc1(resource_mat)).to(self.device)
        matrix = relu(self.fc2(matrix)).to(self.device)
        matrix = self.fc3(matrix).to(self.device)

        return matrix


class DDQNFinalLayerMLP(nn.Module):
    """
    Takes matrix multiplication of the first step
    perform fully connected relu.
    """

    def __init__(self, n_in, n_hidden):
        super().__init__()
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.fc2 = nn.Linear(n_hidden, 1)

        # Initialization of weights
        # nn.init.normal_(self.fc1.weight, mean=0.0, std=0.1)
        # self.fc1.bias.data.fill_(0.0)
        # nn.init.normal_(self.fc2.weight, mean=0.0, std=0.1)
        # self.fc2.bias.data.fill_(0.0)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, in_matrix):
        """Forward function."""
        matrix = relu(self.fc1(in_matrix)).to(self.device)
        matrix = self.fc2(matrix).to(self.device)

        return matrix


class SimpleDDQNModel(nn.Module):
    """
    Simple DDQN Model without the distance matrix. It takes the unprocessed state as the input
    and returns an action as output.
    """

    def __init__(self, state_dim, action_dim, n_hidden=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, action_dim)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, state):
        """Forward function over the whole net."""
        state = torch.flatten(state, 1).to(self.device)
        out = relu(self.fc1(state)).to(self.device)
        out = relu(self.fc2(out)).to(self.device)
        out = self.fc3(out).to(self.device)
        return out
