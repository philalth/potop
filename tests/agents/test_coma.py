from typing import List

import pytest
import torch
from gym import spaces
from agents.multi.coma import COMA


@pytest.fixture()
def coma():
    action_space = spaces.Discrete(188)
    observation_space = spaces.Box(0, 1, shape=(531, 8))
    params = {"lr_actor": 0.001, "lr_critic": 0.001, "batch_size": 1}
    yield COMA(action_space, observation_space, params)


def test_act(coma):
    observation_space = 531*8
    states = 2 * [torch.rand((1, observation_space))]
    actions = coma.act(states, [True])
    assert type(actions) == list
