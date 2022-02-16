import pytest

from agents.random import RandomAgent


@pytest.fixture()
def random_agent():
    yield RandomAgent(action_space=None)


def test_act():
    assert True
