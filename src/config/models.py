"""Module contains list of pre-traied models."""

from typing import List


DDQN_MODELS = {
    "1": ["ddqn/single.pth"],  # network size 256 (all layers)
    "2": {
        "shared": [],
        "independent": []
    },
    "3": {
        "shared": [],
        "independent": []
    },
    "4": {
        "shared": [],
        "independent": []
    }
}
PPO_MODELS = {
    "1": ["ppo/single.pth"],  # network size actor 256, critic 256
}


def get_model(agent_algo: str, num_agents: int, shared: bool) -> List[str]:
    """Returns the filepaths of the pre-trained models based on the current setting."""

    if agent_algo == "ddqn":
        model_dict = DDQN_MODELS
    elif agent_algo == "ppo":
        model_dict = PPO_MODELS
    else:
        return None

    model = model_dict[str(num_agents)]

    if num_agents > 1:
        if shared:
            model = model["shared"]
        else:
            model = model["independent"]
    return model
