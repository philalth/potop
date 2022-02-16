"""Module contains agent hyperparameters which will be tuned using ray-tune."""

from ray import tune

from agents.mixers.qmix import MixingNetwork

ddqn_params = {
    "optimizer": "rmsprop",
    "learning_rate": 0.0001,
    "minibatch_size": 64,
    "training_count": 0,
    "target_update_interval": 256,
    "n_hidden_distance_mlp": 256,
    "n_hidden_resource_mlp": 256,
    "n_out_resource_mlp": 256,
    "n_hidden_final_mlp": 256,
    "warmup_phase": 0,
    "update_step": 32,  # update policy net each x steps
    "memory_size": 512,
    "simple_model": False,  # True means that the distance matrix isn't used
    "gradient_clipping": False,
    "reward_clipping": True,  # clip rewards between 0.0 and 1.0
    "loss": "mse",
    "epsilon": {
        "start": 1.0,
        "min": 0.1,
        "decay": 0.001 / 200  # decay per step (200 steps = 1 day)
    },
    "shared_policy": False,
    "use_qmix": False,
    "qmixer": MixingNetwork({
        "state_shape": (531, 8),  # probably not the best way to hardcode this, fix later
        "mixing_embed_dim": 64,
        "hyper_hidden_dim": 32
    })
}

ddqn_params_tune = {
    "optimizer": "rmsprop",
    "learning_rate": tune.choice([0.00001, 0.00005, 0.0001, 0.0005, 0.001]),
    "minibatch_size": tune.choice([32, 64, 128]),
    "target_update_interval": 50000,
    "n_hidden_distance_mlp": 256,
    "n_hidden_resource_mlp": 256,
    "n_out_resource_mlp": 256,
    "n_hidden_final_mlp": 256,
    "warmup_phase": 5000,
    "update_step": tune.choice([8, 16, 32, 64]),
    "memory_size": 100000,
    "simple_model": False,
    "gradient_clipping": True,
    "reward_clipping": True,
    "loss": "mse",
    "epsilon": {
        "start": 1.0,
        "min": 0.1,
        "decay": 0.0005 / 200  # decay per step (200 steps = 1 day)
    },
    "shared_policy": False,
    "use_qmix": False,
    "use_ninth_column_allowed_parking_time": False
}

ppo_params = {
    "horizon": 1024,
    "batch_size": 256,
    "epochs": 3,
    "eps_clip": 0.2,
    "lr_actor": 0.0015042827919245516,
    "lr_critic": 0.0004631739140423425,
    "betas": (0.9, 0.999),
    "weight_decay": 0,
    "n_hidden_actor": 256,
    "n_hidden_critic": 256,
    "gradient_clipping": False,
    "reward_clipping": False
}

ppo_params_tune = {
    "horizon": tune.choice([512, 1024, 2048]),
    "batch_size": tune.choice([128, 256, 512]),
    "epochs": tune.choice([1, 3, 5]),
    "eps_clip": tune.choice([0.1, 0.2, 0.3]),
    "lr_actor": tune.uniform(0.0005, 0.005),
    "lr_critic": tune.uniform(0.0001, 0.001),
    "n_hidden_actor": tune.choice([64, 128, 256]),
    "n_hidden_critic": tune.choice([64, 128, 256]),
    "betas": (0.9, 0.999),
    "weight_decay": 0,
    "gradient_clipping": tune.choice([True, False]),
    "reward_clipping": tune.choice([True, False])
}

coma_params = {
    "batch_size": 2,
    "lr_actor": 0.0001,
    "lr_critic": 0.0001,
}

aco_params = {
    "ants": 1000,
    "computation_time": 1,
    "speed": 7,
    "evaporation_rate": 0.1,
    "alpha": 1.,
    "beta": 2,
    "prob_alpha": 1800.
}
