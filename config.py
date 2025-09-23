import os

def get_env(key, default, cast=str):
    val = os.getenv(key)
    return cast(val) if val is not None else default

CONFIG_DEFAULT  = {
    "experiment": {
        "name": get_env("EXP_NAME", "debug"),
        "seed": get_env("SEED", 42, int),
        "device": get_env("DEVICE", "mps"),
        "model_dir": get_env("OUTPUT_DIR", "./model_dir"),
    },

    "logging": {
        "use_tensorboard": get_env("USE_TENSORBOARD", "true").lower() == "true",
        "log_dir": get_env("LOG_DIR", "./runs"),
        "visualization": False
    },

    "data": {
        "dataset": get_env("DATASET", "SuperPixelGraph"),
        "dataset_params": {
            "root": get_env("DATA_DIR", "./data"),
            "max_nodes": 75,
            "num_features_per_node": 2
        },
        "train_batch_size": get_env("BATCH_SIZE", 32, int),
        "val_batch_size": get_env("BATCH_SIZE", 32, int),
        "test_batch_size": get_env("BATCH_SIZE", 32, int),

        "num_workers": get_env("NUM_WORKERS", 4, int),
        "train_val_ratio": get_env("TRAIN_VAL_RATIO", 0.8, float),
        "random_state": get_env("RANDOM_STATE", "true").lower() == "true"
    },

    "model": {
        "arch": get_env("MODEL", "GraphAutoencoder_FCN")
    },

    "train": {
        "epochs": get_env("EPOCHS", 50, int),
        "optimizer": get_env("OPTIMIZER", "adamw"),
        "optimizer_params": {
            "lr": get_env("LR", 1e-3, float),
            "weight_decay": get_env("WEIGHT_DECAY", 1e-4, float),
        },
        # add loss_[name] style for multiple losses
        "loss": get_env("LOSS", "cross_entropy"),
        "loss_params": None
    },
}

# Example Usage
CONFIG_1  = {
    "experiment": {
        "name": get_env("EXP_NAME", "debug"),
        "seed": get_env("SEED", 42, int),
        "device": get_env("DEVICE", "mps"),
        "model_dir": get_env("OUTPUT_DIR", "./model_dir"),
    },

    "logging": {
        "use_tensorboard": get_env("USE_TENSORBOARD", "true").lower() == "true",
        "log_dir": get_env("LOG_DIR", "./runs"),
        "visualization": True
    },

    "data": {
        "dataset": get_env("DATASET", "SuperPixelGraphDefault"),
        "dataset_params": {
            "root": get_env("DATA_DIR", "./data"),
            "max_nodes": 75,
            "num_features_per_node": 2
        },
        "train_batch_size": get_env("BATCH_SIZE", 32, int),
        "val_batch_size": get_env("BATCH_SIZE", 32, int),
        "test_batch_size": get_env("BATCH_SIZE", 32, int),

        "num_workers": get_env("NUM_WORKERS", 4, int),
        "train_val_ratio": get_env("TRAIN_VAL_RATIO", 0.8, float),
        "random_state": get_env("RANDOM_STATE", "true").lower() == "true"
    },

    "model": {
        "arch": get_env("MODEL", "GVAE_DEFAULT"),
        "model_params": {
            "in_channels": 2,
            "latent_dim": 8,
            "hidden_channels": 16
        },
        "load_pretrained": get_env("RANDOM_STATE", "false").lower() == "true",
        "model_weight_dir": get_env("MODEL_WEIGHT_DIR", "")
    },

    "train": {
        "epochs": get_env("EPOCHS", 50, int),
        "optimizer": get_env("OPTIMIZER", "adamw"),
        "optimizer_params": {
            "lr": get_env("LR", 1e-3, float),
            "weight_decay": get_env("WEIGHT_DECAY", 1e-4, float),
        },
        # add loss_[name] style for multiple losses
        "loss": get_env("LOSS", "gvae_loss"),
        "loss_params": None
    },
}

CONFIG_DEFAULT = CONFIG_1
