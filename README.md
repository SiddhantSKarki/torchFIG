# TorchFIG

This repository provides a modular framework for experimenting with **graph-based models** (e.g., Graph Autoencoders, Graph Variational Autoencoders, Vision Transformers on graph-structured data).  
It is designed to be **configuration-driven**, **registry-based**, and easily extendable for datasets, models, losses, and optimizers.

---

## Project Structure




---

## ⚙️ Configuration System

All experiments are controlled via `config.py`.  
The config is structured as nested dictionaries, with support for environment variable overrides using `get_env`.

### Example Config

```python
CONFIG_DEFAULT = {
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
        "loss": get_env("LOSS", "cross_entropy"),
        "loss_params": None
    },
}
