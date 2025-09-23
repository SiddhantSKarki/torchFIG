import torch
from .registry import Registry
from . import *


# global registries
torch.models = Registry("Models")
torch.losses = Registry("Losses")
torch.optimizers = Registry("Optimizers")
torch.datasets = Registry("Datasets")