import torch

@torch.optimizers.register("adamw")
def AdamW(params, cfg):
    return torch.optim.AdamW(params, **cfg["train"]["optimizer_params"])

@torch.optimizers.register("sgd")
def SGD(params, cfg):
    return torch.optim.SGD(params, **cfg["train"]["optimizer_params"])