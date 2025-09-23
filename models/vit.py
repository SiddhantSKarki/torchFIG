import torch
import torch.nn as nn

@torch.models.register("vit")
class VisionTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc = nn.Linear(cfg["hidden_dim"], cfg["num_classes"])

    def forward(self, x):
        return self.fc(x)
