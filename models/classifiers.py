import torch
import torch.nn as nn

class BinaryClassifierHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),   # binary output
            nn.Sigmoid()               # probability that node exists
        )

    def forward(self, x):
        return self.fc(x)