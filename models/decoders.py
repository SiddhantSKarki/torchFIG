import torch
import torch.nn as nn

@torch.models.register("DecoderFCN")
class DecoderBlockFCN(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, z):
        return self.decoder(z)

class GVAEDecoder(torch.nn.Module):
    def forward(self, z, batch):
        # z: [num_nodes_in_batch, latent_dim]
        adj_hat = torch.sigmoid(torch.matmul(z, z.t()))
        return adj_hat
