import torch
import torch.nn as nn
import torch.nn.functional as F

# Register as functions
@torch.losses.register("cross_entropy")
def cross_entropy_loss(**kwargs):
    """Return a CrossEntropyLoss instance"""
    return nn.CrossEntropyLoss(**kwargs)

@torch.losses.register("mse")
def mse_loss(**kwargs):
    """Return an MSELoss instance"""
    return nn.MSELoss(**kwargs)

@torch.losses.register("gvae_loss")
def gvae_loss(adj_hat, edge_index, mu, logvar):
    num_nodes = adj_hat.size(0)
    adj_true = torch.zeros_like(adj_hat)
    adj_true[edge_index[0], edge_index[1]] = 1
    
    recon_loss = F.binary_cross_entropy(adj_hat, adj_true)
    # KL divergence
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    total_loss = recon_loss + kl_loss
    return total_loss, recon_loss, kl_loss, adj_true

