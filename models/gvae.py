import torch
import torch.nn as nn

from .ecoders import *
from .decoders import *
from .classifiers import *

@torch.models.register("GraphAutoencoder_FCN")
class GraphAutoencoder_FCN(nn.Module):
    def __init__(self, num_features, embed_dim, latent_dim, output_dim):
        super().__init__()

        self.latent_dim = latent_dim
        self.num_features = num_features
        
        # embeddings for positions
        self.position_embeddings = nn.Embedding(401, embed_dim, padding_idx=0)
        
        self.encoder_block = EncoderBlockFCN(num_features * embed_dim, latent_dim)

        self.decoder_block = DecoderBlockFCN(latent_dim, output_dim)

        self.classifier = BinaryClassifierHead(output_dim)
    
    def forward(self, x):
        """
        B: Batch
        N: Node
        F: Feature
        pos_D: Position Embedding (Dimension = num_features * embed_dim)
        """
        B, N, F = x.shape
        
        # expected shape: (B, N, F, pos_D)
        x = self.position_embeddings(x)
        # expected shape: (B, N, F * pos_D)
        x = x.view(B, N, -1)

        # flatten nodes (treating each node independently in FCN)
        # expected shape (B*N, F * pos_D)
        x = x.view(B * N, -1)

        # expected shapes
        # mu: (B*N, latent_dim)
        # log(sigma): (B*N, latent_dim)
        mu, logvar = self.encoder_block(x)
        
        # Reparameterization
        # expected shape: (B*N, latent_dim)
        z = self.reparameterization(mu, logvar)
        

        # Decode back
        # expected output shape: (B*N, output_dim)
        out = self.decoder_block(z)

        # reshape back to (B, N, output_dim)
        out = out.view(B, N, -1)

        # classifier probabilities (B*N, 1) -> reshape (B, N)
        node_probs = self.classifier(out.view(B * N, -1))
        node_probs = node_probs.view(B, N, 1)

        return out, mu, logvar, node_probs

    def reparameterization(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    

@torch.models.register("GVAE_DEFAULT")
class GVAE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_dim):
        super().__init__()
        self.encoder = GVAEEncoder(in_channels, hidden_channels, latent_dim)
        self.decoder = GVAEDecoder()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, edge_index, batch):
        mu, logvar = self.encoder(x, edge_index, batch)
        z = self.reparameterize(mu, logvar)
        adj_hat = self.decoder(z, batch)
        return adj_hat, mu, logvar