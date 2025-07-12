import torch
import torch.nn as nn
from .base import BaseTrajectoryModel


class DummyModel(BaseTrajectoryModel):
    """
    Minimal dummy model for testing the pipeline.
    Matches VAE interface with num_transport_modes parameter.
    """
    def __init__(self, input_dim=3, sequence_length=2000, 
                 num_transport_modes=5, latent_dim=16, **kwargs):
        config = locals()
        config.pop('self')
        config.pop('kwargs')
        super().__init__(config)
        
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.num_transport_modes = num_transport_modes
        self.latent_dim = latent_dim
        
        # Simple encoder layers
        self.encoder_mu = nn.Linear(input_dim, latent_dim)
        self.encoder_logvar = nn.Linear(input_dim, latent_dim)
        
        # Simple decoder
        self.decoder = nn.Linear(latent_dim, input_dim)
        
        # Mode embeddings
        self.mode_embeddings = nn.Embedding(num_transport_modes, latent_dim)
        
    def forward(self, x: torch.Tensor, *args, **kwargs):
        """
        Simple forward pass with small perturbation
        """
        batch_size, seq_len, _ = x.shape
        
        # Pool over sequence for encoding
        x_pooled = x.mean(dim=1)
        
        # Get mu and logvar
        mu = self.encoder_mu(x_pooled)
        logvar = self.encoder_logvar(x_pooled)
        
        # Simple reconstruction with small noise
        noise = torch.randn_like(x) * 0.01
        recon = x + noise
        
        return {
            'recon': recon,
            'mu': mu,
            'logvar': logvar
        }
    
    def generate(self, conditions, n_samples):
        """
        Generate random trajectories
        """
        device = next(self.parameters()).device
        
        # Just generate random trajectories
        trajectories = torch.randn(n_samples, self.sequence_length, self.input_dim, device=device)
        
        return trajectories