import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class TrajectoryVAE(nn.Module):
    """Simple VAE for trajectory generation."""
    
    def __init__(
        self,
        input_dim: int = 4,  # lat, lon, time, speed
        sequence_length: int = 50,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        num_layers: int = 2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder_lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True
        )
        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)
        
        # Decoder
        self.fc_latent = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(
            hidden_dim, hidden_dim, num_layers, batch_first=True
        )
        self.fc_out = nn.Linear(hidden_dim, input_dim)
        
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode trajectory to latent distribution parameters."""
        # x shape: (batch, seq_len, input_dim)
        _, (h, _) = self.encoder_lstm(x)
        # Concatenate forward and backward hidden states
        h = torch.cat([h[-2], h[-1]], dim=1)
        
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space to trajectory."""
        batch_size = z.size(0)
        
        # Project latent to hidden
        h = self.fc_latent(z)
        h = h.unsqueeze(1).repeat(1, self.sequence_length, 1)
        
        # Decode sequence
        out, _ = self.decoder_lstm(h)
        out = self.fc_out(out)
        
        return out
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
    
    def generate(self, n_samples: int = 1, device: str = 'cpu') -> torch.Tensor:
        """Generate new trajectories."""
        z = torch.randn(n_samples, self.latent_dim).to(device)
        with torch.no_grad():
            trajectories = self.decode(z)
        return trajectories


def vae_loss(recon: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float = 1.0) -> Tuple[torch.Tensor, dict]:
    """VAE loss function."""
    # Reconstruction loss
    recon_loss = F.mse_loss(recon, x, reduction='sum') / x.size(0)
    
    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    
    # Total loss
    loss = recon_loss + beta * kl_loss
    
    return loss, {
        'loss': loss.item(),
        'recon_loss': recon_loss.item(),
        'kl_loss': kl_loss.item()
    }