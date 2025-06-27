# ml_mobility_ns3/models/vae.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class ConditionalTrajectoryVAE(nn.Module):
    """Conditional VAE for trajectory generation with transport mode and length conditioning."""
    
    def __init__(
        self,
        input_dim: int = 3,  # lat, lon, speed
        sequence_length: int = 2000,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        num_layers: int = 2,
        num_transport_modes: int = 10,  # number of transport modes + multimodal
        condition_dim: int = 32,  # dimension for condition embeddings
    ):
        super().__init__()
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.num_transport_modes = num_transport_modes
        self.num_layers = num_layers
        
        # Condition embeddings
        self.transport_mode_embedding = nn.Embedding(num_transport_modes, condition_dim)
        self.length_projection = nn.Linear(1, condition_dim)  # project normalized length
        
        # Total condition dimension (transport mode + length)
        total_condition_dim = condition_dim * 2
        
        # Encoder
        self.encoder_lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True
        )
        # Include conditions in latent space projection
        self.fc_mu = nn.Linear(hidden_dim * 2 + total_condition_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2 + total_condition_dim, latent_dim)
        
        # Decoder
        self.fc_latent = nn.Linear(latent_dim + total_condition_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(
            hidden_dim, hidden_dim, num_layers, batch_first=True
        )
        self.fc_out = nn.Linear(hidden_dim, input_dim)
        
    def get_conditions(self, transport_mode: torch.Tensor, trip_length: torch.Tensor) -> torch.Tensor:
        """Create condition embeddings from transport mode and trip length."""
        # Transport mode embedding
        mode_embed = self.transport_mode_embedding(transport_mode)  # (batch, condition_dim)
        
        # Length embedding (normalize length and project)
        length_normalized = trip_length.unsqueeze(-1).float() / self.sequence_length  # normalize to [0,1]
        length_embed = self.length_projection(length_normalized)  # (batch, condition_dim)
        
        # Concatenate conditions
        conditions = torch.cat([mode_embed, length_embed], dim=-1)  # (batch, condition_dim * 2)
        return conditions
        
    def encode(self, x: torch.Tensor, conditions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode trajectory to latent distribution parameters."""
        # x shape: (batch, seq_len, input_dim)
        _, (h, _) = self.encoder_lstm(x)
        # Concatenate forward and backward hidden states
        h = torch.cat([h[-2], h[-1]], dim=1)  # (batch, hidden_dim * 2)
        
        # Concatenate with conditions
        h_conditioned = torch.cat([h, conditions], dim=-1)  # (batch, hidden_dim * 2 + condition_dim * 2)
        
        mu = self.fc_mu(h_conditioned)
        logvar = self.fc_logvar(h_conditioned)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        """Decode from latent space to trajectory."""
        batch_size = z.size(0)
        
        # Concatenate latent with conditions
        z_conditioned = torch.cat([z, conditions], dim=-1)  # (batch, latent_dim + condition_dim * 2)
        
        # Project to hidden
        h = self.fc_latent(z_conditioned)
        h = h.unsqueeze(1).repeat(1, self.sequence_length, 1)
        
        # Decode sequence
        out, _ = self.decoder_lstm(h)
        out = self.fc_out(out)
        
        return out
    
    def forward(
        self, 
        x: torch.Tensor, 
        transport_mode: torch.Tensor, 
        trip_length: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass."""
        conditions = self.get_conditions(transport_mode, trip_length)
        mu, logvar = self.encode(x, conditions)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, conditions)
        return recon, mu, logvar
    
    def generate(
        self, 
        transport_mode: torch.Tensor, 
        trip_length: torch.Tensor, 
        n_samples: Optional[int] = None,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """Generate new trajectories given conditions."""
        if n_samples is None:
            n_samples = transport_mode.size(0)
            
        conditions = self.get_conditions(transport_mode, trip_length)
        z = torch.randn(n_samples, self.latent_dim).to(device)
        
        with torch.no_grad():
            trajectories = self.decode(z, conditions)
        return trajectories


def masked_vae_loss(
    recon: torch.Tensor, 
    x: torch.Tensor, 
    mu: torch.Tensor, 
    logvar: torch.Tensor, 
    mask: torch.Tensor,
    beta: float = 1.0
) -> Tuple[torch.Tensor, dict]:
    """
    VAE loss function with masking for variable-length sequences.
    
    Args:
        recon: Reconstructed trajectories (batch, seq_len, input_dim)
        x: Original trajectories (batch, seq_len, input_dim)
        mu: Latent mean (batch, latent_dim)
        logvar: Latent log variance (batch, latent_dim)
        mask: Binary mask indicating valid positions (batch, seq_len)
        beta: Weight for KL loss
    """
    # Reconstruction loss with masking
    # Only compute loss on valid (non-padded) positions
    diff = (recon - x) ** 2  # (batch, seq_len, input_dim)
    
    # Expand mask to match input dimensions
    mask_expanded = mask.unsqueeze(-1).expand_as(diff)  # (batch, seq_len, input_dim)
    
    # Apply mask and compute mean over valid positions
    masked_diff = diff * mask_expanded
    num_valid = mask_expanded.sum()
    recon_loss = masked_diff.sum() / (num_valid + 1e-8)  # avoid division by zero
    
    # KL divergence (not masked, applied to latent space)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    
    # Total loss
    loss = recon_loss + beta * kl_loss
    
    return loss, {
        'loss': loss.item(),
        'recon_loss': recon_loss.item(),
        'kl_loss': kl_loss.item(),
        'num_valid_points': num_valid.item()
    }


def create_mask_from_lengths(trip_lengths: torch.Tensor, max_length: int) -> torch.Tensor:
    """
    Create binary mask from trip lengths.
    
    Args:
        trip_lengths: Actual trip lengths (batch,)
        max_length: Maximum sequence length (padding length)
    
    Returns:
        Binary mask (batch, max_length) where 1 indicates valid position
    """
    batch_size = trip_lengths.size(0)
    # Create range tensor
    range_tensor = torch.arange(max_length).unsqueeze(0).expand(batch_size, -1)
    
    # Create mask: 1 where position < trip_length, 0 otherwise
    mask = (range_tensor < trip_lengths.unsqueeze(1)).float()
    
    return mask


# Example usage and training loop helper
def train_step(
    model: ConditionalTrajectoryVAE,
    x: torch.Tensor,
    transport_mode: torch.Tensor,
    trip_length: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    beta: float = 1.0
) -> dict:
    """Single training step with masked loss."""
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    recon, mu, logvar = model(x, transport_mode, trip_length)
    
    # Create mask from trip lengths
    mask = create_mask_from_lengths(trip_length, model.sequence_length)
    mask = mask.to(x.device)
    
    # Compute loss
    loss, metrics = masked_vae_loss(recon, x, mu, logvar, mask, beta)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    return metrics


# Example of how to use for generation
def generate_trajectories(
    model: ConditionalTrajectoryVAE,
    transport_modes: list,  # e.g., [0, 1, 2] for different modes
    trip_lengths: list,     # e.g., [100, 500, 800] for different lengths
    device: str = 'cpu'
) -> torch.Tensor:
    """Generate trajectories with specific conditions."""
    model.eval()
    
    # Convert to tensors
    mode_tensor = torch.tensor(transport_modes, dtype=torch.long).to(device)
    length_tensor = torch.tensor(trip_lengths, dtype=torch.long).to(device)
    
    # Generate
    generated_trajectories = model.generate(mode_tensor, length_tensor, device=device)
    
    # Only return the valid part of each trajectory
    result = []
    for i, length in enumerate(trip_lengths):
        valid_trajectory = generated_trajectories[i, :length, :]
        result.append(valid_trajectory)
    
    return result