# ml_mobility_ns3/models/vae.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import numpy as np


class ConditionalTrajectoryVAE(nn.Module):
    """LSTM-based Conditional VAE for trajectory generation with transport mode and length conditioning."""
    
    def __init__(
        self,
        input_dim: int = 3,  # lat, lon, speed
        sequence_length: int = 2000,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        num_layers: int = 2,
        num_transport_modes: int = 5,  # number of transport modes
        condition_dim: int = 32,  # dimension for condition embeddings
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.num_transport_modes = num_transport_modes
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Condition embeddings
        self.transport_mode_embedding = nn.Embedding(num_transport_modes, condition_dim)
        self.length_projection = nn.Linear(1, condition_dim)
        
        # Total condition dimension (transport mode + length)
        total_condition_dim = condition_dim * 2
        
        # Encoder - LSTM with bidirectional processing
        self.encoder_lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, 
            batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0
        )
        encoder_output_dim = hidden_dim * 2  # bidirectional
        
        # Latent space projections (include conditions)
        self.fc_mu = nn.Linear(encoder_output_dim + total_condition_dim, latent_dim)
        self.fc_logvar = nn.Linear(encoder_output_dim + total_condition_dim, latent_dim)
        
        # Decoder
        self.fc_latent = nn.Linear(latent_dim + total_condition_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(
            hidden_dim, hidden_dim, num_layers, 
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.fc_out = nn.Linear(hidden_dim, input_dim)
        
        # Dropout layers
        self.dropout_layer = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.uniform_(param, -0.1, 0.1)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
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
        # LSTM encoding
        _, (h, _) = self.encoder_lstm(x)
        # Concatenate forward and backward hidden states from last layer
        h = torch.cat([h[-2], h[-1]], dim=1)  # (batch, hidden_dim * 2)
        h = self.dropout_layer(h)
        
        # Concatenate with conditions
        h_conditioned = torch.cat([h, conditions], dim=-1)
        
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
        z_conditioned = torch.cat([z, conditions], dim=-1)
        
        # Project to hidden and create sequence
        h = self.fc_latent(z_conditioned)
        h = torch.tanh(h)  # Add non-linearity
        h = h.unsqueeze(1).repeat(1, self.sequence_length, 1)
        
        # Decode sequence
        out, _ = self.decoder_lstm(h)
        out = self.dropout_layer(out)
        out = self.fc_out(out)
        
        return out
    
    def forward(
        self, 
        x: torch.Tensor, 
        transport_mode: torch.Tensor, 
        trip_length: torch.Tensor,
        mask: Optional[torch.Tensor] = None
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
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            'input_dim': self.input_dim,
            'sequence_length': self.sequence_length,
            'hidden_dim': self.hidden_dim,
            'latent_dim': self.latent_dim,
            'num_layers': self.num_layers,
            'num_transport_modes': self.num_transport_modes,
            'condition_dim': self.condition_dim,
            'dropout': self.dropout,
        }


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
    # Reconstruction loss with masking (MSE)
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


def compute_trajectory_metrics(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:
    """
    Compute trajectory-specific metrics.
    
    Args:
        pred: Predicted trajectories (batch, seq_len, 3) [lat, lon, speed]
        target: Target trajectories (batch, seq_len, 3) [lat, lon, speed]
        mask: Binary mask (batch, seq_len)
    
    Returns:
        Dictionary of metrics
    """
    # Apply mask
    mask_expanded = mask.unsqueeze(-1)
    pred_masked = pred * mask_expanded
    target_masked = target * mask_expanded
    
    # Speed MAE
    speed_mae = torch.abs(pred_masked[:, :, 2] - target_masked[:, :, 2])
    speed_mae = (speed_mae * mask).sum() / (mask.sum() + 1e-8)
    
    # Distance metrics (using lat, lon)
    def compute_distances(traj):
        """Compute total and bird distances for a trajectory."""
        # Total distance (sum of segments)
        lat_diff = torch.diff(traj[:, :, 0], dim=1)
        lon_diff = torch.diff(traj[:, :, 1], dim=1)
        segment_distances = torch.sqrt(lat_diff**2 + lon_diff**2) * 111  # rough km conversion
        total_distance = segment_distances.sum(dim=1)
        
        # Bird distance (start to end)
        valid_lengths = mask.sum(dim=1)
        bird_distances = []
        for i, length in enumerate(valid_lengths):
            if length > 1:
                start = traj[i, 0, :2]
                end = traj[i, length-1, :2]
                bird_dist = torch.sqrt(((end - start)**2).sum()) * 111
                bird_distances.append(bird_dist)
            else:
                bird_distances.append(torch.tensor(0.0, device=traj.device))
        
        return total_distance, torch.stack(bird_distances)
    
    pred_total_dist, pred_bird_dist = compute_distances(pred_masked)
    target_total_dist, target_bird_dist = compute_distances(target_masked)
    
    # Distance MAEs
    total_dist_mae = torch.abs(pred_total_dist - target_total_dist).mean()
    bird_dist_mae = torch.abs(pred_bird_dist - target_bird_dist).mean()
    
    return {
        'speed_mae': speed_mae.item(),
        'total_distance_mae': total_dist_mae.item(),
        'bird_distance_mae': bird_dist_mae.item(),
    }