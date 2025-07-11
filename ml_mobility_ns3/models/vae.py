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
        self.fc_latent_h = nn.Linear(latent_dim + total_condition_dim, hidden_dim * num_layers)
        self.fc_latent_c = nn.Linear(latent_dim + total_condition_dim, hidden_dim * num_layers)
        
        # Learnable start token for decoder input
        self.decoder_start_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.01)

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
        
        # Initialize LSTM states from latent code (THIS IS THE KEY CHANGE)
        h_0 = self.fc_latent_h(z_conditioned).view(self.num_layers, batch_size, self.hidden_dim)
        c_0 = self.fc_latent_c(z_conditioned).view(self.num_layers, batch_size, self.hidden_dim)
        
        # Create decoder input sequence from learnable start token
        decoder_input = self.decoder_start_token.repeat(batch_size, self.sequence_length, 1)
        
        # Add noise during training for robustness
        if self.training:
            decoder_input = decoder_input + torch.randn_like(decoder_input) * 0.02
        
        # Decode sequence with initialized states
        out, _ = self.decoder_lstm(decoder_input, (h_0, c_0))
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
    beta: float = 1.0,
    lambda_dist: float = 0.5
) -> Tuple[torch.Tensor, dict]:
    """Enhanced VAE loss with distance preservation and numerical stability."""
    
    # Ensure numerical stability
    logvar = torch.clamp(logvar, min=-10, max=10)
    
    # 1. Standard reconstruction loss
    mask_expanded = mask.unsqueeze(-1).expand_as(x)
    diff = (recon - x) ** 2
    masked_diff = diff * mask_expanded
    num_valid = mask_expanded.sum()
    recon_loss = masked_diff.sum() / (num_valid + 1e-8)
    
    # Compute segment distances with stability
    def compute_segment_distances(traj, mask):
        # Distance between consecutive points
        diff = traj[:, 1:, :2] - traj[:, :-1, :2]  # lat, lon differences
        # Add small epsilon for numerical stability
        distances = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-8) * 111  # km
        
        # Mask for valid segments (both points must be valid)
        seg_mask = mask[:, 1:] * mask[:, :-1]
        
        return distances, seg_mask
    
    real_dists, seg_mask = compute_segment_distances(x, mask)
    recon_dists, _ = compute_segment_distances(recon, mask)
    
    # Distance loss: preserve segment lengths
    dist_diff = (recon_dists - real_dists) ** 2 * seg_mask
    distance_loss = dist_diff.sum() / (seg_mask.sum() + 1e-8)
    
    # 3. Total trajectory length preservation
    real_total = (real_dists * seg_mask).sum(dim=1)
    recon_total = (recon_dists * seg_mask).sum(dim=1)
    total_dist_loss = F.mse_loss(recon_total, real_total)
    
    # 4. KL divergence with numerical stability
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Combined loss
    total_loss = recon_loss + beta * kl_loss + lambda_dist * (distance_loss + total_dist_loss)
    
    # Check for NaN and replace with large value to continue training
    if torch.isnan(total_loss):
        print(f"NaN detected in loss! recon: {recon_loss}, kl: {kl_loss}, dist: {distance_loss}")
        total_loss = torch.tensor(1e6, device=total_loss.device, requires_grad=True)
    
    # Return with expected metric names for trainer
    return total_loss, {
        'loss': total_loss.item() if not torch.isnan(total_loss) else float('inf'),
        'Recon': recon_loss.item() if not torch.isnan(recon_loss) else float('inf'),
        'KL': kl_loss.item() if not torch.isnan(kl_loss) else float('inf'),
        'distance_loss': distance_loss.item() if not torch.isnan(distance_loss) else float('inf'),
        'total_dist_loss': total_dist_loss.item() if not torch.isnan(total_dist_loss) else float('inf'),
        'mean_real_dist': real_total.mean().item() if not torch.isnan(real_total.mean()) else 0.0,
        'mean_recon_dist': recon_total.mean().item() if not torch.isnan(recon_total.mean()) else 0.0
    }


def compute_trajectory_metrics(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:
    """
    Compute trajectory-specific metrics with correct naming.
    
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
    
    # Speed MAE (with correct key name)
    speed_mae = torch.abs(pred_masked[:, :, 2] - target_masked[:, :, 2])
    speed_mae = (speed_mae * mask).sum() / (mask.sum() + 1e-8)
    
    # Distance metrics (using lat, lon)
    def compute_distances(traj):
        """Compute total and bird distances for a trajectory."""
        # Total distance (sum of segments)
        lat_diff = torch.diff(traj[:, :, 0], dim=1)
        lon_diff = torch.diff(traj[:, :, 1], dim=1)
        # Add epsilon for numerical stability
        segment_distances = torch.sqrt(lat_diff**2 + lon_diff**2 + 1e-8) * 111  # rough km conversion
        total_distance = segment_distances.sum(dim=1)
        
        # Bird distance (start to end)
        valid_lengths = mask.sum(dim=1).long()
        bird_distances = []
        for i, length in enumerate(valid_lengths):
            if length > 1:
                start = traj[i, 0, :2]
                end = traj[i, length-1, :2]
                bird_dist = torch.sqrt(((end - start)**2).sum() + 1e-8) * 111
                bird_distances.append(bird_dist)
            else:
                bird_distances.append(torch.tensor(0.0, device=traj.device))
        
        return total_distance, torch.stack(bird_distances)
    
    pred_total_dist, pred_bird_dist = compute_distances(pred_masked)
    target_total_dist, target_bird_dist = compute_distances(target_masked)
    
    # Distance MAEs
    total_dist_mae = torch.abs(pred_total_dist - target_total_dist).mean()
    bird_dist_mae = torch.abs(pred_bird_dist - target_bird_dist).mean()
    
    # Check for NaN values and handle them
    def safe_item(tensor):
        if torch.isnan(tensor):
            return 0.0
        return tensor.item()
    
    return {
        'Speed MAE': safe_item(speed_mae),  # Changed key name to match trainer expectations
        'total_distance_mae': safe_item(total_dist_mae),
        'bird_distance_mae': safe_item(bird_dist_mae),
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