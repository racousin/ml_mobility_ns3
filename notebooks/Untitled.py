#!/usr/bin/env python
# coding: utf-8

# In[17]:


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


# In[20]:


import torch
import numpy as np
from pathlib import Path
import json
import pickle
import sys

# Add project to path
sys.path.append('../..')
from ml_mobility_ns3.models.vae import ConditionalTrajectoryVAE

def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'cpu'  # Use CPU on Mac to avoid MPS issues
    else:
        return 'cpu'

def load_model_simple(checkpoint_dir: Path, device: str = None):
    """Simply load the model without complications."""
    if device is None:
        device = get_device()

    # Load config
    with open(checkpoint_dir / 'config.json', 'r') as f:
        config = json.load(f)

    # Create model
    model_config = config['model_config']
    model = ConditionalTrajectoryVAE(**model_config)

    # Load weights - force CPU first to avoid device issues
    checkpoint = torch.load(checkpoint_dir / 'best_model.pt', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    # Then move to desired device
    model = model.to(device)
    model.eval()

    return model, config

def generate_simple(
    model, 
    transport_mode: str, 
    duration_minutes: float,
    n_samples: int = 10,
    device: str = None
):
    """Generate trajectories with specified parameters."""
    if device is None:
        device = get_device()

    # Transport modes
    modes = ['BIKE', 'CAR', 'MIXED', 'PUBLIC_TRANSPORT', 'WALKING']
    mode_idx = modes.index(transport_mode)

    # Convert duration to steps (2 seconds per step)
    trip_length = int(duration_minutes * 60 / 2)
    trip_length = min(trip_length, 2000)  # Cap at model max

    # Create tensors
    mode_tensor = torch.full((n_samples,), mode_idx, dtype=torch.long).to(device)
    length_tensor = torch.full((n_samples,), trip_length, dtype=torch.long).to(device)

    # Generate
    with torch.no_grad():
        trajectories = model.generate(mode_tensor, length_tensor, device=device)

    return trajectories.cpu().numpy(), trip_length

def load_scaler_simple(preprocessing_dir: Path):
    """Load the scalers dictionary."""
    scaler_path = preprocessing_dir / 'scalers.pkl'
    with open(scaler_path, 'rb') as f:
        return pickle.load(f)

def inverse_transform_simple(trajectories, scaler):
    """Convert from normalized to real units.

    Args:
        trajectories: numpy array of shape (n_samples, seq_len, n_features)
        scaler: Either a sklearn scaler object or a dict containing scalers
    """
    # Handle both scaler object and dict cases
    if isinstance(scaler, dict):
        # Extract the trajectory scaler from the dict
        trajectory_scaler = scaler['trajectory']
    else:
        trajectory_scaler = scaler

    n_samples, seq_len, n_features = trajectories.shape
    traj_flat = trajectories.reshape(-1, n_features)
    traj_real = trajectory_scaler.inverse_transform(traj_flat)
    return traj_real.reshape(n_samples, seq_len, n_features)

def compute_basic_stats(trajectories, trip_length):
    """Compute basic statistics for generated trajectories."""
    stats = []

    for i, traj in enumerate(trajectories):
        # Only look at valid portion
        valid_traj = traj[:trip_length]

        # Speed stats (assuming index 2 is speed in m/s)
        speeds_ms = valid_traj[:, 2]
        speeds_kmh = speeds_ms * 3.6

        # Distance calculation
        if trip_length > 1:
            lat_diff = np.diff(valid_traj[:, 0])
            lon_diff = np.diff(valid_traj[:, 1])
            # Approximate distance in km
            distances = np.sqrt(lat_diff**2 + lon_diff**2) * 111
            total_distance = distances.sum()

            # Bird distance
            start = valid_traj[0, :2]
            end = valid_traj[trip_length-1, :2]
            bird_distance = np.sqrt(((end - start)**2).sum()) * 111
        else:
            total_distance = 0
            bird_distance = 0

        stats.append({
            'trajectory_id': i,
            'mean_speed_kmh': speeds_kmh.mean(),
            'max_speed_kmh': speeds_kmh.max(),
            'total_distance_km': total_distance,
            'bird_distance_km': bird_distance,
            'duration_min': trip_length * 2 / 60
        })

    return stats

# Main usage example
if __name__ == "__main__":
    # Setup paths
    checkpoint_dir = Path("../results/optimal_medium_v2")
    preprocessing_dir = Path("../data/processed ")

    # Get device
    device = get_device()
    print(f"Using device: {device}")

    # Load model
    print("Loading model...")
    model, config = load_model_simple(checkpoint_dir, device)
    print("Model loaded successfully!")

    # Load scalers (this returns a dict)
    print("Loading scalers...")
    scalers = load_scaler_simple(preprocessing_dir)
    print(f"Loaded scalers: {list(scalers.keys())}")

    # Generate trajectories
    transport_mode = "CAR"
    duration_minutes = 20.0
    n_samples = 5

    print(f"\nGenerating {n_samples} {transport_mode} trajectories of {duration_minutes} minutes...")
    trajectories_norm, trip_length = generate_simple(
        model, 
        transport_mode, 
        duration_minutes, 
        n_samples, 
        device
    )

    # Convert to real units - pass the scalers dict (or just the trajectory scaler)
    print("Converting to real units...")
    trajectories_real = inverse_transform_simple(trajectories_norm, scalers)
    # Or alternatively: trajectories_real = inverse_transform_simple(trajectories_norm, scalers['trajectory'])

    # Compute stats
    print("\nComputing statistics...")
    stats = compute_basic_stats(trajectories_real, trip_length)

    # Print results
    print(f"\nGenerated {n_samples} trajectories:")
    print(f"Trip length: {trip_length} steps ({duration_minutes} minutes)")
    print("\nTrajectory Statistics:")
    print("-" * 80)
    print(f"{'ID':>3} | {'Mean Speed':>10} | {'Max Speed':>10} | {'Total Dist':>10} | {'Bird Dist':>10}")
    print(f"{'':>3} | {'(km/h)':>10} | {'(km/h)':>10} | {'(km)':>10} | {'(km)':>10}")
    print("-" * 80)

    for stat in stats:
        print(f"{stat['trajectory_id']:>3} | "
              f"{stat['mean_speed_kmh']:>10.1f} | "
              f"{stat['max_speed_kmh']:>10.1f} | "
              f"{stat['total_distance_km']:>10.2f} | "
              f"{stat['bird_distance_km']:>10.2f}")

    # Average stats
    mean_speed = np.mean([s['mean_speed_kmh'] for s in stats])
    mean_total_dist = np.mean([s['total_distance_km'] for s in stats])
    mean_bird_dist = np.mean([s['bird_distance_km'] for s in stats])

    print("-" * 80)
    print(f"Average: {mean_speed:>19.1f} | {'':>10} | {mean_total_dist:>10.2f} | {mean_bird_dist:>10.2f}")

    # Save one trajectory example
    print(f"\nFirst trajectory sample (first 5 points):")
    print("Lat, Lon, Speed (m/s)")
    for i in range(min(5, trip_length)):
        point = trajectories_real[0, i]
        print(f"{point[0]:.6f}, {point[1]:.6f}, {point[2]:.3f}")


# In[21]:


import torch
import numpy as np
import pandas as pd
from pathlib import Path
import json
import pickle
import folium
from typing import List, Union, Optional, Tuple, Dict
import sys
from collections import defaultdict
from tqdm import tqdm

# Add project to path
sys.path.append('../..')

# Import generation functions from the fixed script
def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'cpu'  # Use CPU on Mac to avoid MPS issues
    else:
        return 'cpu'

def load_model_simple(checkpoint_dir: Path, device: str = None):
    """Simply load the model without complications."""
    if device is None:
        device = get_device()

    # Load config
    with open(checkpoint_dir / 'config.json', 'r') as f:
        config = json.load(f)

    # Create model
    model_config = config['model_config']
    model = ConditionalTrajectoryVAE(**model_config)

    # Load weights - force CPU first to avoid device issues
    checkpoint = torch.load(checkpoint_dir / 'best_model.pt', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    # Then move to desired device
    model = model.to(device)
    model.eval()

    return model, config

def generate_batch_trajectories(
    model, 
    transport_mode: str, 
    trip_lengths: List[int],
    batch_size: int = 32,
    device: str = None
):
    """Generate multiple trajectories in batches for efficiency."""
    if device is None:
        device = get_device()

    # Transport modes
    modes = ['BIKE', 'CAR', 'MIXED', 'PUBLIC_TRANSPORT', 'WALKING']
    mode_idx = modes.index(transport_mode)

    all_trajectories = []

    # Process in batches
    for i in range(0, len(trip_lengths), batch_size):
        batch_lengths = trip_lengths[i:i+batch_size]
        batch_size_actual = len(batch_lengths)

        # Cap lengths at model max
        batch_lengths = [min(l, 2000) for l in batch_lengths]

        # Create tensors
        mode_tensor = torch.full((batch_size_actual,), mode_idx, dtype=torch.long).to(device)
        length_tensor = torch.tensor(batch_lengths, dtype=torch.long).to(device)

        # Generate
        with torch.no_grad():
            batch_trajectories = model.generate(mode_tensor, length_tensor, device=device)

        all_trajectories.append(batch_trajectories.cpu().numpy())

    # Concatenate all batches
    return np.vstack(all_trajectories) if all_trajectories else np.array([])

def load_scaler_simple(preprocessing_dir: Path):
    """Load the scalers dictionary."""
    scaler_path = preprocessing_dir / 'scalers.pkl'
    with open(scaler_path, 'rb') as f:
        return pickle.load(f)

def inverse_transform_simple(trajectories, scaler):
    """Convert from normalized to real units."""
    if isinstance(scaler, dict):
        trajectory_scaler = scaler['trajectory']
    else:
        trajectory_scaler = scaler

    if len(trajectories.shape) == 2:
        # Single trajectory
        return trajectory_scaler.inverse_transform(trajectories)
    else:
        # Multiple trajectories
        n_samples, seq_len, n_features = trajectories.shape
        traj_flat = trajectories.reshape(-1, n_features)
        traj_real = trajectory_scaler.inverse_transform(traj_flat)
        return traj_real.reshape(n_samples, seq_len, n_features)

def compute_trajectory_metrics(trajectory: np.ndarray, valid_length: Optional[int] = None) -> Dict:
    """Compute metrics for a single trajectory."""
    if valid_length is None:
        # Find valid length by checking for zero padding
        valid_mask = ~np.all(trajectory == 0, axis=1)
        valid_length = np.sum(valid_mask)

    # Get valid portion
    valid_traj = trajectory[:valid_length]

    # Duration in minutes (2 seconds per point)
    duration_min = valid_length * 2 / 60

    # Speed statistics
    speeds_ms = valid_traj[:, 2]
    speeds_kmh = speeds_ms * 3.6
    avg_speed = np.mean(speeds_kmh)
    std_speed = np.std(speeds_kmh)
    max_speed = np.max(speeds_kmh)

    # Distance calculations
    if valid_length > 1:
        lat_diff = np.diff(valid_traj[:, 0])
        lon_diff = np.diff(valid_traj[:, 1])
        distances = np.sqrt(lat_diff**2 + lon_diff**2) * 111  # Approximate km
        total_distance = np.sum(distances)

        # Bird distance
        start = valid_traj[0, :2]
        end = valid_traj[-1, :2]
        bird_distance = np.sqrt(np.sum((end - start)**2)) * 111
    else:
        total_distance = 0
        bird_distance = 0

    return {
        'duration_min': duration_min,
        'avg_speed_kmh': avg_speed,
        'std_speed_kmh': std_speed,
        'max_speed_kmh': max_speed,
        'total_distance_km': total_distance,
        'bird_distance_km': bird_distance,
        'valid_points': valid_length
    }

def load_all_real_trajectories(preprocessing_dir: Path, transport_mode: str):
    """Load ALL real trajectories for a transport mode."""
    # Load interpolated trips
    with open(preprocessing_dir / 'interpolated_trips.pkl', 'rb') as f:
        interpolated_trips = pickle.load(f)

    # Filter by transport mode
    mode_trips = [t for t in interpolated_trips if t['category'] == transport_mode]

    if len(mode_trips) == 0:
        print(f"No trips found for mode: {transport_mode}")
        return []

    print(f"Found {len(mode_trips)} {transport_mode} trajectories")

    # Convert all trips
    real_trajectories = []
    for trip in mode_trips:
        # GPS points: [timestamp, lat, lon, speed]
        gps_points = trip['gps_points']
        # Extract lat, lon, speed
        trajectory = gps_points[:, 1:4].astype(np.float32)

        real_trajectories.append({
            'trajectory': trajectory,
            'trip_id': trip['trip_id'],
            'user_id': trip['user_id'],
            'category': trip['category'],
            'trip_type': trip['trip_type'],
            'original_duration': trip['duration_minutes'],
            'length': len(trajectory),
            'weight': trip.get('weight', 1.0)
        })

    return real_trajectories

def generate_all_matched_trajectories(
    model,
    scalers,
    real_trajectories: List[Dict],
    transport_mode: str,
    device: str,
    batch_size: int = 32
) -> List[Dict]:
    """Generate trajectories matching all real trajectories' lengths."""

    # Extract lengths
    trip_lengths = [t['length'] for t in real_trajectories]

    print(f"\nGenerating {len(trip_lengths)} {transport_mode} trajectories...")

    # Generate in batches
    gen_trajectories_norm = generate_batch_trajectories(
        model, transport_mode, trip_lengths, batch_size, device
    )

    # Convert to real units
    print("Converting to real units...")
    gen_trajectories_real = inverse_transform_simple(gen_trajectories_norm, scalers)

    # Create trajectory info list
    generated_trajectories = []
    for i, (gen_traj, real_info) in enumerate(zip(gen_trajectories_real, real_trajectories)):
        generated_trajectories.append({
            'trajectory': gen_traj,
            'matched_to': real_info['trip_id'],
            'category': transport_mode,
            'length': real_info['length'],
            'weight': real_info['weight']
        })

    return generated_trajectories

def compute_aggregate_statistics(trajectories: List[Dict], label: str = "") -> Dict:
    """Compute aggregate statistics for a set of trajectories."""
    all_metrics = []
    total_weight = 0

    for traj_info in trajectories:
        metrics = compute_trajectory_metrics(
            traj_info['trajectory'], 
            traj_info.get('length')
        )
        metrics['weight'] = traj_info.get('weight', 1.0)
        all_metrics.append(metrics)
        total_weight += metrics['weight']

    # Extract arrays for each metric
    durations = np.array([m['duration_min'] for m in all_metrics])
    avg_speeds = np.array([m['avg_speed_kmh'] for m in all_metrics])
    bird_distances = np.array([m['bird_distance_km'] for m in all_metrics])
    total_distances = np.array([m['total_distance_km'] for m in all_metrics])
    weights = np.array([m['weight'] for m in all_metrics])

    # Compute weighted statistics
    def weighted_mean(values, weights):
        return np.sum(values * weights) / np.sum(weights)

    def weighted_std(values, weights):
        mean = weighted_mean(values, weights)
        variance = weighted_mean((values - mean)**2, weights)
        return np.sqrt(variance)

    return {
        'label': label,
        'n_trajectories': len(trajectories),
        'total_weight': total_weight,
        'duration_mean': weighted_mean(durations, weights),
        'duration_std': weighted_std(durations, weights),
        'duration_min': np.min(durations),
        'duration_max': np.max(durations),
        'speed_mean': weighted_mean(avg_speeds, weights),
        'speed_std': weighted_std(avg_speeds, weights),
        'speed_min': np.min(avg_speeds),
        'speed_max': np.max(avg_speeds),
        'bird_distance_mean': weighted_mean(bird_distances, weights),
        'bird_distance_std': weighted_std(bird_distances, weights),
        'bird_distance_min': np.min(bird_distances),
        'bird_distance_max': np.max(bird_distances),
        'total_distance_mean': weighted_mean(total_distances, weights),
        'total_distance_std': weighted_std(total_distances, weights),
        'total_distance_min': np.min(total_distances),
        'total_distance_max': np.max(total_distances),
    }

def print_mode_comparison(mode: str, real_stats: Dict, gen_stats: Dict):
    """Print detailed comparison for a transport mode."""
    print(f"\n{'='*80}")
    print(f"{mode} COMPARISON")
    print('='*80)

    print(f"\nDataset size:")
    print(f"  Real trajectories: {real_stats['n_trajectories']:,} (weight: {real_stats['total_weight']:,.0f})")
    print(f"  Generated trajectories: {gen_stats['n_trajectories']:,} (weight: {gen_stats['total_weight']:,.0f})")

    metrics = [
        ('Duration (min)', 'duration'),
        ('Speed avg (km/h)', 'speed'),
        ('Bird distance (km)', 'bird_distance'),
        ('Total distance (km)', 'total_distance')
    ]

    print(f"\nMetric comparison (weighted statistics):")
    print("-" * 80)

    for metric_name, metric_key in metrics:
        real_mean = real_stats[f'{metric_key}_mean']
        real_std = real_stats[f'{metric_key}_std']
        gen_mean = gen_stats[f'{metric_key}_mean']
        gen_std = gen_stats[f'{metric_key}_std']

        # Calculate relative error
        rel_error = abs(gen_mean - real_mean) / real_mean * 100 if real_mean > 0 else 0

        print(f"\n{metric_name}:")
        print(f"  Real:      {real_mean:.2f} ± {real_std:.2f} (range: [{real_stats[f'{metric_key}_min']:.2f}, {real_stats[f'{metric_key}_max']:.2f}])")
        print(f"  Generated: {gen_mean:.2f} ± {gen_std:.2f} (range: [{gen_stats[f'{metric_key}_min']:.2f}, {gen_stats[f'{metric_key}_max']:.2f}])")
        print(f"  Relative Error: {rel_error:.1f}%")

def create_sample_visualization_map(
    real_trajectories: List[Dict],
    generated_trajectories: List[Dict],
    mode: str,
    n_samples: int = 10,
    output_file: str = None
) -> folium.Map:
    """Create a map with a sample of trajectories for visualization."""
    # Sample trajectories if needed
    if len(real_trajectories) > n_samples:
        sample_indices = np.random.choice(len(real_trajectories), n_samples, replace=False)
        real_sample = [real_trajectories[i] for i in sample_indices]
        gen_sample = [generated_trajectories[i] for i in sample_indices]
    else:
        real_sample = real_trajectories
        gen_sample = generated_trajectories

    # Prepare for visualization
    all_trajectories = []
    labels = []
    types = []

    for i, traj_info in enumerate(real_sample):
        all_trajectories.append(traj_info['trajectory'])
        labels.append(f"Real {mode} {i+1}")
        types.append('real')

    for i, traj_info in enumerate(gen_sample):
        all_trajectories.append(traj_info['trajectory'])
        labels.append(f"Generated {mode} {i+1}")
        types.append('generated')

    # Calculate center
    all_lats = []
    all_lons = []
    for traj in all_trajectories:
        valid_mask = ~np.all(traj == 0, axis=1)
        all_lats.extend(traj[valid_mask, 0])
        all_lons.extend(traj[valid_mask, 1])
    center = (np.mean(all_lats), np.mean(all_lons))

    # Create map
    m = folium.Map(location=center, zoom_start=11)

    # Define colors
    real_colors = ['blue', 'darkblue', 'lightblue', 'navy', 'steelblue']
    generated_colors = ['red', 'darkred', 'orange', 'pink', 'coral']

    # Add trajectories
    for i, (traj, label, traj_type) in enumerate(zip(all_trajectories, labels, types)):
        if traj_type == 'real':
            color = real_colors[i % len(real_colors)]
            line_style = None
        else:
            color = generated_colors[(i - len(real_sample)) % len(generated_colors)]
            line_style = '10'

        # Get valid points
        if traj_type == 'generated':
            traj_info = gen_sample[i - len(real_sample)]
            valid_length = traj_info['length']
            valid_traj = traj[:valid_length]
        else:
            valid_mask = ~np.all(traj == 0, axis=1)
            valid_traj = traj[valid_mask]

        points = [(lat, lon) for lat, lon in valid_traj[:, :2]]

        # Add polyline
        if line_style:
            folium.PolyLine(
                points,
                color=color,
                weight=3,
                opacity=0.7,
                dash_array=line_style,
                popup=label
            ).add_to(m)
        else:
            folium.PolyLine(
                points,
                color=color,
                weight=3,
                opacity=0.8,
                popup=label
            ).add_to(m)

    # Add legend
    legend_html = f'''
    <div style="position: fixed; 
                top: 50px; right: 50px; width: 250px; height: auto;
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
    <p style="margin: 0;"><b>{mode} Trajectories</b></p>
    <p style="margin: 5px 0;">Sample size: {len(real_sample)} of {len(real_trajectories)} total</p>
    <p style="margin: 10px 0 5px 0;"><b>Line styles:</b></p>
    <p style="margin: 5px 0;"><span style="color: blue;">━━━</span> Real trajectories</p>
    <p style="margin: 5px 0;"><span style="color: red;">┅┅┅</span> Generated trajectories</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    if output_file:
        m.save(output_file)
        print(f"Sample visualization saved to: {output_file}")

    return m

def run_complete_analysis(
    checkpoint_dir: Path,
    preprocessing_dir: Path,
    transport_modes: List[str] = ["CAR", "WALKING", "MIXED", "BIKE", "PUBLIC_TRANSPORT"],
    batch_size: int = 32,
    save_sample_maps: bool = True,
    sample_size: int = 10
):
    """Run complete analysis generating trajectories for ALL real trajectories."""

    # Setup
    device = get_device()
    print(f"Using device: {device}")

    # Load model
    print("\nLoading model...")
    model, config = load_model_simple(checkpoint_dir, device)

    # Load scalers
    print("Loading scalers...")
    scalers = load_scaler_simple(preprocessing_dir)

    # Storage for results
    all_results = []

    # Process each transport mode
    for mode in transport_modes:
        print(f"\n{'='*80}")
        print(f"Processing {mode}")
        print('='*80)

        # Load ALL real trajectories for this mode
        real_trajectories = load_all_real_trajectories(preprocessing_dir, mode)

        if len(real_trajectories) == 0:
            print(f"Skipping {mode} - no trajectories found")
            continue

        # Generate matched trajectories for ALL real ones
        generated_trajectories = generate_all_matched_trajectories(
            model, scalers, real_trajectories, mode, device, batch_size
        )

        # Compute aggregate statistics
        print("\nComputing statistics...")
        real_stats = compute_aggregate_statistics(real_trajectories, f"Real {mode}")
        gen_stats = compute_aggregate_statistics(generated_trajectories, f"Generated {mode}")

        # Print comparison
        print_mode_comparison(mode, real_stats, gen_stats)

        # Store results
        result = {
            'mode': mode,
            'real_stats': real_stats,
            'gen_stats': gen_stats,
            'n_trajectories': len(real_trajectories)
        }
        all_results.append(result)

        # Create sample visualization if requested
        if save_sample_maps:
            create_sample_visualization_map(
                real_trajectories,
                generated_trajectories,
                mode,
                n_samples=sample_size,
                output_file=f"{mode.lower()}_sample_trajectories.html"
            )

    # Create summary DataFrame
    print("\n" + "="*80)
    print("SUMMARY ACROSS ALL MODES")
    print("="*80)

    summary_data = []
    for result in all_results:
        mode = result['mode']
        real_stats = result['real_stats']
        gen_stats = result['gen_stats']

        for metric in ['duration', 'speed', 'bird_distance', 'total_distance']:
            summary_data.append({
                'transport_mode': mode,
                'metric': metric,
                'real_mean': real_stats[f'{metric}_mean'],
                'real_std': real_stats[f'{metric}_std'],
                'generated_mean': gen_stats[f'{metric}_mean'],
                'generated_std': gen_stats[f'{metric}_std'],
                'n_samples': result['n_trajectories'],
                'relative_error': abs(gen_stats[f'{metric}_mean'] - real_stats[f'{metric}_mean']) / real_stats[f'{metric}_mean'] * 100 if real_stats[f'{metric}_mean'] > 0 else 0
            })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('all_trajectories_comparison.csv', index=False)
    print(f"\nComplete comparison saved to: all_trajectories_comparison.csv")

    # Print final summary
    print("\nOverall Performance Summary:")
    print("-" * 60)
    metric_names = {
        'duration': 'Duration (min)',
        'speed': 'Speed (km/h)',
        'bird_distance': 'Bird Distance (km)',
        'total_distance': 'Total Distance (km)'
    }

    for metric in ['duration', 'speed', 'bird_distance', 'total_distance']:
        metric_data = summary_df[summary_df['metric'] == metric]
        avg_error = metric_data['relative_error'].mean()
        print(f"{metric_names[metric]}: Average relative error = {avg_error:.1f}%")

    return summary_df, all_results

# Example usage
if __name__ == "__main__":
    # Set paths
    checkpoint_dir = Path("../results/optimal_medium_v2")
    preprocessing_dir = Path("../data/processed")

    # Run complete analysis for ALL trajectories
    summary_df, results = run_complete_analysis(
        checkpoint_dir=checkpoint_dir,
        preprocessing_dir=preprocessing_dir,
        transport_modes=["CAR", "WALKING", "MIXED", "BIKE", "PUBLIC_TRANSPORT"],
        batch_size=32,  # Process in batches for efficiency
        save_sample_maps=True,  # Save sample visualizations
        sample_size=10  # Number of trajectories to show in sample maps
    )


# In[ ]:





# In[22]:


import torch
import numpy as np
import pandas as pd
from pathlib import Path
import json
import pickle
import folium
from typing import List, Union, Optional, Tuple, Dict
import sys
from collections import defaultdict
from tqdm import tqdm

# Add project to path
sys.path.append('../..')

# Import generation functions from the fixed script
def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'cpu'  # Use CPU on Mac to avoid MPS issues
    else:
        return 'cpu'

def load_model_simple(checkpoint_dir: Path, device: str = None):
    """Simply load the model without complications."""
    if device is None:
        device = get_device()

    # Load config
    with open(checkpoint_dir / 'config.json', 'r') as f:
        config = json.load(f)

    # Create model
    model_config = config['model_config']
    model = ConditionalTrajectoryVAE(**model_config)

    # Load weights - force CPU first to avoid device issues
    checkpoint = torch.load(checkpoint_dir / 'best_model.pt', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    # Then move to desired device
    model = model.to(device)
    model.eval()

    return model, config

def generate_batch_trajectories(
    model, 
    transport_mode: str, 
    trip_lengths: List[int],
    batch_size: int = 32,
    device: str = None
):
    """Generate multiple trajectories in batches for efficiency."""
    if device is None:
        device = get_device()

    # Transport modes
    modes = ['BIKE', 'CAR', 'MIXED', 'PUBLIC_TRANSPORT', 'WALKING']
    mode_idx = modes.index(transport_mode)

    all_trajectories = []

    # Process in batches
    for i in range(0, len(trip_lengths), batch_size):
        batch_lengths = trip_lengths[i:i+batch_size]
        batch_size_actual = len(batch_lengths)

        # Cap lengths at model max
        batch_lengths = [min(l, 2000) for l in batch_lengths]

        # Create tensors
        mode_tensor = torch.full((batch_size_actual,), mode_idx, dtype=torch.long).to(device)
        length_tensor = torch.tensor(batch_lengths, dtype=torch.long).to(device)

        # Generate
        with torch.no_grad():
            batch_trajectories = model.generate(mode_tensor, length_tensor, device=device)

        all_trajectories.append(batch_trajectories.cpu().numpy())

    # Concatenate all batches
    return np.vstack(all_trajectories) if all_trajectories else np.array([])

def load_scaler_simple(preprocessing_dir: Path):
    """Load the scalers dictionary."""
    scaler_path = preprocessing_dir / 'scalers.pkl'
    with open(scaler_path, 'rb') as f:
        return pickle.load(f)

def inverse_transform_simple(trajectories, scaler):
    """Convert from normalized to real units."""
    if isinstance(scaler, dict):
        trajectory_scaler = scaler['trajectory']
    else:
        trajectory_scaler = scaler

    if len(trajectories.shape) == 2:
        # Single trajectory
        return trajectory_scaler.inverse_transform(trajectories)
    else:
        # Multiple trajectories
        n_samples, seq_len, n_features = trajectories.shape
        traj_flat = trajectories.reshape(-1, n_features)
        traj_real = trajectory_scaler.inverse_transform(traj_flat)
        return traj_real.reshape(n_samples, seq_len, n_features)

def compute_trajectory_metrics(trajectory: np.ndarray, valid_length: Optional[int] = None) -> Dict:
    """Compute metrics for a single trajectory."""
    if valid_length is None:
        # Find valid length by checking for zero padding
        valid_mask = ~np.all(trajectory == 0, axis=1)
        valid_length = np.sum(valid_mask)

    # Get valid portion
    valid_traj = trajectory[:valid_length]

    # Duration in minutes (2 seconds per point)
    duration_min = valid_length * 2 / 60

    # Speed statistics
    speeds_ms = valid_traj[:, 2]
    speeds_kmh = speeds_ms * 3.6
    avg_speed = np.mean(speeds_kmh)
    std_speed = np.std(speeds_kmh)
    max_speed = np.max(speeds_kmh)

    # Distance calculations
    if valid_length > 1:
        lat_diff = np.diff(valid_traj[:, 0])
        lon_diff = np.diff(valid_traj[:, 1])
        distances = np.sqrt(lat_diff**2 + lon_diff**2) * 111  # Approximate km
        total_distance = np.sum(distances)

        # Bird distance
        start = valid_traj[0, :2]
        end = valid_traj[-1, :2]
        bird_distance = np.sqrt(np.sum((end - start)**2)) * 111
    else:
        total_distance = 0
        bird_distance = 0

    return {
        'duration_min': duration_min,
        'avg_speed_kmh': avg_speed,
        'std_speed_kmh': std_speed,
        'max_speed_kmh': max_speed,
        'total_distance_km': total_distance,
        'bird_distance_km': bird_distance,
        'valid_points': valid_length
    }

def load_all_real_trajectories(preprocessing_dir: Path, transport_mode: str):
    """Load ALL real trajectories for a transport mode."""
    # Load interpolated trips
    with open(preprocessing_dir / 'interpolated_trips.pkl', 'rb') as f:
        interpolated_trips = pickle.load(f)

    # Filter by transport mode
    mode_trips = [t for t in interpolated_trips if t['category'] == transport_mode]

    if len(mode_trips) == 0:
        print(f"No trips found for mode: {transport_mode}")
        return []

    print(f"Found {len(mode_trips)} {transport_mode} trajectories")

    # Convert all trips
    real_trajectories = []
    for trip in mode_trips:
        # GPS points: [timestamp, lat, lon, speed]
        gps_points = trip['gps_points']
        # Extract lat, lon, speed
        trajectory = gps_points[:, 1:4].astype(np.float32)

        real_trajectories.append({
            'trajectory': trajectory,
            'trip_id': trip['trip_id'],
            'user_id': trip['user_id'],
            'category': trip['category'],
            'trip_type': trip['trip_type'],
            'original_duration': trip['duration_minutes'],
            'length': len(trajectory),
            'weight': trip.get('weight', 1.0)
        })

    return real_trajectories

def generate_all_matched_trajectories(
    model,
    scalers,
    real_trajectories: List[Dict],
    transport_mode: str,
    device: str,
    batch_size: int = 32
) -> List[Dict]:
    """Generate trajectories matching all real trajectories' lengths."""

    # Extract lengths
    trip_lengths = [t['length'] for t in real_trajectories]

    print(f"\nGenerating {len(trip_lengths)} {transport_mode} trajectories...")

    # Generate in batches
    gen_trajectories_norm = generate_batch_trajectories(
        model, transport_mode, trip_lengths, batch_size, device
    )

    # Convert to real units
    print("Converting to real units...")
    gen_trajectories_real = inverse_transform_simple(gen_trajectories_norm, scalers)

    # Create trajectory info list
    generated_trajectories = []
    for i, (gen_traj, real_info) in enumerate(zip(gen_trajectories_real, real_trajectories)):
        generated_trajectories.append({
            'trajectory': gen_traj,
            'matched_to': real_info['trip_id'],
            'category': transport_mode,
            'length': real_info['length'],
            'weight': real_info['weight']
        })

    return generated_trajectories

def compute_aggregate_statistics(trajectories: List[Dict], label: str = "") -> Dict:
    """Compute aggregate statistics for a set of trajectories."""
    all_metrics = []
    total_weight = 0

    for traj_info in trajectories:
        metrics = compute_trajectory_metrics(
            traj_info['trajectory'], 
            traj_info.get('length')
        )
        metrics['weight'] = traj_info.get('weight', 1.0)
        all_metrics.append(metrics)
        total_weight += metrics['weight']

    # Extract arrays for each metric
    durations = np.array([m['duration_min'] for m in all_metrics])
    avg_speeds = np.array([m['avg_speed_kmh'] for m in all_metrics])
    bird_distances = np.array([m['bird_distance_km'] for m in all_metrics])
    total_distances = np.array([m['total_distance_km'] for m in all_metrics])
    weights = np.array([m['weight'] for m in all_metrics])

    # Compute weighted statistics
    def weighted_mean(values, weights):
        return np.sum(values * weights) / np.sum(weights)

    def weighted_std(values, weights):
        mean = weighted_mean(values, weights)
        variance = weighted_mean((values - mean)**2, weights)
        return np.sqrt(variance)

    return {
        'label': label,
        'n_trajectories': len(trajectories),
        'total_weight': total_weight,
        'duration_mean': weighted_mean(durations, weights),
        'duration_std': weighted_std(durations, weights),
        'duration_min': np.min(durations),
        'duration_max': np.max(durations),
        'speed_mean': weighted_mean(avg_speeds, weights),
        'speed_std': weighted_std(avg_speeds, weights),
        'speed_min': np.min(avg_speeds),
        'speed_max': np.max(avg_speeds),
        'bird_distance_mean': weighted_mean(bird_distances, weights),
        'bird_distance_std': weighted_std(bird_distances, weights),
        'bird_distance_min': np.min(bird_distances),
        'bird_distance_max': np.max(bird_distances),
        'total_distance_mean': weighted_mean(total_distances, weights),
        'total_distance_std': weighted_std(total_distances, weights),
        'total_distance_min': np.min(total_distances),
        'total_distance_max': np.max(total_distances),
    }

def print_mode_comparison(mode: str, real_stats: Dict, gen_stats: Dict):
    """Print detailed comparison for a transport mode."""
    print(f"\n{'='*80}")
    print(f"{mode} COMPARISON")
    print('='*80)

    print(f"\nDataset size:")
    print(f"  Real trajectories: {real_stats['n_trajectories']:,} (weight: {real_stats['total_weight']:,.0f})")
    print(f"  Generated trajectories: {gen_stats['n_trajectories']:,} (weight: {gen_stats['total_weight']:,.0f})")

    metrics = [
        ('Duration (min)', 'duration'),
        ('Speed avg (km/h)', 'speed'),
        ('Bird distance (km)', 'bird_distance'),
        ('Total distance (km)', 'total_distance')
    ]

    print(f"\nMetric comparison (weighted statistics):")
    print("-" * 80)

    for metric_name, metric_key in metrics:
        real_mean = real_stats[f'{metric_key}_mean']
        real_std = real_stats[f'{metric_key}_std']
        gen_mean = gen_stats[f'{metric_key}_mean']
        gen_std = gen_stats[f'{metric_key}_std']

        # Calculate relative error
        rel_error = abs(gen_mean - real_mean) / real_mean * 100 if real_mean > 0 else 0

        print(f"\n{metric_name}:")
        print(f"  Real:      {real_mean:.2f} ± {real_std:.2f} (range: [{real_stats[f'{metric_key}_min']:.2f}, {real_stats[f'{metric_key}_max']:.2f}])")
        print(f"  Generated: {gen_mean:.2f} ± {gen_std:.2f} (range: [{gen_stats[f'{metric_key}_min']:.2f}, {gen_stats[f'{metric_key}_max']:.2f}])")
        print(f"  Relative Error: {rel_error:.1f}%")

def create_sample_visualization_map(
    real_trajectories: List[Dict],
    generated_trajectories: List[Dict],
    mode: str,
    n_samples: int = 10,
    output_file: str = None
) -> folium.Map:
    """Create a map with a sample of trajectories for visualization."""
    # Sample trajectories if needed
    if len(real_trajectories) > n_samples:
        sample_indices = np.random.choice(len(real_trajectories), n_samples, replace=False)
        real_sample = [real_trajectories[i] for i in sample_indices]
        gen_sample = [generated_trajectories[i] for i in sample_indices]
    else:
        real_sample = real_trajectories
        gen_sample = generated_trajectories

    # Prepare for visualization
    all_trajectories = []
    labels = []
    types = []

    for i, traj_info in enumerate(real_sample):
        all_trajectories.append(traj_info['trajectory'])
        labels.append(f"Real {mode} {i+1}")
        types.append('real')

    for i, traj_info in enumerate(gen_sample):
        all_trajectories.append(traj_info['trajectory'])
        labels.append(f"Generated {mode} {i+1}")
        types.append('generated')

    # Calculate center
    all_lats = []
    all_lons = []
    for traj in all_trajectories:
        valid_mask = ~np.all(traj == 0, axis=1)
        all_lats.extend(traj[valid_mask, 0])
        all_lons.extend(traj[valid_mask, 1])
    center = (np.mean(all_lats), np.mean(all_lons))

    # Create map
    m = folium.Map(location=center, zoom_start=11)

    # Define colors
    real_colors = ['blue', 'darkblue', 'lightblue', 'navy', 'steelblue']
    generated_colors = ['red', 'darkred', 'orange', 'pink', 'coral']

    # Add trajectories
    for i, (traj, label, traj_type) in enumerate(zip(all_trajectories, labels, types)):
        if traj_type == 'real':
            color = real_colors[i % len(real_colors)]
            line_style = None
        else:
            color = generated_colors[(i - len(real_sample)) % len(generated_colors)]
            line_style = '10'

        # Get valid points
        if traj_type == 'generated':
            traj_info = gen_sample[i - len(real_sample)]
            valid_length = traj_info['length']
            valid_traj = traj[:valid_length]
        else:
            valid_mask = ~np.all(traj == 0, axis=1)
            valid_traj = traj[valid_mask]

        points = [(lat, lon) for lat, lon in valid_traj[:, :2]]

        # Add polyline
        if line_style:
            folium.PolyLine(
                points,
                color=color,
                weight=3,
                opacity=0.7,
                dash_array=line_style,
                popup=label
            ).add_to(m)
        else:
            folium.PolyLine(
                points,
                color=color,
                weight=3,
                opacity=0.8,
                popup=label
            ).add_to(m)

    # Add legend
    legend_html = f'''
    <div style="position: fixed; 
                top: 50px; right: 50px; width: 250px; height: auto;
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
    <p style="margin: 0;"><b>{mode} Trajectories</b></p>
    <p style="margin: 5px 0;">Sample size: {len(real_sample)} of {len(real_trajectories)} total</p>
    <p style="margin: 10px 0 5px 0;"><b>Line styles:</b></p>
    <p style="margin: 5px 0;"><span style="color: blue;">━━━</span> Real trajectories</p>
    <p style="margin: 5px 0;"><span style="color: red;">┅┅┅</span> Generated trajectories</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    if output_file:
        m.save(output_file)
        print(f"Sample visualization saved to: {output_file}")

    return m

def create_all_generated_trajectories_map(
    all_generated_trajectories: Dict[str, List[Dict]],
    output_file: str = "all_generated_trajectories.html",
    max_trajectories_per_mode: int = None
) -> folium.Map:
    """
    Create a map with ALL generated trajectories, colored by transport type.

    Args:
        all_generated_trajectories: Dict with transport mode as key and list of trajectory dicts as value
        output_file: Path to save the HTML file
        max_trajectories_per_mode: Optional limit on trajectories per mode (for performance)
    """

    # Define colors for each transport mode
    mode_colors = {
        'CAR': '#FF0000',        # Red
        'WALKING': '#00FF00',    # Green
        'BIKE': '#0000FF',       # Blue
        'PUBLIC_TRANSPORT': '#FF00FF',  # Magenta
        'MIXED': '#FFA500'       # Orange
    }

    # Collect all trajectories with their modes
    all_trajectories = []
    all_lats = []
    all_lons = []

    total_trajectories = 0
    for mode, trajectories in all_generated_trajectories.items():
        if len(trajectories) == 0:
            continue

        # Limit trajectories per mode if specified
        mode_trajectories = trajectories
        if max_trajectories_per_mode and len(trajectories) > max_trajectories_per_mode:
            sample_indices = np.random.choice(len(trajectories), max_trajectories_per_mode, replace=False)
            mode_trajectories = [trajectories[i] for i in sample_indices]

        for i, traj_info in enumerate(mode_trajectories):
            traj = traj_info['trajectory']
            valid_length = traj_info['length']
            valid_traj = traj[:valid_length]

            # Extract coordinates
            points = [(lat, lon) for lat, lon in valid_traj[:, :2]]
            if len(points) > 1:  # Only add trajectories with more than 1 point
                all_trajectories.append({
                    'points': points,
                    'mode': mode,
                    'color': mode_colors.get(mode, '#808080'),  # Default to gray
                    'label': f"{mode} Trajectory {total_trajectories + 1}"
                })

                # Collect coordinates for centering
                lats, lons = zip(*points)
                all_lats.extend(lats)
                all_lons.extend(lons)
                total_trajectories += 1

    print(f"Creating map with {total_trajectories} generated trajectories")

    if not all_trajectories:
        print("No valid trajectories found!")
        return None

    # Calculate center
    center = (np.mean(all_lats), np.mean(all_lons))

    # Create map
    m = folium.Map(location=center, zoom_start=10)

    # Add trajectories
    for traj_data in all_trajectories:
        folium.PolyLine(
            traj_data['points'],
            color=traj_data['color'],
            weight=2,
            opacity=0.7,
            popup=traj_data['label']
        ).add_to(m)

    # Create legend
    legend_items = []
    mode_counts = {}
    for mode, trajectories in all_generated_trajectories.items():
        if len(trajectories) > 0:
            actual_count = len(trajectories)
            displayed_count = min(actual_count, max_trajectories_per_mode) if max_trajectories_per_mode else actual_count
            mode_counts[mode] = (displayed_count, actual_count)
            color = mode_colors.get(mode, '#808080')
            legend_items.append(f'<span style="color: {color};">●</span> {mode}: {displayed_count:,} trajectories')
            if max_trajectories_per_mode and actual_count > max_trajectories_per_mode:
                legend_items[-1] += f' (of {actual_count:,} total)'

    legend_html = f'''
    <div style="position: fixed; 
                top: 20px; right: 20px; width: 300px; height: auto;
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 15px; border-radius: 5px;">
    <h4 style="margin: 0 0 10px 0;">Generated Trajectories by Transport Mode</h4>
    <p style="margin: 5px 0;"><b>Total displayed: {total_trajectories:,} trajectories</b></p>
    {'<br>'.join(legend_items)}
    <p style="margin: 10px 0 0 0; font-size: 12px; color: #666;">
    Click on any trajectory line for details
    </p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    # Save map
    m.save(output_file)
    print(f"All generated trajectories visualization saved to: {output_file}")

    # Print summary
    print(f"\nVisualization Summary:")
    for mode, (displayed, total) in mode_counts.items():
        print(f"  {mode}: {displayed:,} trajectories displayed (of {total:,} total)")

    return m

def run_complete_analysis(
    checkpoint_dir: Path,
    preprocessing_dir: Path,
    transport_modes: List[str] = ["CAR", "WALKING", "MIXED", "BIKE", "PUBLIC_TRANSPORT"],
    batch_size: int = 32,
    save_sample_maps: bool = True,
    sample_size: int = 10,
    create_all_trajectories_map: bool = True,
    max_trajectories_per_mode: int = 500  # Limit for performance
):
    """Run complete analysis generating trajectories for ALL real trajectories."""

    # Setup
    device = get_device()
    print(f"Using device: {device}")

    # Load model
    print("\nLoading model...")
    model, config = load_model_simple(checkpoint_dir, device)

    # Load scalers
    print("Loading scalers...")
    scalers = load_scaler_simple(preprocessing_dir)

    # Storage for results
    all_results = []
    all_generated_trajectories = {}

    # Process each transport mode
    for mode in transport_modes:
        print(f"\n{'='*80}")
        print(f"Processing {mode}")
        print('='*80)

        # Load ALL real trajectories for this mode
        real_trajectories = load_all_real_trajectories(preprocessing_dir, mode)

        if len(real_trajectories) == 0:
            print(f"Skipping {mode} - no trajectories found")
            continue

        # Generate matched trajectories for ALL real ones
        generated_trajectories = generate_all_matched_trajectories(
            model, scalers, real_trajectories, mode, device, batch_size
        )

        # Store generated trajectories for the combined map
        all_generated_trajectories[mode] = generated_trajectories

        # Compute aggregate statistics
        print("\nComputing statistics...")
        real_stats = compute_aggregate_statistics(real_trajectories, f"Real {mode}")
        gen_stats = compute_aggregate_statistics(generated_trajectories, f"Generated {mode}")

        # Print comparison
        print_mode_comparison(mode, real_stats, gen_stats)

        # Store results
        result = {
            'mode': mode,
            'real_stats': real_stats,
            'gen_stats': gen_stats,
            'n_trajectories': len(real_trajectories)
        }
        all_results.append(result)

        # Create sample visualization if requested
        if save_sample_maps:
            create_sample_visualization_map(
                real_trajectories,
                generated_trajectories,
                mode,
                n_samples=sample_size,
                output_file=f"{mode.lower()}_sample_trajectories.html"
            )

    # Create map with ALL generated trajectories
    if create_all_trajectories_map and all_generated_trajectories:
        print(f"\n{'='*80}")
        print("Creating combined map with all generated trajectories...")
        print('='*80)

        create_all_generated_trajectories_map(
            all_generated_trajectories,
            output_file="all_generated_trajectories.html",
            max_trajectories_per_mode=max_trajectories_per_mode
        )

    # Create summary DataFrame
    print("\n" + "="*80)
    print("SUMMARY ACROSS ALL MODES")
    print("="*80)

    summary_data = []
    for result in all_results:
        mode = result['mode']
        real_stats = result['real_stats']
        gen_stats = result['gen_stats']

        for metric in ['duration', 'speed', 'bird_distance', 'total_distance']:
            summary_data.append({
                'transport_mode': mode,
                'metric': metric,
                'real_mean': real_stats[f'{metric}_mean'],
                'real_std': real_stats[f'{metric}_std'],
                'generated_mean': gen_stats[f'{metric}_mean'],
                'generated_std': gen_stats[f'{metric}_std'],
                'n_samples': result['n_trajectories'],
                'relative_error': abs(gen_stats[f'{metric}_mean'] - real_stats[f'{metric}_mean']) / real_stats[f'{metric}_mean'] * 100 if real_stats[f'{metric}_mean'] > 0 else 0
            })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('all_trajectories_comparison.csv', index=False)
    print(f"\nComplete comparison saved to: all_trajectories_comparison.csv")

    # Print final summary
    print("\nOverall Performance Summary:")
    print("-" * 60)
    metric_names = {
        'duration': 'Duration (min)',
        'speed': 'Speed (km/h)',
        'bird_distance': 'Bird Distance (km)',
        'total_distance': 'Total Distance (km)'
    }

    for metric in ['duration', 'speed', 'bird_distance', 'total_distance']:
        metric_data = summary_df[summary_df['metric'] == metric]
        avg_error = metric_data['relative_error'].mean()
        print(f"{metric_names[metric]}: Average relative error = {avg_error:.1f}%")

    return summary_df, all_results, all_generated_trajectories

# Example usage
if __name__ == "__main__":
    # Set paths
    checkpoint_dir = Path("../results/optimal_medium_v2")
    preprocessing_dir = Path("../data/processed")

    # Run complete analysis for ALL trajectories
    summary_df, results, all_generated = run_complete_analysis(
        checkpoint_dir=checkpoint_dir,
        preprocessing_dir=preprocessing_dir,
        transport_modes=["CAR", "WALKING", "MIXED", "BIKE", "PUBLIC_TRANSPORT"],
        batch_size=32,  # Process in batches for efficiency
        save_sample_maps=True,  # Save sample visualizations
        sample_size=10,  # Number of trajectories to show in sample maps
        create_all_trajectories_map=True,  # Create the new combined map
        max_trajectories_per_mode=500  # Limit trajectories per mode for performance
    )


# In[ ]:




