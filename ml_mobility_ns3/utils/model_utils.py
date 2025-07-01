# ml_mobility_ns3/utils/model_utils.py
"""Utilities for loading and using trained VAE models."""

import torch
from pathlib import Path
import json
from typing import Dict, Any, Optional, Tuple
import numpy as np

from ..models.vae import ConditionalTrajectoryVAE


def load_model_from_checkpoint(checkpoint_path: Path, device: str = 'cpu') -> Tuple[ConditionalTrajectoryVAE, Dict[str, Any]]:
    """
    Load a trained VAE model from checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model on
        
    Returns:
        model: The loaded model
        config: Model configuration
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    # Create model with saved configuration
    model = ConditionalTrajectoryVAE(**config)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, config


def generate_trajectories(
    model: ConditionalTrajectoryVAE,
    transport_modes: np.ndarray,
    trip_lengths: np.ndarray,
    n_samples: Optional[int] = None,
    device: str = 'cpu',
    batch_size: int = 64
) -> np.ndarray:
    """
    Generate trajectories using a trained model.
    
    Args:
        model: Trained VAE model
        transport_modes: Array of transport mode indices
        trip_lengths: Array of trip lengths
        n_samples: Number of samples to generate (if None, uses length of transport_modes)
        device: Device to run generation on
        batch_size: Batch size for generation
        
    Returns:
        Generated trajectories as numpy array
    """
    if n_samples is None:
        n_samples = len(transport_modes)
    
    # Convert to tensors
    transport_modes = torch.tensor(transport_modes, dtype=torch.long).to(device)
    trip_lengths = torch.tensor(trip_lengths, dtype=torch.long).to(device)
    
    # Generate in batches
    all_trajectories = []
    
    for i in range(0, n_samples, batch_size):
        batch_end = min(i + batch_size, n_samples)
        batch_modes = transport_modes[i:batch_end]
        batch_lengths = trip_lengths[i:batch_end]
        
        with torch.no_grad():
            batch_trajectories = model.generate(
                batch_modes, 
                batch_lengths,
                device=device
            )
        
        all_trajectories.append(batch_trajectories.cpu().numpy())
    
    return np.concatenate(all_trajectories, axis=0)


def interpolate_in_latent_space(
    model: ConditionalTrajectoryVAE,
    trajectory1: np.ndarray,
    trajectory2: np.ndarray,
    transport_mode1: int,
    transport_mode2: int,
    trip_length1: int,
    trip_length2: int,
    n_steps: int = 10,
    device: str = 'cpu'
) -> np.ndarray:
    """
    Interpolate between two trajectories in latent space.
    
    Args:
        model: Trained VAE model
        trajectory1, trajectory2: Input trajectories
        transport_mode1, transport_mode2: Transport modes
        trip_length1, trip_length2: Trip lengths
        n_steps: Number of interpolation steps
        device: Device to run on
        
    Returns:
        Array of interpolated trajectories
    """
    # Convert to tensors
    traj1 = torch.tensor(trajectory1, dtype=torch.float32).unsqueeze(0).to(device)
    traj2 = torch.tensor(trajectory2, dtype=torch.float32).unsqueeze(0).to(device)
    mode1 = torch.tensor([transport_mode1], dtype=torch.long).to(device)
    mode2 = torch.tensor([transport_mode2], dtype=torch.long).to(device)
    length1 = torch.tensor([trip_length1], dtype=torch.long).to(device)
    length2 = torch.tensor([trip_length2], dtype=torch.long).to(device)
    
    # Create masks (assuming full trajectories)
    mask1 = torch.ones(1, traj1.shape[1], dtype=torch.bool).to(device)
    mask2 = torch.ones(1, traj2.shape[1], dtype=torch.bool).to(device)
    
    with torch.no_grad():
        # Encode both trajectories
        conditions1 = model.get_conditions(mode1, length1)
        conditions2 = model.get_conditions(mode2, length2)
        
        mu1, _ = model.encode(traj1, conditions1, mask1)
        mu2, _ = model.encode(traj2, conditions2, mask2)
        
        # Interpolate in latent space
        interpolated_trajectories = []
        
        for alpha in np.linspace(0, 1, n_steps):
            # Interpolate latent codes
            z_interp = (1 - alpha) * mu1 + alpha * mu2
            
            # Interpolate conditions
            conditions_interp = (1 - alpha) * conditions1 + alpha * conditions2
            
            # Decode
            traj_interp = model.decode(z_interp, conditions_interp)
            interpolated_trajectories.append(traj_interp.cpu().numpy())
    
    return np.concatenate(interpolated_trajectories, axis=0)


def compute_reconstruction_error(
    model: ConditionalTrajectoryVAE,
    trajectories: np.ndarray,
    masks: np.ndarray,
    transport_modes: np.ndarray,
    trip_lengths: np.ndarray,
    device: str = 'cpu',
    batch_size: int = 64
) -> Dict[str, float]:
    """
    Compute reconstruction error on a dataset.
    
    Args:
        model: Trained VAE model
        trajectories: Input trajectories
        masks: Trajectory masks
        transport_modes: Transport mode indices
        trip_lengths: Trip lengths
        device: Device to run on
        batch_size: Batch size
        
    Returns:
        Dictionary with reconstruction metrics
    """
    n_samples = len(trajectories)
    total_recon_error = 0
    total_valid_points = 0
    
    model.eval()
    
    for i in range(0, n_samples, batch_size):
        batch_end = min(i + batch_size, n_samples)
        
        # Get batch
        batch_traj = torch.tensor(trajectories[i:batch_end], dtype=torch.float32).to(device)
        batch_mask = torch.tensor(masks[i:batch_end], dtype=torch.bool).to(device)
        batch_modes = torch.tensor(transport_modes[i:batch_end], dtype=torch.long).to(device)
        batch_lengths = torch.tensor(trip_lengths[i:batch_end], dtype=torch.long).to(device)
        
        with torch.no_grad():
            # Reconstruct
            recon, _, _ = model(batch_traj, batch_modes, batch_lengths, batch_mask)
            
            # Compute masked error
            diff = (recon - batch_traj) ** 2
            mask_expanded = batch_mask.unsqueeze(-1).expand_as(diff)
            masked_diff = diff * mask_expanded
            
            total_recon_error += masked_diff.sum().item()
            total_valid_points += mask_expanded.sum().item()
    
    mean_recon_error = total_recon_error / (total_valid_points + 1e-8)
    
    return {
        'mean_reconstruction_error': mean_recon_error,
        'total_valid_points': total_valid_points,
        'rmse': np.sqrt(mean_recon_error)
    }