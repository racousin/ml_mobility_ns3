import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class BaseLoss(ABC):
    """Base class for all loss functions."""
    
    @abstractmethod
    def __call__(self, outputs: Dict[str, torch.Tensor], 
                 targets: Dict[str, torch.Tensor], 
                 mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute loss.
        
        Returns:
            Dict with 'total' key and component losses
        """
        pass


class SimpleVAELoss(BaseLoss):
    """Simple VAE loss: reconstruction + beta * KL divergence."""
    
    def __init__(self, beta: float = 1.0):
        self.beta = beta
    
    def __call__(self, outputs: Dict[str, torch.Tensor], 
                 targets: Dict[str, torch.Tensor], 
                 mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        
        recon = outputs['recon']
        x = targets['x']
        mu = outputs['mu']
        logvar = outputs['logvar']

        # Ensure numerical stability
        logvar = torch.clamp(logvar, min=-10, max=10)
        
        # Reconstruction loss (MSE)
        mask_expanded = mask.unsqueeze(-1).expand_as(x)
        diff = (recon - x) ** 2
        masked_diff = diff * mask_expanded
        num_valid = mask_expanded.sum()
        recon_loss = masked_diff.sum() / (num_valid + 1e-8)
        
        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss
        total = recon_loss + self.beta * kl_loss
        
        return {
            'total': total,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'weighted_kl_loss': self.beta * kl_loss
        }


class DistanceAwareVAELoss(BaseLoss):
    """VAE loss with additional distance preservation term."""
    
    def __init__(self, beta: float = 1.0, lambda_dist: float = 0.5):
        self.beta = beta
        self.lambda_dist = lambda_dist
    
    def __call__(self, outputs: Dict[str, torch.Tensor], 
                 targets: Dict[str, torch.Tensor], 
                 mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        
        # First compute simple VAE loss components
        simple_loss = SimpleVAELoss(self.beta)(outputs, targets, mask)
        
        # Add distance preservation loss
        recon = outputs['recon']
        x = targets['x']
        
        # Compute pairwise distances for sequences
        # Using a simple approach: distance between consecutive points
        def compute_sequence_distances(seq, mask):
            # seq: (batch, seq_len, 3) - [lat, lon, speed]
            # Only use lat/lon for distance
            coords = seq[:, :, :2]  # (batch, seq_len, 2)
            
            # Compute distances between consecutive points
            diff = coords[:, 1:] - coords[:, :-1]  # (batch, seq_len-1, 2)
            distances = torch.norm(diff, dim=-1)  # (batch, seq_len-1)
            
            # Apply mask for consecutive point pairs
            pair_mask = mask[:, 1:] * mask[:, :-1]
            masked_distances = distances * pair_mask
            
            return masked_distances.sum(dim=1) / (pair_mask.sum(dim=1) + 1e-8)
        
        pred_distances = compute_sequence_distances(recon, mask)
        target_distances = compute_sequence_distances(x, mask)
        
        distance_loss = F.mse_loss(pred_distances, target_distances)
        
        # Update total loss
        total = simple_loss['total'] + self.lambda_dist * distance_loss
        
        return {
            'total': total,
            'recon_loss': simple_loss['recon_loss'],
            'kl_loss': simple_loss['kl_loss'],
            'weighted_kl_loss': simple_loss['weighted_kl_loss'],
            'distance_loss': distance_loss,
            'weighted_distance_loss': self.lambda_dist * distance_loss
        }


class SpeedAwareVAELoss(BaseLoss):
    """VAE loss with additional speed consistency term."""
    
    def __init__(self, beta: float = 1.0, lambda_speed: float = 0.3):
        self.beta = beta
        self.lambda_speed = lambda_speed
    
    def __call__(self, outputs: Dict[str, torch.Tensor], 
                 targets: Dict[str, torch.Tensor], 
                 mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        
        # First compute simple VAE loss components
        simple_loss = SimpleVAELoss(self.beta)(outputs, targets, mask)
        
        # Add speed consistency loss
        recon = outputs['recon']
        x = targets['x']
        
        # Extract speeds (3rd dimension)
        pred_speeds = recon[:, :, 2]
        target_speeds = x[:, :, 2]
        
        # Apply mask and compute speed loss
        speed_diff = (pred_speeds - target_speeds) ** 2
        masked_speed_diff = speed_diff * mask
        speed_loss = masked_speed_diff.sum() / (mask.sum() + 1e-8)
        
        # Also compute speed smoothness (variation between consecutive speeds)
        def compute_speed_smoothness(speeds, mask):
            speed_diff = torch.abs(speeds[:, 1:] - speeds[:, :-1])
            pair_mask = mask[:, 1:] * mask[:, :-1]
            return (speed_diff * pair_mask).sum() / (pair_mask.sum() + 1e-8)
        
        pred_smoothness = compute_speed_smoothness(pred_speeds, mask)
        target_smoothness = compute_speed_smoothness(target_speeds, mask)
        smoothness_loss = torch.abs(pred_smoothness - target_smoothness)
        
        # Total speed-aware loss
        speed_aware_loss = speed_loss + 0.1 * smoothness_loss
        
        # Update total loss
        total = simple_loss['total'] + self.lambda_speed * speed_aware_loss
        
        return {
            'total': total,
            'recon_loss': simple_loss['recon_loss'],
            'kl_loss': simple_loss['kl_loss'],
            'weighted_kl_loss': simple_loss['weighted_kl_loss'],
            'speed_loss': speed_loss,
            'smoothness_loss': smoothness_loss,
            'weighted_speed_loss': self.lambda_speed * speed_aware_loss
        }


# Loss factory
LOSS_REGISTRY = {
    'simple_vae': SimpleVAELoss,
    'distance_aware_vae': DistanceAwareVAELoss,
    'speed_aware_vae': SpeedAwareVAELoss
}


def create_loss(config: Dict[str, Any]) -> BaseLoss:
    """Create loss function from config."""
    loss_type = config.get('type', 'simple_vae')
    loss_params = config.get('params', {})
    
    if loss_type not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss type: {loss_type}. Available: {list(LOSS_REGISTRY.keys())}")
    
    return LOSS_REGISTRY[loss_type](**loss_params)