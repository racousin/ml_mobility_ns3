import torch
import torch.nn as nn
from typing import Dict, Optional


class DistanceAwareLoss(nn.Module):
    """
    Reconstruction loss that prioritizes real-world distance metrics
    over pixel-perfect coordinate matching.
    """
    
    def __init__(
        self, 
        scaler=None,
        coordinate_weight: float = 0.3,
        point_distance_weight: float = 0.3,
        speed_weight: float = 0.2,
        cumulative_distance_weight: float = 0.2,
        scale_factor: float = 100.0  # km scale for your data
    ):
        super().__init__()
        self.scaler = scaler
        self.coordinate_weight = coordinate_weight
        self.point_distance_weight = point_distance_weight
        self.speed_weight = speed_weight
        self.cumulative_distance_weight = cumulative_distance_weight
        self.scale_factor = scale_factor
        
    def forward(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate distance-aware reconstruction loss.
        
        Args:
            predictions: (batch, seq_len, 2) normalized coordinates
            targets: (batch, seq_len, 2) normalized coordinates
            mask: (batch, seq_len) binary mask for valid points
        """
        # Denormalize if scaler provided
        if self.scaler is not None:
            pred_real = self._denormalize(predictions)
            target_real = self._denormalize(targets)
        else:
            pred_real = predictions * self.scale_factor
            target_real = targets * self.scale_factor
            
        if mask is None:
            mask = torch.ones(predictions.shape[:2], device=predictions.device)
            
        losses = {}
        
        # 1. Standard coordinate MSE (normalized space)
        coord_loss = ((predictions - targets) ** 2).sum(-1)
        losses['coord_mse'] = (coord_loss * mask).sum() / mask.sum()
        
        # 2. Point-to-point distance error (real space)
        point_distances = torch.norm(pred_real - target_real, dim=-1)
        losses['point_distance'] = (point_distances * mask).sum() / mask.sum()
        
        # 3. Speed/velocity matching (consecutive point distances)
        pred_speeds = torch.norm(
            pred_real[:, 1:] - pred_real[:, :-1], dim=-1
        )
        target_speeds = torch.norm(
            target_real[:, 1:] - target_real[:, :-1], dim=-1
        )
        speed_diff = torch.abs(pred_speeds - target_speeds)
        speed_mask = mask[:, 1:] * mask[:, :-1]  # Both points must be valid
        losses['speed_diff'] = (speed_diff * speed_mask).sum() / speed_mask.sum()
        
        # 4. Cumulative trajectory distance
        pred_cumulative = self._compute_trajectory_length(pred_real, mask)
        target_cumulative = self._compute_trajectory_length(target_real, mask)
        losses['trajectory_length_diff'] = torch.abs(
            pred_cumulative - target_cumulative
        ).mean()
        
        # 5. Bird distance (start to end)
        # Find last valid point for each trajectory
        last_indices = (mask.sum(dim=1) - 1).long()
        batch_indices = torch.arange(predictions.shape[0], device=predictions.device)
        
        pred_endpoints = pred_real[batch_indices, last_indices]
        target_endpoints = target_real[batch_indices, last_indices]
        pred_startpoints = pred_real[:, 0]
        target_startpoints = target_real[:, 0]
        
        pred_bird = torch.norm(pred_endpoints - pred_startpoints, dim=-1)
        target_bird = torch.norm(target_endpoints - target_startpoints, dim=-1)
        losses['bird_distance_diff'] = torch.abs(pred_bird - target_bird).mean()
        
        # Combine with weights
        total_loss = (
            self.coordinate_weight * losses['coord_mse'] +
            self.point_distance_weight * losses['point_distance'] / self.scale_factor +
            self.speed_weight * losses['speed_diff'] / self.scale_factor +
            self.cumulative_distance_weight * losses['trajectory_length_diff'] / self.scale_factor
        )
        
        losses['total'] = total_loss
        return losses
    
    def _denormalize(self, coords: torch.Tensor) -> torch.Tensor:
        """Denormalize coordinates using fitted scaler."""
        # Reshape for scaler
        original_shape = coords.shape
        coords_flat = coords.reshape(-1, 2)
        
        # Convert to numpy, denormalize, back to tensor
        coords_np = coords_flat.detach().cpu().numpy()
        coords_denorm = self.scaler.inverse_transform(coords_np)
        coords_tensor = torch.from_numpy(coords_denorm).to(coords.device)
        
        return coords_tensor.reshape(original_shape)
    
    def _compute_trajectory_length(
        self, 
        coords: torch.Tensor, 
        mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute total trajectory length."""
        distances = torch.norm(
            coords[:, 1:] - coords[:, :-1], dim=-1
        )
        valid_distances = distances * mask[:, 1:] * mask[:, :-1]
        return valid_distances.sum(dim=1)


class DistanceVAELoss(nn.Module):
    """
    VAE loss combining distance-aware reconstruction with KL divergence.
    """
    
    def __init__(
        self,
        beta: float = 1.0,
        distance_loss_config: Optional[Dict] = None,
        **kwargs
    ):
        super().__init__()
        self.beta = beta
        config = distance_loss_config or {}
        self.recon_loss = DistanceAwareLoss(**config)
        
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate total VAE loss with distance awareness.
        """
        # Reconstruction losses
        recon_losses = self.recon_loss(predictions, targets, mask)
        
        # KL divergence
        kl_loss = -0.5 * torch.mean(
            1 + logvar - mu.pow(2) - logvar.exp()
        )
        
        # Total VAE loss
        total_loss = recon_losses['total'] + self.beta * kl_loss
        
        return {
            'loss': total_loss,
            'recon_loss': recon_losses['total'],
            'kl_loss': kl_loss,
            **{f'recon_{k}': v for k, v in recon_losses.items() if k != 'total'}
        }