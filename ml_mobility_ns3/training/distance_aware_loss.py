import torch
import torch.nn as nn
from typing import Dict, Optional, Any, Union
from ml_mobility_ns3.training.losses import BaseLoss, create_beta_scheduler, FreeBits


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
        # Denormalize coordinates to real space
        if self.scaler is not None:
            pred_real = self._denormalize(predictions)
            target_real = self._denormalize(targets)
        else:
            # Simple scaling if no scaler provided
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


class DistanceVAELoss(BaseLoss):
    """
    VAE loss combining distance-aware reconstruction with KL divergence.
    Inherits from BaseLoss to integrate with the training framework.
    """
    
    def __init__(
        self,
        beta: Optional[Union[float, Dict[str, Any]]] = None,
        distance_loss_config: Optional[Dict] = None,
        free_bits: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__()
        
        # Setup beta scheduler
        if beta is None:
            self.beta_scheduler = create_beta_scheduler({'type': 'constant', 'value': 1.0})
        elif isinstance(beta, (int, float)):
            self.beta_scheduler = create_beta_scheduler({'type': 'constant', 'value': float(beta)})
        else:
            self.beta_scheduler = create_beta_scheduler(beta)
            
        # Setup free bits if configured
        self.free_bits = None
        if free_bits and free_bits.get('enabled', False):
            self.free_bits = FreeBits(free_bits.get('lambda_free_bits', 2.0))
        
        # Setup distance-aware reconstruction loss
        config = distance_loss_config or {}
        self.recon_loss = DistanceAwareLoss(**config)
        
        # Store latent dim for free bits (will be set on first call)
        self.latent_dim = None
        
        # Track if we're using adaptive scheduler
        from ml_mobility_ns3.training.losses import AdaptiveSlowAnnealingBeta
        self.is_adaptive = isinstance(self.beta_scheduler, AdaptiveSlowAnnealingBeta)
        
    def __call__(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate VAE loss with distance-aware reconstruction.
        
        Args:
            outputs: Dict with 'recon', 'mu', 'logvar' tensors
            targets: Dict with 'x' tensor (trajectories)
            mask: Binary mask for valid points
        """
        predictions = outputs['recon']
        target_traj = targets['x']
        mu = outputs['mu']
        logvar = outputs['logvar']
        
        # Calculate reconstruction losses
        try:
            recon_losses = self.recon_loss(predictions, target_traj, mask)
            if 'total' not in recon_losses:
                raise KeyError(f"'total' key missing from recon_losses. Keys: {list(recon_losses.keys())}")
        except Exception as e:
            raise RuntimeError(f"Error computing reconstruction loss: {e}")
        
        # Calculate KL divergence
        kl_loss = -0.5 * torch.mean(
            1 + logvar - mu.pow(2) - logvar.exp()
        )
        
        # Apply free bits if configured
        if self.free_bits is not None:
            if self.latent_dim is None:
                self.latent_dim = mu.shape[-1]
            kl_loss = self.free_bits.apply(kl_loss, self.latent_dim)
        
        # Get current beta
        beta = self.beta_scheduler.get_beta(self.current_step, self.current_epoch)
        
        # Total VAE loss
        total_loss = recon_losses['total'] + beta * kl_loss
        
        result = {
            'total': total_loss,  # Key must be 'total' for lightning module
            'recon_loss': recon_losses['total'],  # Use 'recon_loss' for consistency
            'kl_loss': kl_loss,
            'weighted_kl_loss': beta * kl_loss,
            'beta': beta,
            **{f'recon_{k}': v for k, v in recon_losses.items() if k != 'total'}
        }
        
        # Add scheduler status if adaptive
        if self.is_adaptive:
            status = self.beta_scheduler.get_status()
            result['epochs_without_improvement'] = status['epochs_without_improvement']
            result['scheduler_converged'] = status['converged']
            
        return result
    
    def update_adaptive_scheduler_epoch(self, epoch: int, loss: float):
        """Update adaptive scheduler with epoch-level loss if applicable."""
        if self.is_adaptive:
            self.beta_scheduler.update_epoch_loss(epoch, loss)
    
    def get_scheduler_status(self) -> Optional[Dict[str, Any]]:
        """Get scheduler status if adaptive."""
        if self.is_adaptive:
            return self.beta_scheduler.get_status()
        return None