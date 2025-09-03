import torch
import torch.nn as nn
import pickle
import logging
from pathlib import Path
from typing import Dict, Optional, Any, Union
from ml_mobility_ns3.training.losses import BaseLoss, create_beta_scheduler, FreeBits

logger = logging.getLogger(__name__)


class DistanceAwareLoss(nn.Module):
    """
    Reconstruction loss that prioritizes real-world distance metrics
    over pixel-perfect coordinate matching.
    Uses the same inverse transform approach as DiffMetrics for consistency.
    """
    
    def __init__(
        self, 
        scaler_path: Optional[Union[str, Path]] = None,
        coordinate_weight: float = 0.3,
        point_distance_weight: float = 0.3,
        consecutive_distance_weight: float = 0.2,  # Replaces speed_weight
        cumulative_distance_weight: float = 0.2,
    ):
        super().__init__()
        self.scaler = None
        
        # Load scaler using the same approach as DiffMetrics
        if scaler_path is None:
            # Try default paths
            default_paths = [
                Path('data/processed/scalers.pkl'),
                Path('../data/processed/scalers.pkl'),
                Path('../../data/processed/scalers.pkl'),
            ]
            for path in default_paths:
                if path.exists():
                    scaler_path = path
                    break
        
        if scaler_path:
            self.load_scaler(scaler_path)
            
        self.coordinate_weight = coordinate_weight
        self.point_distance_weight = point_distance_weight
        self.consecutive_distance_weight = consecutive_distance_weight
        self.cumulative_distance_weight = cumulative_distance_weight
    
    def load_scaler(self, scaler_path: Union[str, Path]):
        """Load scaler from pickle file (same as DiffMetrics)."""
        try:
            with open(scaler_path, 'rb') as f:
                scalers = pickle.load(f)
                self.scaler = scalers.get('trajectory')
                logger.info(f"DistanceAwareLoss: Loaded scaler from {scaler_path}")
        except Exception as e:
            logger.warning(f"DistanceAwareLoss: Could not load scaler from {scaler_path}: {e}")
            self.scaler = None
        
    def forward(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate distance-aware reconstruction loss.
        
        Args:
            predictions: (batch, seq_len, 3) normalized coordinates with speed
            targets: (batch, seq_len, 3) normalized coordinates with speed
            mask: (batch, seq_len) binary mask for valid points
        """
        # Inverse transform to get real GPS coordinates (lat, lon, speed)
        pred_real = self._inverse_transform_trajectories(predictions)
        target_real = self._inverse_transform_trajectories(targets)
            
        if mask is None:
            mask = torch.ones(predictions.shape[:2], device=predictions.device)
            
        losses = {}
        
        # 1. Standard coordinate MSE (normalized space)
        coord_loss = ((predictions - targets) ** 2).sum(-1)
        losses['coord_mse'] = (coord_loss * mask).sum() / mask.sum()
        
        # 2. Point-to-point distance error (real space in km)
        # Extract lat/lon coordinates
        lat_pred = pred_real[:, :, 0]
        lon_pred = pred_real[:, :, 1]
        lat_target = target_real[:, :, 0]
        lon_target = target_real[:, :, 1]
        
        # Euclidean distance in degrees
        point_distances = torch.sqrt(
            (lat_pred - lat_target)**2 + (lon_pred - lon_target)**2 + 1e-8
        )
        # Convert to km (approximate: 1 degree ≈ 111 km)
        point_distances_km = point_distances * 111.0
        losses['point_distance'] = (point_distances_km * mask).sum() / mask.sum()
        
        # 3. Consecutive distance matching - ensures smooth trajectories
        # Compute distances between consecutive points
        pred_distances = torch.norm(
            pred_real[:, 1:, :2] - pred_real[:, :-1, :2], dim=-1
        ) * 111.0  # Convert degrees to km
        target_distances = torch.norm(
            target_real[:, 1:, :2] - target_real[:, :-1, :2], dim=-1
        ) * 111.0
        
        # This ensures consistent spacing between points
        consecutive_diff = torch.abs(pred_distances - target_distances)
        consecutive_mask = mask[:, 1:] * mask[:, :-1]  # Both points must be valid
        losses['consecutive_distance_diff'] = (consecutive_diff * consecutive_mask).sum() / consecutive_mask.sum()
        
        # 4. Cumulative trajectory distance
        pred_cumulative = self._compute_trajectory_length(pred_real, mask)
        target_cumulative = self._compute_trajectory_length(target_real, mask)
        losses['trajectory_length_diff'] = torch.abs(
            pred_cumulative - target_cumulative
        ).mean()
        
        # 5. Bird distance (start to end) in km
        # Find last valid point for each trajectory
        last_indices = (mask.sum(dim=1) - 1).long()
        batch_indices = torch.arange(predictions.shape[0], device=predictions.device)
        
        pred_endpoints = pred_real[batch_indices, last_indices, :2]  # Only lat/lon
        target_endpoints = target_real[batch_indices, last_indices, :2]
        pred_startpoints = pred_real[:, 0, :2]
        target_startpoints = target_real[:, 0, :2]
        
        pred_bird = torch.norm(pred_endpoints - pred_startpoints, dim=-1) * 111.0  # to km
        target_bird = torch.norm(target_endpoints - target_startpoints, dim=-1) * 111.0
        losses['bird_distance_diff'] = torch.abs(pred_bird - target_bird).mean()
        
        # Combine with weights
        # For walking trajectories with 2-second intervals:
        # - point_distance: target < 1 km (no normalization needed)
        # - consecutive_distance_diff: ~0.001-0.005 km (multiply by 500 to scale)
        # - trajectory_length_diff: target ~1 km (no normalization needed)
        
        # Store weighted components for logging
        losses['weighted_coord'] = self.coordinate_weight * losses['coord_mse']
        losses['weighted_point'] = self.point_distance_weight * losses['point_distance']
        losses['weighted_consecutive'] = self.consecutive_distance_weight * losses['consecutive_distance_diff'] * 500.0
        losses['weighted_cumulative'] = self.cumulative_distance_weight * losses['trajectory_length_diff']
        
        total_loss = (
            losses['weighted_coord'] +
            losses['weighted_point'] +
            losses['weighted_consecutive'] +
            losses['weighted_cumulative']
        )
        
        losses['total'] = total_loss
        return losses
    
    def _inverse_transform_trajectories(self, trajectories: torch.Tensor) -> torch.Tensor:
        """Inverse transform scaled trajectories to original GPS coordinates.
        
        This method is identical to the one in DiffMetrics for consistency.
        """
        if self.scaler is None:
            logger.warning("DistanceAwareLoss: No scaler loaded, assuming trajectories are already in original space")
            return trajectories
        
        # Convert to numpy for sklearn
        device = trajectories.device
        traj_numpy = trajectories.detach().cpu().numpy()
        original_shape = traj_numpy.shape
        
        # Reshape for scaler (assumes 3 features: lat, lon, speed)
        num_features = original_shape[-1]
        traj_flat = traj_numpy.reshape(-1, num_features)
        
        # Inverse transform
        traj_original = self.scaler.inverse_transform(traj_flat)
        
        # Reshape back and convert to torch
        traj_original = traj_original.reshape(original_shape)
        return torch.from_numpy(traj_original).to(device)
    
    def _compute_trajectory_length(
        self, 
        coords: torch.Tensor, 
        mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute total trajectory length in km."""
        # Only use lat/lon coordinates (first 2 features)
        distances = torch.norm(
            coords[:, 1:, :2] - coords[:, :-1, :2], dim=-1
        ) * 111.0  # Convert degrees to km
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
        scaler_path: Optional[Union[str, Path]] = None,
        free_bits: Optional[Dict[str, Any]] = None
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
        # Add scaler_path to config if provided
        if scaler_path is not None:
            config['scaler_path'] = scaler_path
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