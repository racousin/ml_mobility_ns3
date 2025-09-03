import numpy as np
import torch
from typing import Dict, Optional, Union
import pickle
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DiffMetrics:
    """Metrics for comparing predicted and target trajectories."""
    
    def __init__(self, scaler_path: Optional[Union[str, Path]] = None):
        """
        Initialize with optional scaler path.
        
        Args:
            scaler_path: Path to scalers.pkl file. If None, looks in default location.
        """
        self.scaler = None
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
    
    def load_scaler(self, scaler_path: Union[str, Path]):
        """Load scaler from pickle file."""
        try:
            with open(scaler_path, 'rb') as f:
                scalers = pickle.load(f)
                self.scaler = scalers.get('trajectory')
                logger.info(f"Loaded scaler from {scaler_path}")
        except Exception as e:
            logger.warning(f"Could not load scaler from {scaler_path}: {e}")
            self.scaler = None
    
    def _ensure_2d_mask(self, mask: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
        """Ensure mask is 2D (batch, seq_len)."""
        if mask.dim() == 1:
            if mask.shape[0] == batch_size:
                new_mask = torch.zeros(batch_size, seq_len, device=mask.device)
                for i in range(batch_size):
                    length = int(mask[i].item())
                    new_mask[i, :length] = 1
                return new_mask
            else:
                raise ValueError(f"1D mask shape {mask.shape} doesn't match batch_size {batch_size}")
        elif mask.dim() == 2:
            return mask
        else:
            raise ValueError(f"Unexpected mask dimensions: {mask.dim()}")
    
    def _inverse_transform_trajectories(self, trajectories: torch.Tensor) -> torch.Tensor:
        """Inverse transform scaled trajectories to original GPS coordinates."""
        if self.scaler is None:
            logger.warning("No scaler loaded, assuming trajectories are already in original space")
            return trajectories
        
        # Convert to numpy for sklearn
        device = trajectories.device
        traj_numpy = trajectories.detach().cpu().numpy()
        original_shape = traj_numpy.shape
        
        # Reshape for scaler
        traj_flat = traj_numpy.reshape(-1, 3)  # 3 features: lat, lon, speed
        
        # Inverse transform
        traj_original = self.scaler.inverse_transform(traj_flat)
        
        # Reshape back and convert to torch
        traj_original = traj_original.reshape(original_shape)
        return torch.from_numpy(traj_original).to(device)
    
    def compute_trajectory_mae(self, pred: torch.Tensor, target: torch.Tensor, 
                              mask: torch.Tensor) -> Dict[str, float]:
        """
        Compute MAE metrics between predicted and target trajectories.
        
        Args:
            pred: Predicted trajectories (batch, seq_len, 3) - scaled
            target: Target trajectories (batch, seq_len, 3) - scaled
            mask: Binary mask (batch, seq_len)
            
        Returns:
            Dictionary of MAE metrics
        """
        batch_size, seq_len, _ = pred.shape
        device = pred.device
        
        # Ensure mask is 2D
        mask = self._ensure_2d_mask(mask, batch_size, seq_len)
        
        # Inverse transform to get original GPS coordinates and speeds
        pred_original = self._inverse_transform_trajectories(pred)
        target_original = self._inverse_transform_trajectories(target)
        
        # Speed MAE - computed from actual consecutive point distances
        # NOT using the 3rd feature which may be informative but not reflect true speed
        # Compute distances between consecutive points
        pred_distances = torch.sqrt(
            (pred_original[:, 1:, 0] - pred_original[:, :-1, 0])**2 + 
            (pred_original[:, 1:, 1] - pred_original[:, :-1, 1])**2 + 1e-8
        ) * 111.0  # Convert degrees to km
        
        target_distances = torch.sqrt(
            (target_original[:, 1:, 0] - target_original[:, :-1, 0])**2 + 
            (target_original[:, 1:, 1] - target_original[:, :-1, 1])**2 + 1e-8
        ) * 111.0  # Convert degrees to km
        
        # Assuming uniform time intervals, speed proportional to distance
        # This measures consistency of point spacing
        speed_mae = torch.abs(pred_distances - target_distances)
        speed_mask = mask[:, 1:] * mask[:, :-1]  # Both points must be valid
        speed_mae = (speed_mae * speed_mask).sum() / (speed_mask.sum() + 1e-8)
        
        # Distance MAE (point-to-point distances)
        lat_pred = pred_original[:, :, 0]
        lon_pred = pred_original[:, :, 1]
        lat_target = target_original[:, :, 0]
        lon_target = target_original[:, :, 1]
        
        # Euclidean distance between corresponding points (in degrees)
        point_distances = torch.sqrt(
            (lat_pred - lat_target)**2 + (lon_pred - lon_target)**2 + 1e-8
        )
        # Convert to km (approximate: 1 degree ≈ 111 km)
        point_distances_km = point_distances * 111.0
        distance_mae = (point_distances_km * mask).sum() / (mask.sum() + 1e-8)
        
        # Total distance MAE
        pred_total_dist = self._compute_total_distance(pred_original, mask)
        target_total_dist = self._compute_total_distance(target_original, mask)
        total_distance_mae = torch.abs(pred_total_dist - target_total_dist).mean()
        
        # Bird distance MAE
        pred_bird_dist = self._compute_bird_distance(pred_original, mask)
        target_bird_dist = self._compute_bird_distance(target_original, mask)
        bird_distance_mae = torch.abs(pred_bird_dist - target_bird_dist).mean()
        
        return {
            'speed_mae': speed_mae.item(),  # Kept for backward compatibility
            'consecutive_distance_mae': speed_mae.item(),  # More accurate name
            'distance_mae': distance_mae.item(),
            'total_distance_mae': total_distance_mae.item(),
            'bird_distance_mae': bird_distance_mae.item()
        }

    def compute_comprehensive_metrics(self, pred: torch.Tensor, target: torch.Tensor, 
                                    mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute comprehensive set of metrics for trajectory comparison.
        
        Args:
            pred: Predicted trajectories (batch, seq_len, 3) - scaled
            target: Target trajectories (batch, seq_len, 3) - scaled
            mask: Binary mask (batch, seq_len)
            
        Returns:
            Dictionary of metrics (as tensors)
        """
        batch_size, seq_len, _ = pred.shape
        device = pred.device
        
        # Ensure mask is 2D
        mask = self._ensure_2d_mask(mask, batch_size, seq_len)
        
        # Basic MSE (in scaled space)
        mask_expanded = mask.unsqueeze(-1).expand_as(pred)
        mse = ((pred - target) ** 2 * mask_expanded).sum() / (mask_expanded.sum() + 1e-8)
        
        # Inverse transform for distance and speed metrics
        pred_original = self._inverse_transform_trajectories(pred)
        target_original = self._inverse_transform_trajectories(target)
        
        # Speed MAE from 3rd feature (informative speed in km/h)
        speed_pred = pred_original[:, :, 2]
        speed_target = target_original[:, :, 2]
        speed_mae = (torch.abs(speed_pred - speed_target) * mask).sum() / (mask.sum() + 1e-8)
        
        # Consecutive distance MAE (actual distance between consecutive points)
        pred_consecutive_dist = torch.sqrt(
            (pred_original[:, 1:, 0] - pred_original[:, :-1, 0])**2 + 
            (pred_original[:, 1:, 1] - pred_original[:, :-1, 1])**2 + 1e-8
        ) * 111.0  # Convert to km
        target_consecutive_dist = torch.sqrt(
            (target_original[:, 1:, 0] - target_original[:, :-1, 0])**2 + 
            (target_original[:, 1:, 1] - target_original[:, :-1, 1])**2 + 1e-8
        ) * 111.0
        consecutive_mask = mask[:, 1:] * mask[:, :-1]
        consecutive_distance_mae = (torch.abs(pred_consecutive_dist - target_consecutive_dist) * consecutive_mask).sum() / (consecutive_mask.sum() + 1e-8)
        
        # Distance MAE (point-to-point)
        lat_pred = pred_original[:, :, 0]
        lon_pred = pred_original[:, :, 1]
        lat_target = target_original[:, :, 0]
        lon_target = target_original[:, :, 1]
        
        point_distances = torch.sqrt(
            (lat_pred - lat_target)**2 + (lon_pred - lon_target)**2 + 1e-8
        ) * 111.0  # Convert to km
        distance_mae = (point_distances * mask).sum() / (mask.sum() + 1e-8)
        
        # Total and bird distance MAEs
        pred_total_dist = self._compute_total_distance(pred_original, mask)
        target_total_dist = self._compute_total_distance(target_original, mask)
        total_distance_mae = torch.abs(pred_total_dist - target_total_dist).mean()
        
        pred_bird_dist = self._compute_bird_distance(pred_original, mask)
        target_bird_dist = self._compute_bird_distance(target_original, mask)
        bird_distance_mae = torch.abs(pred_bird_dist - target_bird_dist).mean()
        
        return {
            'mse': mse,
            'speed_mae': speed_mae,          # Mean Absolute Error for speed from 3rd feature (km/h)
            'consecutive_distance_mae': consecutive_distance_mae,  # Consecutive point distance MAE (km)
            'distance_mae': distance_mae,    # Point-to-point distance MAE (km)
            'total_distance_mae': total_distance_mae,  # Total trajectory distance MAE (km)
            'bird_distance_mae': bird_distance_mae,    # Straight-line distance MAE (km)
        }
    
    def _compute_total_distance(self, trajectories: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute total distance traveled for each trajectory.
        
        Args:
            trajectories: GPS trajectories (batch, seq_len, 3) in original coordinates
            mask: Binary mask (batch, seq_len)
            
        Returns:
            Total distances in km (batch,)
        """
        batch_size, seq_len, _ = trajectories.shape
        device = trajectories.device
        
        if seq_len < 2:
            return torch.zeros(batch_size, device=device)
        
        # Extract coordinates
        lats = trajectories[:, :, 0]
        lons = trajectories[:, :, 1]
        
        # Compute distances between consecutive points
        lat_diffs = torch.diff(lats, dim=1)
        lon_diffs = torch.diff(lons, dim=1)
        segment_distances = torch.sqrt(lat_diffs**2 + lon_diffs**2 + 1e-8) * 111.0  # km
        
        # Create mask for valid segments (both points must be valid)
        segment_mask = mask[:, :-1] * mask[:, 1:]
        
        # Sum valid segments
        total_distances = (segment_distances * segment_mask).sum(dim=1)
        
        return total_distances
    
    def _compute_bird_distance(self, trajectories: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute bird (straight-line) distance for each trajectory.
        
        Args:
            trajectories: GPS trajectories (batch, seq_len, 3) in original coordinates
            mask: Binary mask (batch, seq_len)
            
        Returns:
            Bird distances in km (batch,)
        """
        batch_size = trajectories.shape[0]
        device = trajectories.device
        
        # Find last valid index for each sequence
        valid_lengths = mask.sum(dim=1).long()
        bird_distances = torch.zeros(batch_size, device=device)
        
        for i in range(batch_size):
            if valid_lengths[i] > 1:
                last_idx = valid_lengths[i] - 1
                lat_diff = trajectories[i, last_idx, 0] - trajectories[i, 0, 0]
                lon_diff = trajectories[i, last_idx, 1] - trajectories[i, 0, 1]
                bird_distances[i] = torch.sqrt(lat_diff**2 + lon_diff**2 + 1e-8) * 111.0  # km
        
        return bird_distances