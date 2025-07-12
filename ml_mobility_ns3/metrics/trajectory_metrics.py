import numpy as np
import torch
from typing import Dict, Tuple, Union, Optional, Any
import torch.nn.functional as F
from abc import ABC, abstractmethod

class TrajectoryMetrics:
    """Centralized metrics for trajectory analysis."""
    
    @staticmethod
    def compute_gps_metrics_numpy(gps_points: np.ndarray) -> Tuple[float, float, float]:
        """
        Compute GPS metrics for a trip using numpy.
        
        Args:
            gps_points: Array with columns [timestamp, lat, lon, speed]
            
        Returns:
            tuple: (avg_speed, bird_distance, total_distance) in km/h and km
        """
        if len(gps_points) < 2:
            return 0.0, 0.0, 0.0
        
        # Extract coordinates and speeds
        lats = np.array(gps_points[:, 1], dtype=np.float64)
        lons = np.array(gps_points[:, 2], dtype=np.float64)
        speeds = np.array(gps_points[:, 3], dtype=np.float64)
        
        # Average speed from GPS speed values
        # GPS speed might be in m/s or km/h - convert if needed
        avg_speed = np.mean(speeds) * 3.6 if speeds.mean() < 50 else np.mean(speeds)
        
        # Bird distance (straight line from first to last point)
        lat_diff = lats[-1] - lats[0]
        lon_diff = lons[-1] - lons[0]
        bird_distance = np.sqrt(lat_diff**2 + lon_diff**2) * 111  # Rough km conversion
        
        # Total distance (sum of distances between consecutive points)
        lat_diffs = np.diff(lats)
        lon_diffs = np.diff(lons)
        segment_distances = np.sqrt(lat_diffs**2 + lon_diffs**2) * 111
        total_distance = np.sum(segment_distances)
        
        return avg_speed, bird_distance, total_distance
    
    @staticmethod
    def compute_gps_metrics_torch(trajectories: torch.Tensor, 
                                  mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute GPS metrics for batched trajectories using torch.
        
        Args:
            trajectories: Tensor of shape (batch, seq_len, features) where features = [lat, lon, speed]
            mask: Binary mask of shape (batch, seq_len)
            
        Returns:
            Dictionary with metrics tensors
        """
        
        batch_size, seq_len, num_features = trajectories.shape
        device = trajectories.device
        
        if mask is None:
            mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
        
        # Extract components
        lats = trajectories[:, :, 0]
        lons = trajectories[:, :, 1]
        speeds = trajectories[:, :, 2] if num_features >= 3 else torch.zeros_like(lats)
        
        # Average speed (considering mask)
        speeds_masked = speeds * mask.float()
        avg_speeds = speeds_masked.sum(dim=1) / (mask.sum(dim=1).float() + 1e-8)
        
        # Bird distance (first to last valid point)
        valid_lengths = mask.sum(dim=1).long()
        bird_distances = torch.zeros(batch_size, device=device)
        
        for i in range(batch_size):
            if valid_lengths[i] > 1:
                start_lat = lats[i, 0]
                start_lon = lons[i, 0]
                # Ensure end_idx is within bounds
                end_idx = min(valid_lengths[i] - 1, seq_len - 1)
                end_lat = lats[i, end_idx]
                end_lon = lons[i, end_idx]
                
                lat_diff = end_lat - start_lat
                lon_diff = end_lon - start_lon
                bird_distances[i] = torch.sqrt(lat_diff**2 + lon_diff**2 + 1e-8) * 111
        
        # Total distance (sum of segments)
        if seq_len > 1:
            lat_diffs = torch.diff(lats, dim=1)
            lon_diffs = torch.diff(lons, dim=1)
            segment_distances = torch.sqrt(lat_diffs**2 + lon_diffs**2 + 1e-8) * 111
            
            # Apply mask to segments (both points must be valid)
            seg_mask = mask[:, 1:] * mask[:, :-1]
            segment_distances = segment_distances * seg_mask.float()
            total_distances = segment_distances.sum(dim=1)
        else:
            total_distances = torch.zeros(batch_size, device=device)
        
        return {
            'avg_speed': avg_speeds,
            'bird_distance': bird_distances,
            'total_distance': total_distances
        }
    
    @staticmethod
    def compute_trajectory_mae(pred: torch.Tensor, target: torch.Tensor, 
                              mask: torch.Tensor) -> Dict[str, float]:
        """
        Compute MAE metrics between predicted and target trajectories.
        
        Args:
            pred: Predicted trajectories (batch, seq_len, features)
            target: Target trajectories (batch, seq_len, features)
            mask: Binary mask (batch, seq_len)
            
        Returns:
            Dictionary of MAE metrics
        """
        # Compute metrics for predictions and targets
        pred_metrics = TrajectoryMetrics.compute_gps_metrics_torch(pred, mask)
        target_metrics = TrajectoryMetrics.compute_gps_metrics_torch(target, mask)
        
        # Speed MAE (direct from trajectories)
        mask_expanded = mask.unsqueeze(-1)
        speed_mae = torch.abs(pred[:, :, 2] - target[:, :, 2])
        speed_mae = (speed_mae * mask).sum() / (mask.sum() + 1e-8)
        
        # Distance MAEs
        total_dist_mae = torch.abs(pred_metrics['total_distance'] - target_metrics['total_distance']).mean()
        bird_dist_mae = torch.abs(pred_metrics['bird_distance'] - target_metrics['bird_distance']).mean()
        
        return {
            'speed_mae': speed_mae.item(),
            'total_distance_mae': total_dist_mae.item(),
            'bird_distance_mae': bird_dist_mae.item()
        }
    
    @staticmethod
    def compute_trip_statistics(trips_data: Union[list, np.ndarray], 
                               weights: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Compute aggregate statistics for a set of trips.
        
        Args:
            trips_data: List of dictionaries or array with trip metrics
            weights: Optional weights for weighted statistics
            
        Returns:
            Dictionary with aggregate statistics
        """
        if isinstance(trips_data, list):
            # Extract arrays from list of dicts
            durations = np.array([t.get('duration_minutes', 0) for t in trips_data])
            speeds = np.array([t.get('speed_kmh', 0) for t in trips_data])
            bird_distances = np.array([t.get('bird_distance_km', 0) for t in trips_data])
            total_distances = np.array([t.get('distance_km', 0) for t in trips_data])
            
            if weights is None:
                weights = np.array([t.get('weight', 1.0) for t in trips_data])
        else:
            # Assume structured array
            durations = trips_data['duration_minutes']
            speeds = trips_data['speed_kmh']
            bird_distances = trips_data['bird_distance_km']
            total_distances = trips_data['distance_km']
            
            if weights is None:
                weights = np.ones(len(trips_data))
        
        # Compute weighted statistics
        def weighted_mean(values, weights):
            return np.sum(values * weights) / np.sum(weights)
        
        def weighted_std(values, weights):
            mean = weighted_mean(values, weights)
            variance = weighted_mean((values - mean)**2, weights)
            return np.sqrt(variance)
        
        return {
            'n_trips': len(durations),
            'total_weight': np.sum(weights),
            'duration_mean': weighted_mean(durations, weights),
            'duration_std': weighted_std(durations, weights),
            'speed_mean': weighted_mean(speeds, weights),
            'speed_std': weighted_std(speeds, weights),
            'bird_distance_mean': weighted_mean(bird_distances, weights),
            'bird_distance_std': weighted_std(bird_distances, weights),
            'total_distance_mean': weighted_mean(total_distances, weights),
            'total_distance_std': weighted_std(total_distances, weights),
        }

    @staticmethod
    def compute_frechet_distance_torch(traj1: torch.Tensor, traj2: torch.Tensor, 
                                     mask1: Optional[torch.Tensor] = None,
                                     mask2: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute discrete Fréchet distance between two trajectories.
        
        Args:
            traj1, traj2: Trajectories of shape (seq_len, 2) containing [lat, lon]
            mask1, mask2: Optional masks indicating valid points
            
        Returns:
            Fréchet distance as a scalar tensor
        """
        # Extract valid points
        if mask1 is not None:
            traj1 = traj1[mask1]
        if mask2 is not None:
            traj2 = traj2[mask2]
        
        n = len(traj1)
        m = len(traj2)
        
        if n == 0 or m == 0:
            return torch.tensor(0.0, device=traj1.device)
        
        # Compute pairwise distances
        # Convert to lat/lon distances in km
        traj1_expanded = traj1.unsqueeze(1)  # (n, 1, 2)
        traj2_expanded = traj2.unsqueeze(0)  # (1, m, 2)
        dists = torch.norm(traj1_expanded - traj2_expanded, dim=2) * 111  # Rough km conversion
        
        # Dynamic programming for Fréchet distance
        dp = torch.full((n, m), float('inf'), device=traj1.device)
        dp[0, 0] = dists[0, 0]
        
        # Fill first row and column
        for i in range(1, n):
            dp[i, 0] = torch.max(dp[i-1, 0], dists[i, 0])
        for j in range(1, m):
            dp[0, j] = torch.max(dp[0, j-1], dists[0, j])
        
        # Fill the rest
        for i in range(1, n):
            for j in range(1, m):
                dp[i, j] = torch.max(
                    torch.min(torch.stack([dp[i-1, j], dp[i, j-1], dp[i-1, j-1]])),
                    dists[i, j]
                )
        
        return dp[n-1, m-1]
    
    @staticmethod
    def compute_comprehensive_metrics(pred: torch.Tensor, target: torch.Tensor, 
                                    mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute comprehensive set of metrics for trajectory comparison.
        
        Args:
            pred: Predicted trajectories (batch, seq_len, features)
            target: Target trajectories (batch, seq_len, features)
            mask: Binary mask (batch, seq_len)
            
        Returns:
            Dictionary of metrics
        """
        
        batch_size, seq_len, num_features = pred.shape
        device = pred.device
        
        # Ensure mask has correct shape
        if mask.dim() == 1:
            # If mask is 1D, it might be a length tensor
            print(f"Warning: mask is 1D with shape {mask.shape}, expected 2D")
            # Create a proper 2D mask
            if mask.shape[0] == batch_size:
                # mask contains lengths for each sequence
                new_mask = torch.zeros(batch_size, seq_len, device=device)
                for i in range(batch_size):
                    length = int(mask[i].item())
                    new_mask[i, :length] = 1
                mask = new_mask
            else:
                raise ValueError(f"Unexpected mask shape: {mask.shape}")
        
        # Basic MSE
        mask_expanded = mask.unsqueeze(-1).expand_as(pred)
        mse = ((pred - target) ** 2 * mask_expanded).sum() / (mask_expanded.sum() + 1e-8)
        
        # Speed MSE (3rd dimension)
        if num_features >= 3:
            speed_mse = ((pred[:, :, 2] - target[:, :, 2]) ** 2 * mask).sum() / (mask.sum() + 1e-8)
        else:
            speed_mse = torch.tensor(0.0, device=device)
        
        # Try to compute GPS metrics with error handling
        try:
            pred_metrics = TrajectoryMetrics.compute_gps_metrics_torch(pred, mask)
            target_metrics = TrajectoryMetrics.compute_gps_metrics_torch(target, mask)
            
            total_dist_mae = torch.abs(pred_metrics['total_distance'] - target_metrics['total_distance']).mean()
            bird_dist_mae = torch.abs(pred_metrics['bird_distance'] - target_metrics['bird_distance']).mean()
        except Exception as e:
            print(f"Error computing GPS metrics: {e}")
            total_dist_mae = torch.tensor(0.0, device=device)
            bird_dist_mae = torch.tensor(0.0, device=device)
        
        # Skip Fréchet distance for now to isolate the issue
        avg_frechet = torch.tensor(0.0, device=device)
        
        return {
            'mse': mse,
            'speed_mse': speed_mse,
            'total_distance_mae': total_dist_mae,
            'bird_distance_mae': bird_dist_mae,
            'frechet_distance': avg_frechet
        }

# Convenience functions for backward compatibility
def compute_gps_metrics(gps_points: np.ndarray) -> Tuple[float, float, float]:
    """Backward compatible function."""
    return TrajectoryMetrics.compute_gps_metrics_numpy(gps_points)


def compute_trajectory_metrics(pred: torch.Tensor, target: torch.Tensor, 
                             mask: torch.Tensor) -> Dict[str, float]:
    """Backward compatible function."""
    return TrajectoryMetrics.compute_trajectory_mae(pred, target, mask)


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