import numpy as np
import torch
from typing import Dict, Tuple, Union, Optional


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
        batch_size, seq_len, _ = trajectories.shape
        device = trajectories.device
        
        if mask is None:
            mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
        
        # Extract components
        lats = trajectories[:, :, 0]
        lons = trajectories[:, :, 1]
        speeds = trajectories[:, :, 2]
        
        # Average speed (considering mask)
        speeds_masked = speeds * mask
        avg_speeds = speeds_masked.sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        
        # Bird distance (first to last valid point)
        valid_lengths = mask.sum(dim=1).long()
        bird_distances = torch.zeros(batch_size, device=device)
        
        for i in range(batch_size):
            if valid_lengths[i] > 1:
                start_lat = lats[i, 0]
                start_lon = lons[i, 0]
                end_idx = valid_lengths[i] - 1
                end_lat = lats[i, end_idx]
                end_lon = lons[i, end_idx]
                
                lat_diff = end_lat - start_lat
                lon_diff = end_lon - start_lon
                bird_distances[i] = torch.sqrt(lat_diff**2 + lon_diff**2 + 1e-8) * 111
        
        # Total distance (sum of segments)
        lat_diffs = torch.diff(lats, dim=1)
        lon_diffs = torch.diff(lons, dim=1)
        segment_distances = torch.sqrt(lat_diffs**2 + lon_diffs**2 + 1e-8) * 111
        
        # Apply mask to segments (both points must be valid)
        seg_mask = mask[:, 1:] * mask[:, :-1]
        segment_distances = segment_distances * seg_mask
        total_distances = segment_distances.sum(dim=1)
        
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


# Convenience functions for backward compatibility
def compute_gps_metrics(gps_points: np.ndarray) -> Tuple[float, float, float]:
    """Backward compatible function."""
    return TrajectoryMetrics.compute_gps_metrics_numpy(gps_points)


def compute_trajectory_metrics(pred: torch.Tensor, target: torch.Tensor, 
                             mask: torch.Tensor) -> Dict[str, float]:
    """Backward compatible function."""
    return TrajectoryMetrics.compute_trajectory_mae(pred, target, mask)