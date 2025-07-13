import numpy as np
import torch
from typing import Dict, List, Optional, Union, Tuple
import pickle
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class StatMetrics:
    """Statistical metrics for trajectory analysis."""
    
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
                Path('output/scalers.pkl'),
                Path('../output/scalers.pkl'),
                Path('../../output/scalers.pkl'),
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
    
    def _inverse_transform_numpy(self, trajectories: np.ndarray) -> np.ndarray:
        """Inverse transform scaled trajectories to original GPS coordinates (numpy)."""
        if self.scaler is None:
            logger.warning("No scaler loaded, assuming trajectories are already in original space")
            return trajectories
        
        original_shape = trajectories.shape
        traj_flat = trajectories.reshape(-1, 3)
        traj_original = self.scaler.inverse_transform(traj_flat)
        return traj_original.reshape(original_shape)
    
    def _inverse_transform_torch(self, trajectories: torch.Tensor) -> torch.Tensor:
        """Inverse transform scaled trajectories to original GPS coordinates (torch)."""
        if self.scaler is None:
            logger.warning("No scaler loaded, assuming trajectories are already in original space")
            return trajectories
        
        device = trajectories.device
        traj_numpy = trajectories.detach().cpu().numpy()
        traj_original = self._inverse_transform_numpy(traj_numpy)
        return torch.from_numpy(traj_original).to(device)
    
    def compute_gps_metrics_numpy(self, gps_points: np.ndarray, scaled: bool = False) -> Tuple[float, float, float]:
        """
        Compute GPS metrics for a trip using numpy.
        
        Args:
            gps_points: Array with columns [timestamp, lat, lon, speed] or [lat, lon, speed]
            scaled: Whether the input is already scaled (if True, will inverse transform)
            
        Returns:
            tuple: (avg_speed, bird_distance, total_distance) in km/h and km
        """
        if len(gps_points) < 2:
            return 0.0, 0.0, 0.0
        
        # Handle different input formats
        if gps_points.shape[1] == 4:
            # Has timestamp column
            coords_speed = gps_points[:, 1:4]
        else:
            # Just lat, lon, speed
            coords_speed = gps_points
        
        # Inverse transform if scaled
        if scaled and self.scaler is not None:
            coords_speed = self._inverse_transform_numpy(coords_speed)
        
        # Extract coordinates and speeds
        lats = coords_speed[:, 0].astype(np.float64)
        lons = coords_speed[:, 1].astype(np.float64)
        speeds = coords_speed[:, 2].astype(np.float64)
        
        # Average speed (already in km/h after inverse transform)
        avg_speed = np.mean(speeds)
        
        # Bird distance (straight line from first to last point)
        lat_diff = lats[-1] - lats[0]
        lon_diff = lons[-1] - lons[0]
        bird_distance = np.sqrt(lat_diff**2 + lon_diff**2) * 111.0  # km
        
        # Total distance (sum of distances between consecutive points)
        lat_diffs = np.diff(lats)
        lon_diffs = np.diff(lons)
        segment_distances = np.sqrt(lat_diffs**2 + lon_diffs**2) * 111.0  # km
        total_distance = np.sum(segment_distances)
        
        return avg_speed, bird_distance, total_distance
    
    def compute_gps_metrics_torch(self, trajectories: torch.Tensor, 
                                mask: Optional[torch.Tensor] = None,
                                scaled: bool = True) -> Dict[str, torch.Tensor]:
        """
        Compute GPS metrics for batched trajectories.
        
        Args:
            trajectories: Trajectories (batch, seq_len, 3) with [lat, lon, speed]
            mask: Optional binary mask (batch, seq_len)
            scaled: Whether input is scaled (default True for torch tensors)
            
        Returns:
            Dictionary with avg_speed, bird_distance, total_distance tensors
        """
        batch_size, seq_len, _ = trajectories.shape
        device = trajectories.device
        
        if mask is None:
            mask = torch.ones(batch_size, seq_len, device=device)
        
        # Inverse transform if scaled
        if scaled:
            trajectories = self._inverse_transform_torch(trajectories)
        
        # Extract components
        lats = trajectories[:, :, 0]
        lons = trajectories[:, :, 1]
        speeds = trajectories[:, :, 2]
        
        # Average speed
        speeds_masked = speeds * mask
        avg_speeds = speeds_masked.sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        
        # Bird distance
        valid_lengths = mask.sum(dim=1).long()
        bird_distances = torch.zeros(batch_size, device=device)
        
        for i in range(batch_size):
            if valid_lengths[i] > 1:
                last_idx = valid_lengths[i] - 1
                lat_diff = lats[i, last_idx] - lats[i, 0]
                lon_diff = lons[i, last_idx] - lons[i, 0]
                bird_distances[i] = torch.sqrt(lat_diff**2 + lon_diff**2 + 1e-8) * 111.0  # km
        
        # Total distance
        total_distances = torch.zeros(batch_size, device=device)
        if seq_len > 1:
            for i in range(batch_size):
                if valid_lengths[i] > 1:
                    valid_len = valid_lengths[i]
                    lat_seq = lats[i, :valid_len]
                    lon_seq = lons[i, :valid_len]
                    
                    lat_diffs = torch.diff(lat_seq)
                    lon_diffs = torch.diff(lon_seq)
                    segment_distances = torch.sqrt(lat_diffs**2 + lon_diffs**2 + 1e-8) * 111.0  # km
                    total_distances[i] = segment_distances.sum()
        
        return {
            'avg_speed': avg_speeds,
            'bird_distance': bird_distances,
            'total_distance': total_distances
        }
    
    def compute_trip_statistics(self, trips_data: Union[List[Dict], np.ndarray], 
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
        
        # Normalize weights
        weights = weights / weights.sum()
        
        # Compute weighted statistics
        def weighted_mean(values, weights):
            return np.sum(values * weights)
        
        def weighted_std(values, weights):
            mean = weighted_mean(values, weights)
            variance = np.sum(weights * (values - mean)**2)
            return np.sqrt(variance)
        
        def weighted_percentile(values, weights, percentile):
            """Compute weighted percentile."""
            sorted_indices = np.argsort(values)
            sorted_values = values[sorted_indices]
            sorted_weights = weights[sorted_indices]
            cumsum = np.cumsum(sorted_weights)
            cutoff = percentile / 100.0
            return sorted_values[np.searchsorted(cumsum, cutoff)]
        
        return {
            'n_trips': len(durations),
            'total_weight': weights.sum() * len(weights),  # Unnormalize for display
            
            # Duration statistics
            'duration_mean': weighted_mean(durations, weights),
            'duration_std': weighted_std(durations, weights),
            'duration_median': weighted_percentile(durations, weights, 50),
            'duration_p25': weighted_percentile(durations, weights, 25),
            'duration_p75': weighted_percentile(durations, weights, 75),
            
            # Speed statistics
            'speed_mean': weighted_mean(speeds, weights),
            'speed_std': weighted_std(speeds, weights),
            'speed_median': weighted_percentile(speeds, weights, 50),
            'speed_p25': weighted_percentile(speeds, weights, 25),
            'speed_p75': weighted_percentile(speeds, weights, 75),
            
            # Bird distance statistics
            'bird_distance_mean': weighted_mean(bird_distances, weights),
            'bird_distance_std': weighted_std(bird_distances, weights),
            'bird_distance_median': weighted_percentile(bird_distances, weights, 50),
            
            # Total distance statistics
            'total_distance_mean': weighted_mean(total_distances, weights),
            'total_distance_std': weighted_std(total_distances, weights),
            'total_distance_median': weighted_percentile(total_distances, weights, 50),
            
            # Ratios
            'distance_ratio_mean': weighted_mean(
                total_distances / (bird_distances + 1e-8), weights
            ),
        }
    
    def compute_batch_statistics(self, trajectories: torch.Tensor, 
                               mask: torch.Tensor,
                               categories: Optional[torch.Tensor] = None,
                               scaled: bool = True) -> Dict[str, Union[float, Dict[str, float]]]:
        """
        Compute statistics for a batch of trajectories.
        
        Args:
            trajectories: Batch of trajectories (batch, seq_len, 3)
            mask: Binary mask (batch, seq_len)
            categories: Optional category labels (batch,)
            scaled: Whether trajectories are scaled
            
        Returns:
            Dictionary with overall and per-category statistics
        """
        metrics = self.compute_gps_metrics_torch(trajectories, mask, scaled)
        
        # Overall statistics
        stats = {
            'batch_size': trajectories.shape[0],
            'avg_speed_mean': metrics['avg_speed'].mean().item(),
            'avg_speed_std': metrics['avg_speed'].std().item(),
            'bird_distance_mean': metrics['bird_distance'].mean().item(),
            'bird_distance_std': metrics['bird_distance'].std().item(),
            'total_distance_mean': metrics['total_distance'].mean().item(),
            'total_distance_std': metrics['total_distance'].std().item(),
        }
        
        # Per-category statistics if categories provided
        if categories is not None:
            unique_categories = torch.unique(categories)
            category_stats = {}
            
            for cat in unique_categories:
                cat_mask = categories == cat
                cat_metrics = {
                    'count': cat_mask.sum().item(),
                    'avg_speed_mean': metrics['avg_speed'][cat_mask].mean().item(),
                    'bird_distance_mean': metrics['bird_distance'][cat_mask].mean().item(),
                    'total_distance_mean': metrics['total_distance'][cat_mask].mean().item(),
                }
                category_stats[f'category_{cat.item()}'] = cat_metrics
            
            stats['per_category'] = category_stats
        
        return stats