import numpy as np
import torch
from typing import Dict, Optional, Union, Tuple
import pickle
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class StatMetrics:
    """Simplified statistical metrics for trajectory analysis."""
    
    def __init__(self, scaler_path: Optional[Union[str, Path]] = None):
        """
        Initialize with optional scaler path.
        
        Args:
            scaler_path: Path to scalers.pkl file. If None, looks in default location.
        """
        self.scaler = None
        self.category_encoder = None
        
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
            self.load_scalers(scaler_path)
    
    def load_scalers(self, scaler_path: Union[str, Path]):
        """Load scaler and category encoder from pickle file."""
        try:
            with open(scaler_path, 'rb') as f:
                scalers = pickle.load(f)
                self.scaler = scalers.get('trajectory')
                self.category_encoder = scalers.get('category_encoder')
                logger.info(f"Loaded scalers from {scaler_path}")
        except Exception as e:
            logger.warning(f"Could not load scalers from {scaler_path}: {e}")
    
    def compute_category_statistics(self, 
                                  trajectories: Union[np.ndarray, torch.Tensor],
                                  masks: Union[np.ndarray, torch.Tensor],
                                  categories: Union[np.ndarray, torch.Tensor],
                                  lengths: Union[np.ndarray, torch.Tensor]) -> Dict[str, Dict]:
        """
        Compute statistics per transport category.
        
        Args:
            trajectories: Trajectories (n_sequences, seq_len, 3) - scaled
            masks: Binary masks (n_sequences, seq_len)
            categories: Category indices (n_sequences,)
            lengths: Original sequence lengths (n_sequences,)
            
        Returns:
            Dictionary with statistics per category
        """
        # Convert everything to numpy for consistency
        if isinstance(trajectories, torch.Tensor):
            trajectories = trajectories.cpu().numpy()
        if isinstance(masks, torch.Tensor):
            masks = masks.cpu().numpy()
        if isinstance(categories, torch.Tensor):
            categories = categories.cpu().numpy()
        if isinstance(lengths, torch.Tensor):
            lengths = lengths.cpu().numpy()
        
        # Get unique categories
        unique_categories = np.unique(categories)
        category_stats = {}
        
        for cat_idx in unique_categories:
            cat_mask = categories == cat_idx
            
            # Get category name
            if self.category_encoder is not None:
                cat_name = self.category_encoder.classes_[cat_idx]
            else:
                cat_name = f'category_{cat_idx}'
            
            # Extract data for this category
            cat_trajectories = trajectories[cat_mask]
            cat_masks = masks[cat_mask]
            cat_lengths = lengths[cat_mask]
            
            # Compute GPS metrics for this category
            metrics = self._compute_gps_metrics_batch(cat_trajectories, cat_masks)
            
            # Duration in minutes (2 seconds per point)
            durations_min = (cat_lengths * 2) / 60.0
            
            # Compute statistics
            category_stats[cat_name] = {
                'sequences': int(cat_mask.sum()),
                'duration_mean': float(np.mean(durations_min)),
                'duration_std': float(np.std(durations_min)),
                'total_distance_mean': float(np.mean(metrics['total_distances'])),
                'total_distance_std': float(np.std(metrics['total_distances'])),
                'bird_distance_mean': float(np.mean(metrics['bird_distances'])),
                'bird_distance_std': float(np.std(metrics['bird_distances'])),
                'speed_mean': float(np.mean(metrics['avg_speeds'])),
                'speed_std': float(np.std(metrics['avg_speeds'])),
            }
        
        return category_stats
    
    def _compute_gps_metrics_batch(self, trajectories: np.ndarray, 
                                  masks: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute GPS metrics for a batch of trajectories.
        
        Args:
            trajectories: Scaled trajectories (batch, seq_len, 3)
            masks: Binary masks (batch, seq_len)
            
        Returns:
            Dictionary with avg_speeds, bird_distances, total_distances arrays
        """
        batch_size = trajectories.shape[0]
        
        # Inverse transform to get GPS coordinates
        if self.scaler is not None:
            original_shape = trajectories.shape
            traj_flat = trajectories.reshape(-1, 3)
            traj_original = self.scaler.inverse_transform(traj_flat)
            trajectories = traj_original.reshape(original_shape)
        
        # Extract components
        lats = trajectories[:, :, 0]
        lons = trajectories[:, :, 1]
        speeds = trajectories[:, :, 2]
        
        # Initialize results
        avg_speeds = np.zeros(batch_size)
        bird_distances = np.zeros(batch_size)
        total_distances = np.zeros(batch_size)
        
        for i in range(batch_size):
            # Get valid length for this sequence
            valid_mask = masks[i]
            valid_length = valid_mask.sum()
            
            if valid_length > 0:
                # Average speed
                valid_speeds = speeds[i][valid_mask]
                avg_speeds[i] = np.mean(valid_speeds)
                
                if valid_length > 1:
                    # Bird distance (first to last valid point)
                    valid_indices = np.where(valid_mask)[0]
                    first_idx = valid_indices[0]
                    last_idx = valid_indices[-1]
                    
                    lat_diff = lats[i, last_idx] - lats[i, first_idx]
                    lon_diff = lons[i, last_idx] - lons[i, first_idx]
                    bird_distances[i] = np.sqrt(lat_diff**2 + lon_diff**2) * 111.0  # km
                    
                    # Total distance (sum of segments)
                    valid_lats = lats[i][valid_mask]
                    valid_lons = lons[i][valid_mask]
                    
                    lat_diffs = np.diff(valid_lats)
                    lon_diffs = np.diff(valid_lons)
                    segment_distances = np.sqrt(lat_diffs**2 + lon_diffs**2) * 111.0  # km
                    total_distances[i] = np.sum(segment_distances)
        
        return {
            'avg_speeds': avg_speeds,
            'bird_distances': bird_distances,
            'total_distances': total_distances
        }
    
    def print_category_statistics(self, category_stats: Dict[str, Dict], 
                                title: str = "Category Statistics"):
        """
        Print category statistics in formatted way.
        
        Args:
            category_stats: Dictionary with statistics per category
            title: Title to print
        """
        print(f"\n=== {title} ===")
        for cat_name, stats in sorted(category_stats.items()):
            print(f"{cat_name}:")
            print(f"  Sequences: {stats['sequences']:,}")
            print(f"  Duration (min): {stats['duration_mean']:.1f} ± {stats['duration_std']:.1f}")
            print(f"  Total distance (km): {stats['total_distance_mean']:.2f} ± {stats['total_distance_std']:.2f}")
            print(f"  Bird distance (km): {stats['bird_distance_mean']:.2f} ± {stats['bird_distance_std']:.2f}")
            print(f"  Speed (km/h): {stats['speed_mean']:.1f} ± {stats['speed_std']:.1f}")
    
    def compute_gps_metrics_numpy(self, gps_points: np.ndarray, 
                                 scaled: bool = False) -> Tuple[float, float, float]:
        """
        Compute GPS metrics for a single trip (backward compatibility).
        
        Args:
            gps_points: Array with columns [timestamp, lat, lon, speed] or [lat, lon, speed]
            scaled: Whether the input is scaled
            
        Returns:
            tuple: (avg_speed, bird_distance, total_distance) in km/h and km
        """
        if len(gps_points) < 2:
            return 0.0, 0.0, 0.0
        
        # Handle different input formats
        if gps_points.shape[1] == 4:
            coords_speed = gps_points[:, 1:4]
        else:
            coords_speed = gps_points
        
        # Inverse transform if scaled
        if scaled and self.scaler is not None:
            coords_speed = self.scaler.inverse_transform(coords_speed)
        
        # Extract coordinates and speeds
        lats = coords_speed[:, 0].astype(np.float64)
        lons = coords_speed[:, 1].astype(np.float64)
        speeds = coords_speed[:, 2].astype(np.float64)
        
        # Average speed
        avg_speed = np.mean(speeds)
        
        # Bird distance
        lat_diff = lats[-1] - lats[0]
        lon_diff = lons[-1] - lons[0]
        bird_distance = np.sqrt(lat_diff**2 + lon_diff**2) * 111.0  # km
        
        # Total distance
        lat_diffs = np.diff(lats)
        lon_diffs = np.diff(lons)
        segment_distances = np.sqrt(lat_diffs**2 + lon_diffs**2) * 111.0  # km
        total_distance = np.sum(segment_distances)
        
        return avg_speed, bird_distance, total_distance