#!/usr/bin/env python
"""Minimal script to inspect the scaled dataset."""

import numpy as np
import torch
from pathlib import Path
import pickle

def inspect_dataset():
    """Inspect the processed dataset and scaler."""
    
    # Load dataset
    dataset_path = Path('data/processed/dataset.npz')
    if not dataset_path.exists():
        print(f"Dataset not found at {dataset_path}")
        return
    
    data = np.load(dataset_path)
    print("=== Dataset Keys ===")
    print(list(data.keys()))
    
    # Inspect trajectories
    trajectories = data['trajectories']
    print(f"\n=== Trajectories Shape ===")
    print(f"Shape: {trajectories.shape}")
    print(f"Dtype: {trajectories.dtype}")
    
    print(f"\n=== Scaled Data Statistics ===")
    print(f"Min: {trajectories.min():.6f}")
    print(f"Max: {trajectories.max():.6f}")
    print(f"Mean: {trajectories.mean():.6f}")
    print(f"Std: {trajectories.std():.6f}")
    
    # Check each feature separately
    print(f"\n=== Per-Feature Statistics (scaled) ===")
    for i, feature in enumerate(['lat', 'lon', 'speed']):
        feat_data = trajectories[:, :, i]
        print(f"{feature}:")
        print(f"  Min: {feat_data.min():.6f}")
        print(f"  Max: {feat_data.max():.6f}")
        print(f"  Mean: {feat_data.mean():.6f}")
        print(f"  Std: {feat_data.std():.6f}")
    
    # Check for boundary values
    print(f"\n=== Boundary Analysis ===")
    at_zero = (trajectories == 0.0).sum()
    at_one = (trajectories == 1.0).sum()
    below_zero = (trajectories < 0.0).sum()
    above_one = (trajectories > 1.0).sum()
    
    print(f"Values exactly 0.0: {at_zero}")
    print(f"Values exactly 1.0: {at_one}")
    print(f"Values < 0.0: {below_zero}")
    print(f"Values > 1.0: {above_one}")
    
    # Sample trajectory inspection
    print(f"\n=== Sample Trajectory ===")
    sample_idx = 0
    sample_traj = trajectories[sample_idx]
    lengths = data.get('lengths', None)
    if lengths is not None:
        actual_length = lengths[sample_idx]
        print(f"Trajectory {sample_idx} length: {actual_length}")
        print(f"First 5 points:")
        for i in range(min(5, actual_length)):
            print(f"  Point {i}: lat={sample_traj[i,0]:.4f}, lon={sample_traj[i,1]:.4f}, speed={sample_traj[i,2]:.4f}")
    
    # Load and inspect scaler
    scaler_path = Path('data/processed/scalers.pkl')
    if scaler_path.exists():
        print(f"\n=== Scaler Information ===")
        with open(scaler_path, 'rb') as f:
            scalers = pickle.load(f)
        
        if 'trajectory' in scalers:
            scaler = scalers['trajectory']
            print(f"Scaler type: {type(scaler).__name__}")
            if hasattr(scaler, 'data_min_'):
                print(f"Original min values: {scaler.data_min_}")
            if hasattr(scaler, 'data_max_'):
                print(f"Original max values: {scaler.data_max_}")
            if hasattr(scaler, 'data_range_'):
                print(f"Original ranges: {scaler.data_range_}")
    
    # Test inverse transform on a sample
    print(f"\n=== Inverse Transform Test ===")
    if 'scalers' in locals():
        scaler = scalers.get('trajectory')
        if scaler is not None:
            # Take first few points of first trajectory
            sample_scaled = trajectories[0, :3, :]  # First 3 points
            sample_original = scaler.inverse_transform(sample_scaled)
            
            print("Scaled -> Original transform:")
            for i in range(3):
                print(f"  Point {i}:")
                print(f"    Scaled: lat={sample_scaled[i,0]:.4f}, lon={sample_scaled[i,1]:.4f}, speed={sample_scaled[i,2]:.4f}")
                print(f"    Original: lat={sample_original[i,0]:.4f}, lon={sample_original[i,1]:.4f}, speed={sample_original[i,2]:.4f}")

def check_model_outputs():
    """Quick check of what the model outputs look like."""
    print("\n" + "="*50)
    print("=== MODEL OUTPUT INSPECTION ===")
    
    # Load a trained model checkpoint if available
    import sys
    from pathlib import Path
    sys.path.append(str(Path('.').absolute()))
    
    try:
        from ml_mobility_ns3.data.dataset import TrajectoryDataset
        from torch.utils.data import DataLoader
        
        # Load dataset
        dataset = TrajectoryDataset('data/processed/dataset.npz')
        loader = DataLoader(dataset, batch_size=4, shuffle=False)
        
        # Get a sample batch
        batch = next(iter(loader))
        x, mask, transport_mode, length = batch
        
        print(f"Input batch shape: {x.shape}")
        print(f"Input range: [{x.min():.4f}, {x.max():.4f}]")
        print(f"Mask shape: {mask.shape}")
        
        # Show sample input values
        print(f"\nSample input values (first trajectory, first 3 points):")
        for i in range(3):
            print(f"  Point {i}: [{x[0,i,0]:.4f}, {x[0,i,1]:.4f}, {x[0,i,2]:.4f}]")
        
    except Exception as e:
        print(f"Could not load model/data: {e}")

if __name__ == "__main__":
    inspect_dataset()
    check_model_outputs()