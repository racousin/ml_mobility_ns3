#!/usr/bin/env python
"""Comprehensive script to understand why MSE ~0.02 is easy to achieve."""

import numpy as np
import torch
from pathlib import Path
import pickle
from sklearn.metrics import mean_squared_error

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
    lengths = data.get('lengths', None)
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
    
    # NEW: Analyze why 0.02 MSE is easy
    print("\n" + "="*60)
    print("=== ANALYZING WHY MSE ~0.02 IS EASY TO ACHIEVE ===")
    print("="*60)
    
    # 1. Trivial baselines
    print("\n=== Trivial Baseline MSEs ===")
    
    # Create masks for valid data points
    if lengths is not None:
        masks = np.zeros_like(trajectories[:, :, 0], dtype=bool)
        for i, length in enumerate(lengths):
            masks[i, :length] = True
    else:
        masks = np.ones_like(trajectories[:, :, 0], dtype=bool)
    
    # Baseline 1: Predict mean
    mean_prediction = np.ones_like(trajectories) * trajectories[masks].mean(axis=0)
    mse_mean = ((trajectories - mean_prediction)**2)[masks].mean()
    print(f"MSE (predict global mean): {mse_mean:.6f}")
    
    # Baseline 2: Predict zeros
    zero_prediction = np.zeros_like(trajectories)
    mse_zero = ((trajectories - zero_prediction)**2)[masks].mean()
    print(f"MSE (predict zeros): {mse_zero:.6f}")
    
    # Baseline 3: Predict previous point (persistence model)
    persistence_pred = np.zeros_like(trajectories)
    persistence_pred[:, 1:] = trajectories[:, :-1]
    persistence_pred[:, 0] = trajectories[:, 0]  # First point stays same
    mse_persistence = ((trajectories - persistence_pred)**2)[masks].mean()
    print(f"MSE (predict previous point): {mse_persistence:.6f}")
    
    # Baseline 4: Add small noise to input
    noise_levels = [0.01, 0.05, 0.1]
    for noise in noise_levels:
        noisy_pred = trajectories + np.random.normal(0, noise, trajectories.shape)
        mse_noise = ((trajectories - noisy_pred)**2)[masks].mean()
        print(f"MSE (add {noise} std noise): {mse_noise:.6f}")
    
    # 2. Analyze data smoothness
    print("\n=== Data Smoothness Analysis ===")
    
    # Calculate point-to-point differences
    diffs = trajectories[:, 1:] - trajectories[:, :-1]
    if lengths is not None:
        diff_masks = np.zeros_like(diffs[:, :, 0], dtype=bool)
        for i, length in enumerate(lengths):
            if length > 1:
                diff_masks[i, :length-1] = True
    else:
        diff_masks = np.ones_like(diffs[:, :, 0], dtype=bool)
    
    print(f"Average step size (consecutive points):")
    for i, feature in enumerate(['lat', 'lon', 'speed']):
        avg_diff = np.abs(diffs[:, :, i])[diff_masks].mean()
        std_diff = np.abs(diffs[:, :, i])[diff_masks].std()
        print(f"  {feature}: {avg_diff:.6f} ± {std_diff:.6f}")
    
    # 3. Analyze variance structure
    print("\n=== Variance Analysis ===")
    
    # Total variance
    total_var = trajectories[masks].var()
    print(f"Total variance in data: {total_var:.6f}")
    
    # Variance per feature
    for i, feature in enumerate(['lat', 'lon', 'speed']):
        feat_var = trajectories[:, :, i][masks].var()
        print(f"Variance in {feature}: {feat_var:.6f}")
    
    # Within-trajectory vs between-trajectory variance
    within_vars = []
    for i in range(len(trajectories)):
        if lengths is not None:
            traj_data = trajectories[i, :lengths[i]]
        else:
            traj_data = trajectories[i]
        if len(traj_data) > 1:
            within_vars.append(traj_data.var(axis=0))
    
    within_var = np.mean(within_vars, axis=0)
    between_var = trajectories[masks].mean(axis=0).var()
    
    print(f"\nWithin-trajectory variance: {within_var}")
    print(f"Between-trajectory variance: {between_var:.6f}")
    
    # 4. Check correlation structure
    print("\n=== Temporal Correlation ===")
    
    # Autocorrelation at different lags
    lags = [1, 5, 10, 20]
    for lag in lags:
        correlations = []
        for i in range(len(trajectories)):
            if lengths is not None and lengths[i] > lag:
                traj = trajectories[i, :lengths[i]]
            else:
                traj = trajectories[i]
            
            if len(traj) > lag:
                for feat in range(3):
                    corr = np.corrcoef(traj[:-lag, feat], traj[lag:, feat])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)
        
        if correlations:
            print(f"Average autocorrelation at lag {lag}: {np.mean(correlations):.4f}")
    
    # 5. Check if data is mostly static
    print("\n=== Static Point Analysis ===")
    
    # Check how many points don't change
    static_threshold = 0.001
    for i, feature in enumerate(['lat', 'lon', 'speed']):
        static_points = np.abs(diffs[:, :, i])[diff_masks] < static_threshold
        percent_static = static_points.sum() / diff_masks.sum() * 100
        print(f"{feature} - Points with change < {static_threshold}: {percent_static:.1f}%")
    
    # 6. MSE decomposition
    print("\n=== MSE Decomposition ===")
    
    # What MSE would we get per feature with mean prediction?
    for i, feature in enumerate(['lat', 'lon', 'speed']):
        feat_data = trajectories[:, :, i][masks]
        feat_mean = feat_data.mean()
        feat_mse = ((feat_data - feat_mean)**2).mean()
        print(f"{feature} MSE (predict mean): {feat_mse:.6f}")
        
        # What's the contribution to total MSE?
        print(f"  Contribution to total: {feat_mse/3:.6f}")
    
    # 7. Distribution analysis
    print("\n=== Distribution Analysis ===")
    
    # Check if data is concentrated in small range
    percentiles = [1, 5, 25, 50, 75, 95, 99]
    for i, feature in enumerate(['lat', 'lon', 'speed']):
        feat_data = trajectories[:, :, i][masks]
        percs = np.percentile(feat_data, percentiles)
        print(f"\n{feature} percentiles:")
        for p, v in zip(percentiles, percs):
            print(f"  {p}%: {v:.4f}")
        print(f"  IQR: {percs[4] - percs[2]:.4f}")
        print(f"  90% range: {percs[5] - percs[1]:.4f}")
    
    # 8. Check what your models are likely learning
    print("\n=== What Models Might Be Learning ===")
    
    # Average trajectory shape
    avg_trajectory = trajectories[masks].reshape(-1, 3).mean(axis=0)
    print(f"Average trajectory point: {avg_trajectory}")
    
    # MSE if we always predict the average trajectory
    constant_pred = np.ones_like(trajectories) * avg_trajectory
    mse_constant = ((trajectories - constant_pred)**2)[masks].mean()
    print(f"MSE (predict constant average): {mse_constant:.6f}")
    
    # Compare to your model's MSE
    your_model_mse = 0.0209  # From your results
    print(f"\nYour models achieve: {your_model_mse:.6f}")
    print(f"Improvement over mean baseline: {((mse_mean - your_model_mse) / mse_mean * 100):.1f}%")
    print(f"Improvement over persistence: {((mse_persistence - your_model_mse) / mse_persistence * 100):.1f}%")
    
    # Load and inspect scaler
    scaler_path = Path('data/processed/scalers.pkl')
    if scaler_path.exists():
        print(f"\n=== Scaler Impact ===")
        with open(scaler_path, 'rb') as f:
            scalers = pickle.load(f)
        
        if 'trajectory' in scalers:
            scaler = scalers['trajectory']
            if hasattr(scaler, 'data_range_'):
                print(f"Original data ranges: {scaler.data_range_}")
                print(f"Compression ratio: {1.0 / scaler.data_range_}")
                
                # What's 0.02 MSE in original scale?
                # MSE in scaled = (error_original / range)^2
                # error_original = sqrt(MSE_scaled) * range
                mse_scaled = 0.0209
                error_original = np.sqrt(mse_scaled) * scaler.data_range_
                print(f"\nMSE {mse_scaled:.4f} in scaled space corresponds to:")
                print(f"RMSE in original space: {error_original}")

if __name__ == "__main__":
    inspect_dataset()