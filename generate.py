#!/usr/bin/env python
# generate.py
"""Generate trajectories using a trained Conditional VAE model."""

import argparse
import logging
from pathlib import Path
import numpy as np
import torch
import pickle
import json

# Make sure the project root is in the python path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml_mobility_ns3.models.vae import ConditionalTrajectoryVAE
from ml_mobility_ns3.utils.model_utils import load_model_from_checkpoint

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Generate and un-scale trajectories from a trained VAE")
    parser.add_argument("--model-path", type=Path, required=True, help="Path to the trained model checkpoint (.pt)")
    parser.add_argument("--data-path", type=Path, required=True, help="Path to the original preprocessed VAE dataset (.npz) to get metadata and scalers")
    parser.add_argument("--output-path", type=Path, default="generated_trajectory.npy", help="Path to save the un-scaled generated trajectory")
    parser.add_argument("--mode", type=str, required=True, help="Transport mode to generate (e.g., 'car', 'walk')")
    parser.add_argument("--length", type=int, required=True, help="Desired length of the trajectory in points")
    parser.add_argument("--n-samples", type=int, default=1, help="Number of trajectories to generate")
    parser.add_argument("--device", type=str, default=None, help="Device to use (e.g., 'cuda', 'cpu')")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU ID to use if running on CUDA")

    args = parser.parse_args()

    # Setup device
    if args.device:
        device_name = args.device
    else:
        device_name = 'cuda' if torch.cuda.is_available() else 'cpu'

    if device_name == 'cuda':
        if not torch.cuda.is_available():
            logger.warning("CUDA is not available, falling back to CPU.")
            device = 'cpu'
        elif args.gpu_id >= torch.cuda.device_count():
            logger.warning(
                f"GPU ID {args.gpu_id} is invalid. "
                f"Available GPUs: {torch.cuda.device_count()}. Falling back to GPU 0."
            )
            device = f'cuda:{args.gpu_id}'
        else:
            device = f'cuda:{args.gpu_id}'
    else:
        device = 'cpu'
    logger.info(f"Using device: {device}")

    # --- Load Metadata and Scalers ---
    data_dir = args.data_path.parent
    logger.info(f"Loading metadata and scalers from: {data_dir}")
    try:
        # Load metadata from metadata.pkl
        with open(data_dir / "metadata.pkl", 'rb') as f:
            metadata = pickle.load(f)
        transport_modes_map = {name: i for i, name in enumerate(metadata['transport_modes'])}
        
        # Load scalers from scalers.pkl
        with open(data_dir / "scalers.pkl", 'rb') as f:
            scalers = pickle.load(f)
        # The order of features in the VAE model (lat, lon, speed)
        feature_names = metadata.get('feature_names', ['latitude', 'longitude', 'speed'])

    except FileNotFoundError as e:
        logger.error(f"Could not load metadata or scalers. Please check the path. Error: {e}")
        return

    # --- Load Model ---
    if not args.model_path.exists():
        logger.error(f"Model checkpoint not found at {args.model_path}")
        return

    logger.info(f"Loading model from {args.model_path}...")
    model, config = load_model_from_checkpoint(args.model_path, device)

    # --- Prepare inputs for generation ---
    if args.mode not in transport_modes_map:
        logger.error(f"Mode '{args.mode}' not found. Available modes: {list(transport_modes_map.keys())}")
        return
    
    mode_idx = transport_modes_map[args.mode]
    mode_tensor = torch.tensor([mode_idx] * args.n_samples, dtype=torch.long, device=device)
    length_tensor = torch.tensor([args.length] * args.n_samples, dtype=torch.long, device=device)

    # --- Generate Trajectories ---
    logger.info(f"Generating {args.n_samples} trajectory/trajectories for mode '{args.mode}' with length {args.length}...")
    with torch.no_grad():
        generated_trajectories_scaled = model.generate(mode_tensor, length_tensor, n_samples=args.n_samples, device=device)

    # --- Inverse Transform (Un-scale) the Data ---
    logger.info("Un-scaling generated data to real-world values...")
    results = []
    
    # Get the trajectory scaler
    trajectory_scaler = scalers.get('trajectory')
    if trajectory_scaler is None:
        logger.error("Trajectory scaler not found in scalers.pkl")
        return
    
    for i in range(args.n_samples):
        # Get the valid part of the scaled trajectory
        valid_trajectory_scaled = generated_trajectories_scaled[i, :args.length, :].cpu().numpy()

        # Apply inverse transform
        unscaled_trajectory = trajectory_scaler.inverse_transform(valid_trajectory_scaled)
        
        # Create a structured array with proper field names
        trajectory_data = {
            'latitude': unscaled_trajectory[:, 0],
            'longitude': unscaled_trajectory[:, 1],
            'speed': unscaled_trajectory[:, 2],
            'transport_mode': args.mode,
            'length': args.length
        }
        
        results.append(trajectory_data)
    
    # Save results
    if args.n_samples == 1:
        output_data = results[0]
    else:
        output_data = results

    np.save(args.output_path, output_data, allow_pickle=True)
    logger.info(f"Successfully generated and saved un-scaled trajectory to {args.output_path}")
    
    # Also save some statistics
    stats_path = args.output_path.with_suffix('.stats.json')
    stats = {
        'n_samples': args.n_samples,
        'transport_mode': args.mode,
        'requested_length': args.length,
        'model_path': str(args.model_path),
        'generation_stats': []
    }
    
    for i, traj_data in enumerate(results):
        traj_stats = {
            'sample_id': i,
            'lat_range': [float(np.min(traj_data['latitude'])), float(np.max(traj_data['latitude']))],
            'lon_range': [float(np.min(traj_data['longitude'])), float(np.max(traj_data['longitude']))],
            'speed_stats': {
                'min': float(np.min(traj_data['speed'])),
                'max': float(np.max(traj_data['speed'])),
                'mean': float(np.mean(traj_data['speed'])),
                'std': float(np.std(traj_data['speed']))
            }
        }
        stats['generation_stats'].append(traj_stats)
    
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Saved generation statistics to {stats_path}")

if __name__ == "__main__":
    main()