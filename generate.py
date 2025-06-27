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
        feature_names = metadata.get('feature_names', ['lat', 'lon', 'speed'])

    except FileNotFoundError as e:
        logger.error(f"Could not load metadata or scalers. Please check the path. Error: {e}")
        return

    # --- Load Model ---
    if not args.model_path.exists():
        logger.error(f"Model checkpoint not found at {args.model_path}")
        return

    logger.info(f"Loading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)
    model_config = checkpoint['config']
    
    model = ConditionalTrajectoryVAE(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

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
    for i in range(args.n_samples):
        # Get the valid part of the scaled trajectory
        valid_trajectory_scaled = generated_trajectories_scaled[i, :args.length, :].cpu().numpy()

        # Create an empty array for the un-scaled data
        unscaled_trajectory = np.zeros_like(valid_trajectory_scaled)
        
        # Apply inverse transform for each feature
        for feature_idx, feature_name in enumerate(feature_names):
            if feature_name in scalers:
                scaler = scalers[feature_name]
                feature_col_scaled = valid_trajectory_scaled[:, feature_idx:feature_idx+1]
                feature_col_unscaled = scaler.inverse_transform(feature_col_scaled)
                unscaled_trajectory[:, feature_idx:feature_idx+1] = feature_col_unscaled
            else:
                logger.warning(f"Scaler for feature '{feature_name}' not found. Leaving it as is.")
                unscaled_trajectory[:, feature_idx:feature_idx+1] = valid_trajectory_scaled[:, feature_idx:feature_idx+1]

        results.append(unscaled_trajectory)
    
    # If only one sample, save directly, otherwise save as a list of arrays (allowing for different lengths in future use cases)
    output_data = results[0] if args.n_samples == 1 else results

    np.save(args.output_path, output_data, allow_pickle=True)
    logger.info(f"Successfully generated and saved un-scaled trajectory to {args.output_path}")

if __name__ == "__main__":
    main()
