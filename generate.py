#!/usr/bin/env python
# generate.py
"""Generate trajectories using a trained Conditional VAE model."""

import argparse
import logging
from pathlib import Path
import numpy as np
import torch
import json

# Make sure the project root is in the python path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml_mobility_ns3.models.vae import ConditionalTrajectoryVAE

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Generate trajectories from a trained VAE")
    parser.add_argument("--model-path", type=Path, required=True, help="Path to the trained model checkpoint (.pt)")
    parser.add_argument("--data-path", type=Path, required=True, help="Path to the original preprocessed VAE dataset (.npz) to get metadata")
    parser.add_argument("--output-path", type=Path, default="generated_trajectory.npy", help="Path to save the generated trajectory")
    parser.add_argument("--mode", type=str, required=True, help="Transport mode to generate (e.g., 'car', 'walk')")
    parser.add_argument("--length", type=int, required=True, help="Desired length of the trajectory in points")
    parser.add_argument("--n-samples", type=int, default=1, help="Number of trajectories to generate")
    parser.add_argument("--device", type=str, default=None, help="Device to use (e.g., 'cuda', 'cpu')")

    args = parser.parse_args()

    # Setup device
    if args.device:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # Load metadata
    logger.info(f"Loading metadata from {args.data_path}...")
    try:
        data = np.load(args.data_path)
        metadata = data['metadata'].item()
        transport_modes_map = {name: i for i, name in enumerate(metadata['transport_modes'])}
    except (FileNotFoundError, KeyError):
        logger.error("Could not load metadata. Please provide the correct path to the .npz dataset.")
        return

    # Load model checkpoint
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

    # Prepare inputs for generation
    if args.mode not in transport_modes_map:
        logger.error(f"Mode '{args.mode}' not found. Available modes: {list(transport_modes_map.keys())}")
        return
    
    mode_idx = transport_modes_map[args.mode]
    
    # Create tensors for generation
    mode_tensor = torch.tensor([mode_idx] * args.n_samples, dtype=torch.long, device=device)
    length_tensor = torch.tensor([args.length] * args.n_samples, dtype=torch.long, device=device)

    # Generate trajectory
    logger.info(f"Generating {args.n_samples} trajectory/trajectories for mode '{args.mode}' with length {args.length}...")
    with torch.no_grad():
        generated_trajectories = model.generate(mode_tensor, length_tensor, n_samples=args.n_samples, device=device)

    # Process and save the output
    # The output from the model is (n_samples, seq_len, features)
    # We only want the valid part based on the specified length
    results = []
    for i in range(args.n_samples):
        valid_trajectory = generated_trajectories[i, :args.length, :].cpu().numpy()
        results.append(valid_trajectory)
    
    # If only one sample, save directly, otherwise save as a list of arrays
    output_data = results[0] if args.n_samples == 1 else results

    np.save(args.output_path, output_data)
    logger.info(f"Successfully generated and saved trajectory to {args.output_path}")

if __name__ == "__main__":
    main()
