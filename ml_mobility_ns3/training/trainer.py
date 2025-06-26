#!/usr/bin/env python
"""Simple training script for trajectory VAE."""

import argparse
import logging
from pathlib import Path
import numpy as np

from ml_mobility_ns3 import (
    NetMob25Loader,
    TrajectoryPreprocessor,
    TrajectoryVAE,
    VAETrainer,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train Trajectory VAE")
    parser.add_argument("--data-dir", type=Path, required=True, help="Path to NetMob25 data")
    parser.add_argument("--output-dir", type=Path, default=Path("output"), help="Output directory")
    parser.add_argument("--n-samples", type=int, default=1000, help="Number of trajectories to sample")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--beta", type=float, default=1.0, help="Beta parameter for VAE")
    parser.add_argument("--latent-dim", type=int, default=32, help="Latent dimension")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--seq-length", type=int, default=50, help="Sequence length")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info("Loading data...")
    loader = NetMob25Loader(args.data_dir)
    trajectories = loader.sample_trajectories(n_samples=args.n_samples, min_points=10)
    logger.info(f"Loaded {len(trajectories)} trajectories")
    
    # Split data
    n_val = int(len(trajectories) * args.val_split)
    train_trajectories = trajectories[:-n_val]
    val_trajectories = trajectories[-n_val:]
    logger.info(f"Train: {len(train_trajectories)}, Val: {len(val_trajectories)}")
    
    # Preprocess
    logger.info("Preprocessing data...")
    preprocessor = TrajectoryPreprocessor(sequence_length=args.seq_length)
    preprocessor.fit(train_trajectories)
    
    train_data = preprocessor.transform(train_trajectories)
    val_data = preprocessor.transform(val_trajectories)
    
    # Save preprocessor
    import pickle
    with open(args.output_dir / "preprocessor.pkl", "wb") as f:
        pickle.dump(preprocessor, f)
    
    # Create model
    logger.info("Creating model...")
    model = TrajectoryVAE(
        input_dim=4,
        sequence_length=args.seq_length,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
    )
    
    # Train
    logger.info("Training...")
    trainer = VAETrainer(
        model,
        device=args.device,
        learning_rate=args.lr,
        beta=args.beta,
    )
    
    trainer.fit(
        train_data,
        val_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        save_path=args.output_dir / "best_model.pt",
    )
    
    # Save training history
    import json
    with open(args.output_dir / "history.json", "w") as f:
        json.dump(trainer.history, f)
    
    logger.info(f"Training complete! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()