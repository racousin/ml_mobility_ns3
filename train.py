#!/usr/bin/env python
# train.py
"""Training script for the Conditional Trajectory VAE."""

import argparse
import logging
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
import pickle

# Make sure the project root is in the python path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml_mobility_ns3.models.vae import ConditionalTrajectoryVAE
from ml_mobility_ns3.training.trainer import VAETrainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Train Conditional Trajectory VAE")
    parser.add_argument("--data-path", type=Path, required=True, help="Path to the preprocessed VAE dataset (.npz file)")
    parser.add_argument("--results-dir", type=Path, default=Path("results"), help="Directory to save checkpoints and logs")
    parser.add_argument("--val-split", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--beta", type=float, default=0.001, help="Beta parameter for VAE KL divergence")
    parser.add_argument("--latent-dim", type=int, default=32, help="Latent dimension")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension in LSTMs")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of layers in LSTMs")
    parser.add_argument("--condition-dim", type=int, default=32, help="Dimension for condition embeddings")
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
            device = 'cuda:0'
        else:
            device = f'cuda:{args.gpu_id}'
    else:
        device = 'cpu'
        
    logger.info(f"Using device: {device}")
    
    # Load data
    logger.info(f"Loading preprocessed data from {args.data_path}...")
    try:
        data = np.load(args.data_path)
        # Metadata is in a separate .pkl file, construct its path
        metadata_path = args.data_path.parent / "metadata.pkl"
        logger.info(f"Loading metadata from {metadata_path}...")
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)

    except FileNotFoundError as e:
        logger.error(f"A data file was not found. Please check paths. Error: {e}")
        return

    trajectories = torch.from_numpy(data['trajectories']).float()
    masks = torch.from_numpy(data['masks']).bool()
    transport_modes = torch.from_numpy(data['transport_modes']).long()
    trip_lengths = torch.from_numpy(data['trip_lengths']).long()

    logger.info(f"Loaded {len(trajectories)} trajectories.")
    logger.info(f"Max sequence length: {trajectories.shape[1]}")
    logger.info(f"Number of transport modes: {len(metadata['transport_modes'])}")

    dataset = TensorDataset(trajectories, masks, transport_modes, trip_lengths)
    
    # Split data
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    logger.info(f"Training set size: {len(train_dataset)}")
    logger.info(f"Validation set size: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Create model
    logger.info("Creating VAE model...")
    model = ConditionalTrajectoryVAE(
        input_dim=trajectories.shape[-1],
        sequence_length=trajectories.shape[1],
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_layers=args.num_layers,
        num_transport_modes=len(metadata['transport_modes']),
        condition_dim=args.condition_dim
    )
    
    # To be able to reload the model architecture from the checkpoint, add a get_config method
    def get_config():
        return {
            'input_dim': model.input_dim,
            'sequence_length': model.sequence_length,
            'hidden_dim': model.hidden_dim,
            'latent_dim': model.latent_dim,
            'num_layers': model.num_layers,
            'num_transport_modes': model.transport_mode_embedding.num_embeddings,
            'condition_dim': model.condition_dim
        }
    model.get_config = get_config

    # Train
    logger.info("Starting training...")
    trainer = VAETrainer(
        model,
        device=device,
        learning_rate=args.lr,
        beta=args.beta,
    )
    
    trainer.fit(
        train_loader,
        val_loader,
        epochs=args.epochs,
        results_dir=args.results_dir,
    )

if __name__ == "__main__":
    main()
