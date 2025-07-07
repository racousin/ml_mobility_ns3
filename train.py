#!/usr/bin/env python
# train.py
"""Training script for the Conditional Trajectory VAE with LSTM architecture."""

import argparse
import logging
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
import pickle
import json

# Make sure the project root is in the python path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml_mobility_ns3.models.vae import ConditionalTrajectoryVAE
from ml_mobility_ns3.training.trainer import VAETrainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(data_path: Path, metadata_path: Path):
    """Load and prepare the NetMob25 dataset."""
    logger.info(f"Loading preprocessed data from {data_path}...")
    
    try:
        data = np.load(data_path)
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
    except FileNotFoundError as e:
        logger.error(f"Data file not found. Please check paths. Error: {e}")
        raise

    trajectories = torch.from_numpy(data['trajectories']).float()
    masks = torch.from_numpy(data['masks']).bool()
    
    # Handle different possible column names for transport modes and trip lengths
    if 'transport_modes' in data:
        transport_modes = torch.from_numpy(data['transport_modes']).long()
    elif 'categories' in data:
        transport_modes = torch.from_numpy(data['categories']).long()
    else:
        raise KeyError("Neither 'transport_modes' nor 'categories' found in data")
    
    if 'trip_lengths' in data:
        trip_lengths = torch.from_numpy(data['trip_lengths']).long()
    elif 'weights' in data:
        # If trip_lengths not available, use mask to compute lengths
        trip_lengths = masks.sum(dim=1).long()
    else:
        trip_lengths = masks.sum(dim=1).long()

    logger.info(f"Loaded {len(trajectories)} trajectories.")
    logger.info(f"Sequence length: {trajectories.shape[1]}")
    logger.info(f"Input dimensions: {trajectories.shape[2]}")
    
    # Get transport mode information
    if 'categories' in metadata:
        transport_modes_list = metadata['categories']
    elif 'transport_modes' in metadata:
        transport_modes_list = metadata['transport_modes']
    else:
        transport_modes_list = [f"Mode_{i}" for i in range(transport_modes.max().item() + 1)]
    
    logger.info(f"Transport modes: {transport_modes_list}")
    logger.info(f"Number of transport modes: {len(transport_modes_list)}")

    return trajectories, masks, transport_modes, trip_lengths, transport_modes_list, metadata

def create_dataloaders(trajectories, masks, transport_modes, trip_lengths, batch_size, val_split, num_workers=4):
    """Create train and validation dataloaders."""
    dataset = TensorDataset(trajectories, masks, transport_modes, trip_lengths)
    
    # Split data
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    logger.info(f"Training set size: {len(train_dataset)}")
    logger.info(f"Validation set size: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader

def main():
    parser = argparse.ArgumentParser(description="Train Conditional Trajectory VAE with LSTM")
    
    # Data arguments
    parser.add_argument("--data-path", type=Path, required=True, 
                        help="Path to the preprocessed VAE dataset (.npz file)")
    parser.add_argument("--metadata-path", type=Path, 
                        help="Path to metadata file (.pkl). If not provided, will look in same dir as data")
    parser.add_argument("--results-dir", type=Path, default=Path("results"), 
                        help="Directory to save checkpoints and logs")
    parser.add_argument("--val-split", type=float, default=0.15, help="Validation split ratio")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--beta", type=float, default=0.001, help="Beta parameter for VAE KL divergence")
    parser.add_argument("--gradient-clip", type=float, default=1.0, help="Gradient clipping value")
    
    # Model architecture arguments
    parser.add_argument("--latent-dim", type=int, default=32, help="Latent dimension")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of LSTM layers")
    parser.add_argument("--condition-dim", type=int, default=32, help="Dimension for condition embeddings")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    
    # Hardware arguments
    parser.add_argument("--device", type=str, default=None, help="Device to use (e.g., 'cuda', 'cpu')")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU ID to use if running on CUDA")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of dataloader workers")
    
    # Learning rate scheduler arguments
    parser.add_argument("--lr-patience", type=int, default=10, help="Patience for learning rate scheduler")
    parser.add_argument("--lr-factor", type=float, default=0.5, help="Factor to reduce learning rate")
    parser.add_argument("--lr-min", type=float, default=1e-6, help="Minimum learning rate")
    
    # Early stopping arguments
    parser.add_argument("--early-stopping-patience", type=int, default=20, help="Patience for early stopping")
    parser.add_argument("--early-stopping-delta", type=float, default=1e-4, 
                        help="Minimum improvement for early stopping")
    
    # Resume training
    parser.add_argument("--resume", type=Path, default=None, 
                        help="Path to checkpoint to resume training from")
    
    # Experiment naming
    parser.add_argument("--experiment-name", type=str, default=None,
                        help="Name for this experiment (will be used as subdirectory)")
    
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
    
    # Set metadata path if not provided
    if args.metadata_path is None:
        args.metadata_path = args.data_path.parent / "metadata.pkl"
    
    # Load data
    trajectories, masks, transport_modes, trip_lengths, transport_modes_list, metadata = load_data(
        args.data_path, args.metadata_path
    )
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        trajectories, masks, transport_modes, trip_lengths, 
        args.batch_size, args.val_split, args.num_workers
    )
    
    # Setup results directory
    if args.experiment_name:
        results_dir = args.results_dir / args.experiment_name
    else:
        # Create experiment name based on key parameters
        exp_name = f"lstm_h{args.hidden_dim}_l{args.latent_dim}_layers{args.num_layers}_beta{args.beta}"
        results_dir = args.results_dir / exp_name
    
    results_dir.mkdir(parents=True, exist_ok=True)
    
    if args.resume:
        # Resume from checkpoint
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer = VAETrainer.load_checkpoint(args.resume, device=device)
        model = trainer.model
    else:
        # Create new model
        logger.info("Creating LSTM VAE model...")
        model = ConditionalTrajectoryVAE(
            input_dim=trajectories.shape[-1],
            sequence_length=trajectories.shape[1],
            hidden_dim=args.hidden_dim,
            latent_dim=args.latent_dim,
            num_layers=args.num_layers,
            num_transport_modes=len(transport_modes_list),
            condition_dim=args.condition_dim,
            dropout=args.dropout,
        )
    
        # Create trainer
        logger.info("Creating VAE trainer...")
        trainer = VAETrainer(
            model,
            device=device,
            learning_rate=args.lr,
            beta=args.beta,
            lr_scheduler_patience=args.lr_patience,
            lr_scheduler_factor=args.lr_factor,
            lr_scheduler_min_lr=args.lr_min,
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_min_delta=args.early_stopping_delta,
            gradient_clip_val=args.gradient_clip,
        )
    
    # Log model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Save configuration
    config = {
        'data_path': str(args.data_path),
        'model_config': model.get_config(),
        'training_config': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'beta': args.beta,
            'val_split': args.val_split,
            'gradient_clip': args.gradient_clip,
            'lr_patience': args.lr_patience,
            'lr_factor': args.lr_factor,
            'lr_min': args.lr_min,
            'early_stopping_patience': args.early_stopping_patience,
            'early_stopping_delta': args.early_stopping_delta,
        },
        'dataset_info': {
            'n_samples': len(trajectories),
            'sequence_length': trajectories.shape[1],
            'input_dim': trajectories.shape[2],
            'transport_modes': transport_modes_list,
            'n_transport_modes': len(transport_modes_list),
        },
        'device': device,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
    }
    
    with open(results_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    # Train
    logger.info("Starting VAE training...")
    trainer.fit(
        train_loader,
        val_loader,
        epochs=args.epochs,
        results_dir=results_dir,
    )
    
    logger.info("Training completed successfully!")
    logger.info(f"Results saved to: {results_dir}")

if __name__ == "__main__":
    main()