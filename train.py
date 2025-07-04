#!/usr/bin/env python
# train.py
"""Training script for the Conditional Trajectory VAE with optional VAE-GAN training."""

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

from ml_mobility_ns3.models.vae import ConditionalTrajectoryVAE, TrajectoryDiscriminator
from ml_mobility_ns3.training.trainer import VAETrainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Train Conditional Trajectory VAE or VAE-GAN")
    
    # Data arguments
    parser.add_argument("--data-path", type=Path, required=True, help="Path to the preprocessed VAE dataset (.npz file)")
    parser.add_argument("--results-dir", type=Path, default=Path("results"), help="Directory to save checkpoints and logs")
    parser.add_argument("--val-split", type=float, default=0.15, help="Validation split ratio")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--beta", type=float, default=0.001, help="Beta parameter for VAE KL divergence")
    
    # Model architecture arguments
    parser.add_argument("--architecture", type=str, default='lstm', choices=['lstm', 'attention'], 
                        help="Model architecture to use")
    parser.add_argument("--latent-dim", type=int, default=32, help="Latent dimension")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of layers")
    parser.add_argument("--condition-dim", type=int, default=32, help="Dimension for condition embeddings")
    
    # Attention-specific arguments
    parser.add_argument("--n-heads", type=int, default=8, help="Number of attention heads (for attention architecture)")
    parser.add_argument("--d-ff", type=int, default=1024, help="Feed-forward dimension (for attention architecture)")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate (for attention architecture)")
    parser.add_argument("--use-causal-mask", action='store_true', help="Use causal masking in attention")
    parser.add_argument("--pooling", type=str, default='mean', choices=['mean', 'max', 'cls'],
                        help="Pooling method for attention architecture")
    
    # VAE-GAN specific arguments
    parser.add_argument("--use-gan", action='store_true', help="Use VAE-GAN training instead of standard VAE")
    parser.add_argument("--disc-lr", type=float, default=1e-4, help="Discriminator learning rate")
    parser.add_argument("--gamma", type=float, default=1.0, help="Weight for GAN loss in VAE-GAN")
    parser.add_argument("--disc-update-freq", type=int, default=1, help="Update discriminator every N generator updates")
    parser.add_argument("--disc-hidden-dim", type=int, default=128, help="Discriminator hidden dimension")
    parser.add_argument("--disc-num-layers", type=int, default=2, help="Number of discriminator layers")
    
    # Hardware arguments
    parser.add_argument("--device", type=str, default=None, help="Device to use (e.g., 'cuda', 'cpu')")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU ID to use if running on CUDA")
    
    # Learning rate scheduler arguments
    parser.add_argument("--lr-patience", type=int, default=10, help="Patience for learning rate scheduler")
    parser.add_argument("--lr-factor", type=float, default=0.5, help="Factor to reduce learning rate")
    parser.add_argument("--lr-min", type=float, default=1e-6, help="Minimum learning rate")
    
    # Early stopping arguments
    parser.add_argument("--early-stopping-patience", type=int, default=20, help="Patience for early stopping")
    parser.add_argument("--early-stopping-delta", type=float, default=1e-4, help="Minimum improvement for early stopping")
    
    # Resume training
    parser.add_argument("--resume", type=Path, default=None, help="Path to checkpoint to resume training from")
    
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
    
    # Prepare attention parameters if using attention architecture
    attention_params = None
    if args.architecture == 'attention':
        attention_params = {
            'n_heads': args.n_heads,
            'd_ff': args.d_ff if args.d_ff > 0 else args.hidden_dim * 4,  # Default to 4x hidden_dim
            'dropout': args.dropout,
            'use_causal_mask': args.use_causal_mask,
            'pooling': args.pooling,
        }
        logger.info(f"Using attention architecture with params: {attention_params}")
    
    if args.resume:
        # Resume from checkpoint
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer = VAETrainer.load_checkpoint(args.resume, device=device)
        model = trainer.model
        discriminator = trainer.discriminator if args.use_gan else None
    else:
        # Create new model
        logger.info(f"Creating {args.architecture.upper()} VAE model...")
        model = ConditionalTrajectoryVAE(
            input_dim=trajectories.shape[-1],
            sequence_length=trajectories.shape[1],
            hidden_dim=args.hidden_dim,
            latent_dim=args.latent_dim,
            num_layers=args.num_layers,
            num_transport_modes=len(metadata['transport_modes']),
            condition_dim=args.condition_dim,
            architecture=args.architecture,
            attention_params=attention_params,
        )
        
        # Create discriminator if using GAN
        discriminator = None
        if args.use_gan:
            logger.info(f"Creating {args.architecture.upper()} discriminator...")
            disc_attention_params = None
            if args.architecture == 'attention':
                disc_attention_params = {
                    'n_heads': args.n_heads,
                    'd_ff': args.disc_hidden_dim * 4,
                    'dropout': args.dropout,
                    'pooling': args.pooling,
                }
            
            discriminator = TrajectoryDiscriminator(
                input_dim=trajectories.shape[-1],
                sequence_length=trajectories.shape[1],
                hidden_dim=args.disc_hidden_dim,
                num_layers=args.disc_num_layers,
                num_transport_modes=len(metadata['transport_modes']),
                condition_dim=args.condition_dim,
                architecture=args.architecture,
                attention_params=disc_attention_params,
            )
    
    # Log model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"VAE total parameters: {total_params:,}")
    logger.info(f"VAE trainable parameters: {trainable_params:,}")
    
    if discriminator is not None:
        disc_params = sum(p.numel() for p in discriminator.parameters())
        logger.info(f"Discriminator parameters: {disc_params:,}")
        logger.info(f"Total model parameters: {total_params + disc_params:,}")
    
    # Save configuration
    model_type = "vae_gan" if args.use_gan else "vae"
    results_dir = args.results_dir / f"{args.architecture}_{model_type}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    config = {
        'model_type': model_type,
        'data_path': str(args.data_path),
        'architecture': args.architecture,
        'model_config': model.get_config(),
        'training_config': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'beta': args.beta,
            'val_split': args.val_split,
        },
        'device': device,
    }
    
    if args.use_gan:
        config['gan_config'] = {
            'disc_lr': args.disc_lr,
            'gamma': args.gamma,
            'disc_update_freq': args.disc_update_freq,
            'disc_hidden_dim': args.disc_hidden_dim,
            'disc_num_layers': args.disc_num_layers,
        }
    
    with open(results_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    # Create trainer
    if not args.resume:
        logger.info(f"Creating {'VAE-GAN' if args.use_gan else 'VAE'} trainer...")
        trainer = VAETrainer(
            model,
            device=device,
            learning_rate=args.lr,
            beta=args.beta,
            use_gan=args.use_gan,
            discriminator=discriminator,
            disc_lr=args.disc_lr if args.use_gan else None,
            gamma=args.gamma if args.use_gan else None,
            disc_update_freq=args.disc_update_freq if args.use_gan else None,
            lr_scheduler_patience=args.lr_patience,
            lr_scheduler_factor=args.lr_factor,
            lr_scheduler_min_lr=args.lr_min,
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_min_delta=args.early_stopping_delta,
        )
    
    # Train
    logger.info(f"Starting {'VAE-GAN' if args.use_gan else 'VAE'} training...")
    trainer.fit(
        train_loader,
        val_loader,
        epochs=args.epochs,
        results_dir=results_dir,
    )
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()