# ml_mobility_ns3/training/trainer.py

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
from typing import Dict, Optional, List
from tqdm import tqdm
import logging
import json
import numpy as np

from ..models.vae import ConditionalTrajectoryVAE, masked_vae_loss, compute_trajectory_metrics

logger = logging.getLogger(__name__)


class VAETrainer:
    """Trainer for the Conditional Trajectory VAE."""

    def __init__(
        self,
        model: ConditionalTrajectoryVAE,
        device: str,
        learning_rate: float = 1e-4,
        beta: float = 1.0,
        lr_scheduler_patience: int = 10,
        lr_scheduler_factor: float = 0.5,
        lr_scheduler_min_lr: float = 1e-6,
        early_stopping_patience: int = 20,
        early_stopping_min_delta: float = 1e-4,
        gradient_clip_val: float = 1.0,
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.beta = beta
        self.gradient_clip_val = gradient_clip_val
        
        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=lr_scheduler_patience,
            factor=lr_scheduler_factor,
            min_lr=lr_scheduler_min_lr
        )
        
        # Early stopping parameters
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.early_stopping_counter = 0
        self.best_val_loss = float('inf')
        
        # Training history
        self.history = {
            'train_loss': [], 'val_loss': [], 
            'train_recon_loss': [], 'val_recon_loss': [],
            'train_kl_loss': [], 'val_kl_loss': [],
            'train_speed_mae': [], 'val_speed_mae': [],
            'train_total_distance_mae': [], 'val_total_distance_mae': [],
            'train_bird_distance_mae': [], 'val_bird_distance_mae': [],
            'learning_rates': [],
        }
        
        self.global_step = 0

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        epoch_metrics = {
            'loss': 0, 'recon_loss': 0, 'kl_loss': 0,
            'speed_mae': 0, 'total_distance_mae': 0, 'bird_distance_mae': 0,
        }
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training")):
            x, mask, mode, length = [b.to(self.device) for b in batch]
            
            # Forward pass
            recon, mu, logvar = self.model(x, mode, length, mask)
            loss, loss_metrics = masked_vae_loss(recon, x, mu, logvar, mask, self.beta)
            
            # Compute trajectory metrics
            traj_metrics = compute_trajectory_metrics(recon, x, mask)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
            
            self.optimizer.step()
            
            # Accumulate metrics
            for key in loss_metrics:
                if key in epoch_metrics:
                    epoch_metrics[key] += loss_metrics[key]
            
            for key in traj_metrics:
                if key in epoch_metrics:
                    epoch_metrics[key] += traj_metrics[key]
            
            self.global_step += 1

        # Average metrics
        n_batches = len(dataloader)
        for key in epoch_metrics:
            epoch_metrics[key] /= n_batches
            
        return epoch_metrics

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate on a validation set."""
        self.model.eval()
        
        epoch_metrics = {
            'loss': 0, 'recon_loss': 0, 'kl_loss': 0,
            'speed_mae': 0, 'total_distance_mae': 0, 'bird_distance_mae': 0,
        }
        
        for batch in tqdm(dataloader, desc="Evaluating"):
            x, mask, mode, length = [b.to(self.device) for b in batch]
            
            # Forward pass
            recon, mu, logvar = self.model(x, mode, length, mask)
            loss, loss_metrics = masked_vae_loss(recon, x, mu, logvar, mask, self.beta)
            
            # Compute trajectory metrics
            traj_metrics = compute_trajectory_metrics(recon, x, mask)
            
            # Accumulate metrics
            for key in loss_metrics:
                if key in epoch_metrics:
                    epoch_metrics[key] += loss_metrics[key]
            
            for key in traj_metrics:
                if key in epoch_metrics:
                    epoch_metrics[key] += traj_metrics[key]

        # Average metrics
        n_batches = len(dataloader)
        for key in epoch_metrics:
            epoch_metrics[key] /= n_batches
            
        return epoch_metrics

    @torch.no_grad()
    def evaluate_generation(self, dataloader: DataLoader, n_samples_per_mode: int = 100) -> Dict[str, float]:
        """Evaluate generation quality by generating samples for each transport mode."""
        self.model.eval()
        
        # Get unique transport modes from the dataloader
        all_modes = []
        all_lengths = []
        for batch in dataloader:
            _, _, mode, length = batch
            all_modes.extend(mode.tolist())
            all_lengths.extend(length.tolist())
        
        unique_modes = list(set(all_modes))
        mode_avg_lengths = {}
        for mode in unique_modes:
            mode_lengths = [length for m, length in zip(all_modes, all_lengths) if m == mode]
            mode_avg_lengths[mode] = int(np.mean(mode_lengths))
        
        generation_metrics = {
            'gen_total_distance_mae': 0,
            'gen_bird_distance_mae': 0,
            'gen_speed_mae': 0,
        }
        
        # Generate samples for each mode
        for mode in unique_modes:
            avg_length = mode_avg_lengths[mode]
            
            # Create mode and length tensors
            modes = torch.full((n_samples_per_mode,), mode, dtype=torch.long, device=self.device)
            lengths = torch.full((n_samples_per_mode,), avg_length, dtype=torch.long, device=self.device)
            
            # Generate trajectories
            generated = self.model.generate(modes, lengths, device=self.device)
            
            # Create fake target (we can't evaluate against real targets, so we compute internal consistency)
            # Instead, let's compute some basic trajectory statistics
            generated_np = generated.cpu().numpy()
            
            # Compute basic trajectory statistics
            for i in range(len(generated_np)):
                traj = generated_np[i, :avg_length, :]
                if len(traj) > 1:
                    # Total distance
                    lat_diff = np.diff(traj[:, 0])
                    lon_diff = np.diff(traj[:, 1])
                    total_dist = np.sum(np.sqrt(lat_diff**2 + lon_diff**2)) * 111
                    
                    # Bird distance
                    bird_dist = np.sqrt((traj[-1, 0] - traj[0, 0])**2 + (traj[-1, 1] - traj[0, 1])**2) * 111
                    
                    # Average speed
                    avg_speed = np.mean(traj[:, 2])
                    
                    # Store for later analysis (simplified here)
                    generation_metrics['gen_total_distance_mae'] += total_dist / (n_samples_per_mode * len(unique_modes))
                    generation_metrics['gen_bird_distance_mae'] += bird_dist / (n_samples_per_mode * len(unique_modes))
                    generation_metrics['gen_speed_mae'] += avg_speed / (n_samples_per_mode * len(unique_modes))
        
        return generation_metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        results_dir: Path,
    ):
        """Train the model and save checkpoints."""
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Reset early stopping counter if resuming training
        if self.best_val_loss == float('inf'):
            self.best_val_loss = float('inf')
            self.early_stopping_counter = 0

        for epoch in range(epochs):
            logger.info(f"Epoch {epoch+1}/{epochs}")
            
            # Training
            train_metrics = self.train_epoch(train_loader)
            
            # Validation
            val_metrics = self.evaluate(val_loader)
            
            # Update history
            for key in ['loss', 'recon_loss', 'kl_loss', 'speed_mae', 'total_distance_mae', 'bird_distance_mae']:
                if key in train_metrics:
                    self.history[f'train_{key}'].append(train_metrics[key])
                    self.history[f'val_{key}'].append(val_metrics[key])

            # Track learning rates
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rates'].append(current_lr)

            # Logging
            log_msg = (
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} "
                f"(Recon: {train_metrics['recon_loss']:.4f}, KL: {train_metrics['kl_loss']:.4f}, "
                f"Speed MAE: {train_metrics['speed_mae']:.4f}) | "
                f"Val Loss: {val_metrics['loss']:.4f} "
                f"(Recon: {val_metrics['recon_loss']:.4f}, KL: {val_metrics['kl_loss']:.4f}, "
                f"Speed MAE: {val_metrics['speed_mae']:.4f}) | "
                f"LR: {current_lr:.2e}"
            )
            logger.info(log_msg)

            # Learning rate scheduling
            self.scheduler.step(val_metrics['loss'])

            # Early stopping check
            if val_metrics['loss'] < self.best_val_loss - self.early_stopping_min_delta:
                self.best_val_loss = val_metrics['loss']
                self.early_stopping_counter = 0
                self.save_checkpoint(results_dir / 'best_model.pt')
                logger.info(f"New best model saved with validation loss: {self.best_val_loss:.4f}")
            else:
                self.early_stopping_counter += 1
                logger.info(f"Early stopping counter: {self.early_stopping_counter}/{self.early_stopping_patience}")
                
                if self.early_stopping_counter >= self.early_stopping_patience:
                    logger.info(f"Early stopping triggered! No improvement for {self.early_stopping_patience} epochs.")
                    break

        # Final evaluation including generation
        logger.info("Running final evaluation including generation quality...")
        final_val_metrics = self.evaluate(val_loader)
        generation_metrics = self.evaluate_generation(val_loader)
        
        # Save final results
        final_results = {
            'final_validation': final_val_metrics,
            'generation_evaluation': generation_metrics,
            'training_history': self.history
        }
        
        with open(results_dir / 'final_results.json', 'w') as f:
            json.dump(final_results, f, indent=2)

        # Save final model and history
        self.save_checkpoint(results_dir / 'final_model.pt')
        self.save_history(results_dir / 'history.json')
        
        if self.early_stopping_counter >= self.early_stopping_patience:
            logger.info(f"Training stopped early at epoch {epoch+1}. Final model and history saved to {results_dir}")
        else:
            logger.info(f"Training completed after {epochs} epochs. Final model and history saved to {results_dir}")

    def save_checkpoint(self, path: Path):
        """Save model and optimizer state."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.model.get_config(),
            'beta': self.beta,
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'early_stopping_counter': self.early_stopping_counter,
            'early_stopping_patience': self.early_stopping_patience,
            'early_stopping_min_delta': self.early_stopping_min_delta,
            'gradient_clip_val': self.gradient_clip_val,
        }
        
        torch.save(checkpoint, path)
        logger.debug(f"Saved checkpoint to {path}")

    def save_history(self, path: Path):
        """Save training history to a JSON file."""
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=4)
    
    @classmethod
    def load_checkpoint(cls, checkpoint_path: Path, device: str = 'cpu'):
        """Load a model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Recreate model
        config = checkpoint['config']
        model = ConditionalTrajectoryVAE(**config)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Create trainer with saved parameters
        trainer = cls(
            model=model,
            device=device,
            beta=checkpoint.get('beta', 1.0),
            early_stopping_patience=checkpoint.get('early_stopping_patience', 20),
            early_stopping_min_delta=checkpoint.get('early_stopping_min_delta', 1e-4),
            gradient_clip_val=checkpoint.get('gradient_clip_val', 1.0),
        )
        
        # Load optimizer state
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.global_step = checkpoint.get('global_step', 0)
        trainer.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        trainer.early_stopping_counter = checkpoint.get('early_stopping_counter', 0)
        
        # Load scheduler state
        if 'scheduler_state_dict' in checkpoint:
            trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return trainer