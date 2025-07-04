# ml_mobility_ns3/training/trainer.py

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
from typing import Dict, Optional
from tqdm import tqdm
import logging
import json

from ..models.vae import ConditionalTrajectoryVAE, TrajectoryDiscriminator, masked_vae_loss, vae_gan_loss

logger = logging.getLogger(__name__)


class VAETrainer:
    """Trainer for the Conditional Trajectory VAE with optional GAN training."""

    def __init__(
        self,
        model: ConditionalTrajectoryVAE,
        device: str,
        learning_rate: float = 1e-4,
        beta: float = 1.0,
        use_gan: bool = False,
        discriminator: Optional[TrajectoryDiscriminator] = None,
        disc_lr: float = 1e-4,
        gamma: float = 1.0,
        disc_update_freq: int = 1,
        lr_scheduler_patience: int = 10,
        lr_scheduler_factor: float = 0.5,
        lr_scheduler_min_lr: float = 1e-6,
        early_stopping_patience: int = 20,
        early_stopping_min_delta: float = 1e-4,
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.beta = beta
        self.use_gan = use_gan
        self.gamma = gamma
        self.disc_update_freq = disc_update_freq
        
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
        
        # GAN-specific components
        if self.use_gan:
            if discriminator is None:
                raise ValueError("Discriminator must be provided when use_gan=True")
            self.discriminator = discriminator.to(device)
            self.disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=disc_lr)
            self.disc_scheduler = ReduceLROnPlateau(
                self.disc_optimizer,
                mode='min',
                patience=lr_scheduler_patience,
                factor=lr_scheduler_factor,
                min_lr=lr_scheduler_min_lr
            )
        else:
            self.discriminator = None
            self.disc_optimizer = None
            self.disc_scheduler = None
        
        self.history = {
            'train_loss': [], 'val_loss': [], 
            'train_recon_loss': [], 'val_recon_loss': [],
            'train_kl_loss': [], 'val_kl_loss': [],
            'learning_rates': [],
        }
        
        if self.use_gan:
            self.history.update({
                'train_gen_loss': [], 'val_gen_loss': [],
                'train_disc_loss': [], 'val_disc_loss': [],
                'train_disc_real': [], 'val_disc_real': [],
                'train_disc_fake': [], 'val_disc_fake': [],
                'train_disc_recon': [], 'val_disc_recon': [],
                'disc_learning_rates': [],
            })
        
        self.global_step = 0

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        if self.use_gan:
            self.discriminator.train()
            
        epoch_metrics = {
            'loss': 0, 'recon_loss': 0, 'kl_loss': 0,
            'gen_loss': 0, 'disc_loss': 0,
            'disc_real': 0, 'disc_fake': 0, 'disc_recon': 0,
        }
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training")):
            x, mask, mode, length = [b.to(self.device) for b in batch]
            
            if self.use_gan:
                metrics = self._train_step_gan(x, mask, mode, length, batch_idx)
            else:
                metrics = self._train_step_vae(x, mask, mode, length)
            
            # Accumulate metrics
            for key in metrics:
                if key in epoch_metrics:
                    epoch_metrics[key] += metrics[key]
            
            self.global_step += 1

        # Average metrics
        n_batches = len(dataloader)
        for key in epoch_metrics:
            epoch_metrics[key] /= n_batches
            
        return epoch_metrics
    
    def _train_step_vae(self, x, mask, mode, length):
        """Standard VAE training step."""
        # Forward pass
        recon, mu, logvar = self.model(x, mode, length, mask)
        loss, metrics = masked_vae_loss(recon, x, mu, logvar, mask, self.beta)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return metrics
    
    def _train_step_gan(self, x, mask, mode, length, batch_idx):
        """VAE-GAN training step."""
        batch_size = x.size(0)
        
        # =================
        # Train Discriminator
        # =================
        if batch_idx % self.disc_update_freq == 0:
            # Generate fake samples
            with torch.no_grad():
                # Sample from prior
                z = torch.randn(batch_size, self.model.latent_dim).to(self.device)
                conditions = self.model.get_conditions(mode, length)
                fake = self.model.decode(z, conditions)
                
                # Get reconstructions
                recon, mu, logvar = self.model(x, mode, length, mask)
            
            # Discriminator forward pass
            disc_real_logits = self.discriminator(x, mode, length, mask)
            disc_fake_logits = self.discriminator(fake.detach(), mode, length, mask)
            disc_recon_logits = self.discriminator(recon.detach(), mode, length, mask)
            
            # Calculate discriminator loss
            disc_loss_real = torch.nn.functional.binary_cross_entropy_with_logits(
                disc_real_logits, torch.ones_like(disc_real_logits)
            )
            disc_loss_fake = torch.nn.functional.binary_cross_entropy_with_logits(
                disc_fake_logits, torch.zeros_like(disc_fake_logits)
            )
            disc_loss_recon = torch.nn.functional.binary_cross_entropy_with_logits(
                disc_recon_logits, torch.zeros_like(disc_recon_logits)
            )
            disc_loss = disc_loss_real + disc_loss_fake + disc_loss_recon
            
            # Update discriminator
            self.disc_optimizer.zero_grad()
            disc_loss.backward()
            self.disc_optimizer.step()
        
        # =================
        # Train Generator (VAE)
        # =================
        # Generate new samples for generator training
        z = torch.randn(batch_size, self.model.latent_dim).to(self.device)
        conditions = self.model.get_conditions(mode, length)
        fake = self.model.decode(z, conditions)
        
        # Get reconstructions
        recon, mu, logvar = self.model(x, mode, length, mask)
        
        # Discriminator predictions for generator loss
        disc_fake_logits = self.discriminator(fake, mode, length, mask)
        disc_recon_logits = self.discriminator(recon, mode, length, mask)
        
        # For metrics only (not used in backprop)
        with torch.no_grad():
            disc_real_logits = self.discriminator(x, mode, length, mask)
        
        # Calculate VAE-GAN loss
        losses, metrics = vae_gan_loss(
            recon, x, mu, logvar, mask,
            disc_real_logits, disc_fake_logits, disc_recon_logits,
            self.beta, self.gamma
        )
        
        # Update generator (VAE)
        self.optimizer.zero_grad()
        losses['gen_total'].backward()
        self.optimizer.step()
        
        return metrics

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate on a validation set."""
        self.model.eval()
        if self.use_gan:
            self.discriminator.eval()
            
        epoch_metrics = {
            'loss': 0, 'recon_loss': 0, 'kl_loss': 0,
            'gen_loss': 0, 'disc_loss': 0,
            'disc_real': 0, 'disc_fake': 0, 'disc_recon': 0,
        }
        
        for batch in tqdm(dataloader, desc="Evaluating"):
            x, mask, mode, length = [b.to(self.device) for b in batch]
            
            if self.use_gan:
                metrics = self._eval_step_gan(x, mask, mode, length)
            else:
                metrics = self._eval_step_vae(x, mask, mode, length)
            
            # Accumulate metrics
            for key in metrics:
                if key in epoch_metrics:
                    epoch_metrics[key] += metrics[key]

        # Average metrics
        n_batches = len(dataloader)
        for key in epoch_metrics:
            epoch_metrics[key] /= n_batches
            
        return epoch_metrics
    
    def _eval_step_vae(self, x, mask, mode, length):
        """Standard VAE evaluation step."""
        recon, mu, logvar = self.model(x, mode, length, mask)
        loss, metrics = masked_vae_loss(recon, x, mu, logvar, mask, self.beta)
        return metrics
    
    def _eval_step_gan(self, x, mask, mode, length):
        """VAE-GAN evaluation step."""
        batch_size = x.size(0)
        
        # Generate samples
        z = torch.randn(batch_size, self.model.latent_dim).to(self.device)
        conditions = self.model.get_conditions(mode, length)
        fake = self.model.decode(z, conditions)
        
        # Get reconstructions
        recon, mu, logvar = self.model(x, mode, length, mask)
        
        # Discriminator predictions
        disc_real_logits = self.discriminator(x, mode, length, mask)
        disc_fake_logits = self.discriminator(fake, mode, length, mask)
        disc_recon_logits = self.discriminator(recon, mode, length, mask)
        
        # Calculate losses
        _, metrics = vae_gan_loss(
            recon, x, mu, logvar, mask,
            disc_real_logits, disc_fake_logits, disc_recon_logits,
            self.beta, self.gamma
        )
        
        return metrics

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
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)

            # Update history
            for key in ['loss', 'recon_loss', 'kl_loss']:
                if key in train_metrics:
                    self.history[f'train_{key}'].append(train_metrics[key])
                    self.history[f'val_{key}'].append(val_metrics[key])
            
            if self.use_gan:
                for key in ['gen_loss', 'disc_loss', 'disc_real', 'disc_fake', 'disc_recon']:
                    if key in train_metrics:
                        self.history[f'train_{key}'].append(train_metrics[key])
                        self.history[f'val_{key}'].append(val_metrics[key])

            # Track learning rates
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rates'].append(current_lr)
            if self.use_gan:
                disc_lr = self.disc_optimizer.param_groups[0]['lr']
                self.history['disc_learning_rates'].append(disc_lr)

            # Logging
            log_msg = (
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} "
                f"(Recon: {train_metrics['recon_loss']:.4f}, KL: {train_metrics['kl_loss']:.4f}"
            )
            
            if self.use_gan:
                log_msg += (
                    f", Gen: {train_metrics['gen_loss']:.4f}, "
                    f"Disc: {train_metrics['disc_loss']:.4f}"
                )
            
            log_msg += (
                f") | Val Loss: {val_metrics['loss']:.4f} "
                f"(Recon: {val_metrics['recon_loss']:.4f}, KL: {val_metrics['kl_loss']:.4f}"
            )
            
            if self.use_gan:
                log_msg += (
                    f", Gen: {val_metrics['gen_loss']:.4f}, "
                    f"Disc: {val_metrics['disc_loss']:.4f}"
                )
            
            log_msg += f") | LR: {current_lr:.2e}"
            
            if self.use_gan:
                log_msg += f" | Disc LR: {disc_lr:.2e}"
                
            logger.info(log_msg)

            # Learning rate scheduling
            self.scheduler.step(val_metrics['loss'])
            if self.use_gan and self.disc_scheduler:
                self.disc_scheduler.step(val_metrics['disc_loss'])

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
            'use_gan': self.use_gan,
            'beta': self.beta,
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'early_stopping_counter': self.early_stopping_counter,
            'early_stopping_patience': self.early_stopping_patience,
            'early_stopping_min_delta': self.early_stopping_min_delta,
        }
        
        if self.use_gan:
            checkpoint.update({
                'discriminator_state_dict': self.discriminator.state_dict(),
                'disc_optimizer_state_dict': self.disc_optimizer.state_dict(),
                'disc_scheduler_state_dict': self.disc_scheduler.state_dict(),
                'gamma': self.gamma,
                'disc_update_freq': self.disc_update_freq,
                'disc_config': {
                    'input_dim': self.discriminator.input_dim,
                    'sequence_length': self.discriminator.sequence_length,
                    'hidden_dim': self.discriminator.hidden_dim,
                    'num_layers': self.discriminator.num_layers,
                    'num_transport_modes': self.model.num_transport_modes,
                    'condition_dim': self.discriminator.condition_dim,
                    'architecture': self.discriminator.architecture,
                }
            })
        
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
            use_gan=checkpoint.get('use_gan', False),
            early_stopping_patience=checkpoint.get('early_stopping_patience', 20),
            early_stopping_min_delta=checkpoint.get('early_stopping_min_delta', 1e-4),
        )
        
        # Load optimizer state
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.global_step = checkpoint.get('global_step', 0)
        trainer.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        trainer.early_stopping_counter = checkpoint.get('early_stopping_counter', 0)
        
        # Load scheduler state
        if 'scheduler_state_dict' in checkpoint:
            trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load GAN components if used
        if trainer.use_gan and 'discriminator_state_dict' in checkpoint:
            disc_config = checkpoint.get('disc_config', {})
            trainer.discriminator = TrajectoryDiscriminator(**disc_config)
            trainer.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            trainer.discriminator.to(device)
            
            trainer.disc_optimizer = torch.optim.Adam(trainer.discriminator.parameters())
            trainer.disc_optimizer.load_state_dict(checkpoint['disc_optimizer_state_dict'])
            
            trainer.gamma = checkpoint.get('gamma', 1.0)
            trainer.disc_update_freq = checkpoint.get('disc_update_freq', 1)
            
            # Load discriminator scheduler if available
            if 'disc_scheduler_state_dict' in checkpoint:
                trainer.disc_scheduler = ReduceLROnPlateau(
                    trainer.disc_optimizer,
                    mode='min',
                    patience=10,
                    factor=0.5,
                    min_lr=1e-6
                )
                trainer.disc_scheduler.load_state_dict(checkpoint['disc_scheduler_state_dict'])
        
        return trainer