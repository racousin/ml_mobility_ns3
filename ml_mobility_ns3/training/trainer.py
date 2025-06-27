# ml_mobility_ns3/training/trainer.py

import torch
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict
from tqdm import tqdm
import logging
import json

from ..models.vae import ConditionalTrajectoryVAE, masked_vae_loss

logger = logging.getLogger(__name__)


class VAETrainer:
    """Trainer for the Conditional Trajectory VAE."""

    def __init__(
        self,
        model: ConditionalTrajectoryVAE,
        device: str,
        learning_rate: float = 1e-4,
        beta: float = 1.0,
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.beta = beta
        self.history = {'train_loss': [], 'val_loss': [], 'train_recon_loss': [], 'val_recon_loss': []}

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        
        for batch in tqdm(dataloader, desc="Training"):
            x, mask, mode, length = [b.to(self.device) for b in batch]

            # Forward pass
            recon, mu, logvar = self.model(x, mode, length)
            loss, metrics = masked_vae_loss(recon, x, mu, logvar, mask, self.beta)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += metrics['loss']
            total_recon_loss += metrics['recon_loss']
            total_kl_loss += metrics['kl_loss']

        n_batches = len(dataloader)
        return {
            'loss': total_loss / n_batches,
            'recon_loss': total_recon_loss / n_batches,
            'kl_loss': total_kl_loss / n_batches
        }

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate on a validation set."""
        self.model.eval()
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        
        for batch in tqdm(dataloader, desc="Evaluating"):
            x, mask, mode, length = [b.to(self.device) for b in batch]

            recon, mu, logvar = self.model(x, mode, length)
            loss, metrics = masked_vae_loss(recon, x, mu, logvar, mask, self.beta)
            
            total_loss += metrics['loss']
            total_recon_loss += metrics['recon_loss']
            total_kl_loss += metrics['kl_loss']

        n_batches = len(dataloader)
        return {
            'loss': total_loss / n_batches,
            'recon_loss': total_recon_loss / n_batches,
            'kl_loss': total_kl_loss / n_batches
        }

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        results_dir: Path,
    ):
        """Train the model and save checkpoints."""
        results_dir.mkdir(parents=True, exist_ok=True)
        best_val_loss = float('inf')

        for epoch in range(epochs):
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)

            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_recon_loss'].append(train_metrics['recon_loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_recon_loss'].append(val_metrics['recon_loss'])

            logger.info(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} (Recon: {train_metrics['recon_loss']:.4f}, KL: {train_metrics['kl_loss']:.4f}) | "
                f"Val Loss: {val_metrics['loss']:.4f} (Recon: {val_metrics['recon_loss']:.4f}, KL: {val_metrics['kl_loss']:.4f})"
            )

            # Save checkpoint if validation loss improved
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                self.save_checkpoint(results_dir / 'best_model.pt')
                logger.info(f"New best model saved with validation loss: {best_val_loss:.4f}")

        # Save final model and history
        self.save_checkpoint(results_dir / 'final_model.pt')
        self.save_history(results_dir / 'history.json')
        logger.info(f"Training complete. Final model and history saved to {results_dir}")

    def save_checkpoint(self, path: Path):
        """Save model and optimizer state."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.model.get_config(), # Assuming you add a get_config method
        }, path)
        logger.debug(f"Saved checkpoint to {path}")

    def save_history(self, path: Path):
        """Save training history to a JSON file."""
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=4)
