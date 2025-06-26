import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
from typing import Optional, Dict
from tqdm import tqdm
import logging

from ..models.vae import TrajectoryVAE, vae_loss

logger = logging.getLogger(__name__)


class VAETrainer:
    """Simple trainer for Trajectory VAE."""
    
    def __init__(
        self,
        model: TrajectoryVAE,
        device: Optional[str] = None,
        learning_rate: float = 1e-3,
        beta: float = 1.0,
    ):
        self.model = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.beta = beta
        
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.history = {'train_loss': [], 'val_loss': []}
        
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_recon = 0
        total_kl = 0
        
        for batch in tqdm(dataloader, desc="Training"):
            x = batch[0].to(self.device)
            
            # Forward pass
            recon, mu, logvar = self.model(x)
            loss, metrics = vae_loss(recon, x, mu, logvar, self.beta)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += metrics['loss']
            total_recon += metrics['recon_loss']
            total_kl += metrics['kl_loss']
            
        n_batches = len(dataloader)
        return {
            'loss': total_loss / n_batches,
            'recon_loss': total_recon / n_batches,
            'kl_loss': total_kl / n_batches
        }
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0
        total_recon = 0
        total_kl = 0
        
        with torch.no_grad():
            for batch in dataloader:
                x = batch[0].to(self.device)
                recon, mu, logvar = self.model(x)
                loss, metrics = vae_loss(recon, x, mu, logvar, self.beta)
                
                total_loss += metrics['loss']
                total_recon += metrics['recon_loss']
                total_kl += metrics['kl_loss']
                
        n_batches = len(dataloader)
        return {
            'loss': total_loss / n_batches,
            'recon_loss': total_recon / n_batches,
            'kl_loss': total_kl / n_batches
        }
    
    def fit(
        self,
        train_data: np.ndarray,
        val_data: Optional[np.ndarray] = None,
        epochs: int = 50,
        batch_size: int = 32,
        save_path: Optional[Path] = None,
    ):
        """Train the model."""
        # Create dataloaders
        train_tensor = torch.FloatTensor(train_data)
        train_dataset = TensorDataset(train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_loader = None
        if val_data is not None:
            val_tensor = torch.FloatTensor(val_data)
            val_dataset = TensorDataset(val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Train
            train_metrics = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_metrics['loss'])
            
            # Validate
            if val_loader:
                val_metrics = self.evaluate(val_loader)
                self.history['val_loss'].append(val_metrics['loss'])
                
                logger.info(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"Train Loss: {train_metrics['loss']:.4f} "
                    f"(Recon: {train_metrics['recon_loss']:.4f}, KL: {train_metrics['kl_loss']:.4f}) - "
                    f"Val Loss: {val_metrics['loss']:.4f}"
                )
                
                # Save best model
                if save_path and val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    self.save_checkpoint(save_path)
            else:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"Train Loss: {train_metrics['loss']:.4f} "
                    f"(Recon: {train_metrics['recon_loss']:.4f}, KL: {train_metrics['kl_loss']:.4f})"
                )
                
                if save_path:
                    self.save_checkpoint(save_path)
    
    def save_checkpoint(self, path: Path):
        """Save model checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'beta': self.beta,
        }, path)
        logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)
        self.beta = checkpoint.get('beta', self.beta)