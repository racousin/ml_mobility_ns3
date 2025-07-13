import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import torch
import json
from pathlib import Path
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class BestMetricsTracker(Callback):
    """Track best metrics when validation loss improves."""
    
    def __init__(self, experiment_dir: Path):
        super().__init__()
        self.experiment_dir = experiment_dir
        self.best_metrics = {}
        self.best_val_loss = float('inf')
        
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Capture all metrics when validation loss improves."""
        metrics = trainer.callback_metrics
        
        # Get current validation loss
        val_loss = metrics.get('val_loss', float('inf'))
        if isinstance(val_loss, torch.Tensor):
            val_loss = val_loss.item()
        
        # If validation loss improved, capture all metrics
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            
            # Capture all important metrics
            metric_keys = [
                'val_loss', 'val_recon_loss', 'val_kl_loss', 'val_weighted_kl_loss',
                'val_mse', 'val_speed_mse', 'val_total_distance_mae', 
                'val_bird_distance_mae', 'val_frechet_distance',
                'val_distance_loss', 'val_speed_loss', 'val_smoothness_loss'
            ]
            
            self.best_metrics = {
                'epoch': trainer.current_epoch,
                'step': trainer.global_step,
                'val_loss': val_loss
            }
            
            for key in metric_keys:
                if key in metrics:
                    value = metrics[key]
                    if isinstance(value, torch.Tensor):
                        value = value.item()
                    self.best_metrics[key] = value
            
            # Save immediately
            self._save_best_metrics()
            
            logger.info(f"New best validation loss: {val_loss:.6f} at epoch {trainer.current_epoch}")
            logger.info(f"Best metrics: {self.best_metrics}")
    
    def _save_best_metrics(self):
        """Save best metrics to file."""
        metrics_file = self.experiment_dir / "best_metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(self.best_metrics, f, indent=2)
    
    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Update model_info.json with best metrics."""
        model_info_path = self.experiment_dir / "model_info.json"
        
        if model_info_path.exists():
            with open(model_info_path, "r") as f:
                model_info = json.load(f)
            
            # Add best metrics
            model_info['best_metrics'] = self.best_metrics
            
            # Extract key metrics for quick access
            model_info['key_metrics'] = {
                'best_val_loss': self.best_metrics.get('val_loss'),
                'best_mse': self.best_metrics.get('val_mse'),
                'best_frechet': self.best_metrics.get('val_frechet_distance'),
                'best_kl_loss': self.best_metrics.get('val_kl_loss'),
                'best_total_distance_mae': self.best_metrics.get('val_total_distance_mae'),
                'best_bird_distance_mae': self.best_metrics.get('val_bird_distance_mae'),
                'best_speed_mse': self.best_metrics.get('val_speed_mse'),
                'best_epoch': self.best_metrics.get('epoch')
            }
            
            with open(model_info_path, "w") as f:
                json.dump(model_info, f, indent=2)