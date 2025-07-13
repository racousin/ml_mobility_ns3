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
            
            # Capture only metrics that actually exist
            # Core metrics (always available)
            metric_keys = [
                'val_loss', 'val_recon_loss', 'val_kl_loss', 'val_weighted_kl_loss',
                'val_mse', 'val_speed_mae', 'val_distance_mae', 'val_total_distance_mae', 
                'val_bird_distance_mae'
            ]
            
            # Optional loss components (only if using specific loss functions)
            optional_keys = [
                'val_distance_loss', 'val_weighted_distance_loss',  # DistanceAwareVAELoss
                'val_speed_loss', 'val_smoothness_loss', 'val_weighted_speed_loss'  # SpeedAwareVAELoss
            ]
            
            # Add optional metrics if they exist
            for key in optional_keys:
                if key in metrics:
                    metric_keys.append(key)
            
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
            
            # IMPROVED: Better formatted logging
            logger.info(f"New best validation loss: {val_loss:.6f} at epoch {trainer.current_epoch}")
            self._log_formatted_metrics()
    
    def _log_formatted_metrics(self):
        """Log metrics in a clear, formatted way."""
        logger.info("=" * 60)
        logger.info("BEST VALIDATION METRICS")
        logger.info("=" * 60)
        logger.info(f"Epoch: {self.best_metrics.get('epoch', 'N/A'):>8}")
        logger.info(f"Step:  {self.best_metrics.get('step', 'N/A'):>8}")
        logger.info("-" * 60)
        
        # Core loss metrics (always available)
        if 'val_loss' in self.best_metrics:
            logger.info(f"Validation Loss:     {self.best_metrics['val_loss']:>10.6f}")
        if 'val_recon_loss' in self.best_metrics:
            logger.info(f"Reconstruction:      {self.best_metrics['val_recon_loss']:>10.6f}")
        if 'val_kl_loss' in self.best_metrics:
            logger.info(f"KL Divergence:       {self.best_metrics['val_kl_loss']:>10.6f}")
        
        # Optional loss components
        if 'val_distance_loss' in self.best_metrics:
            logger.info(f"Distance Loss:       {self.best_metrics['val_distance_loss']:>10.6f}")
        if 'val_speed_loss' in self.best_metrics:
            logger.info(f"Speed Loss:          {self.best_metrics['val_speed_loss']:>10.6f}")
        
        logger.info("-" * 60)
        # Evaluation metrics (in interpretable units)
        if 'val_mse' in self.best_metrics:
            logger.info(f"MSE (scaled):        {self.best_metrics['val_mse']:>10.6f}")
        if 'val_speed_mae' in self.best_metrics:
            logger.info(f"Speed MAE (km/h):    {self.best_metrics['val_speed_mae']:>10.3f}")
        if 'val_distance_mae' in self.best_metrics:
            logger.info(f"Point Distance MAE:  {self.best_metrics['val_distance_mae']:>10.3f} km")
        if 'val_total_distance_mae' in self.best_metrics:
            logger.info(f"Total Distance MAE:  {self.best_metrics['val_total_distance_mae']:>10.3f} km")
        if 'val_bird_distance_mae' in self.best_metrics:
            logger.info(f"Bird Distance MAE:   {self.best_metrics['val_bird_distance_mae']:>10.3f} km")
        
        logger.info("=" * 60)
    
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
            
            # Extract key metrics for quick access (only include metrics that exist)
            model_info['key_metrics'] = {
                'best_val_loss': self.best_metrics.get('val_loss'),
                'best_mse': self.best_metrics.get('val_mse'),
                'best_kl_loss': self.best_metrics.get('val_kl_loss'),
                'best_speed_mae': self.best_metrics.get('val_speed_mae'),
                'best_distance_mae': self.best_metrics.get('val_distance_mae'),
                'best_total_distance_mae': self.best_metrics.get('val_total_distance_mae'),
                'best_bird_distance_mae': self.best_metrics.get('val_bird_distance_mae'),
                'best_epoch': self.best_metrics.get('epoch')
            }
            
            with open(model_info_path, "w") as f:
                json.dump(model_info, f, indent=2)


class EarlyStoppingTracker(Callback):
    """Track early stopping and learning rate reduction events."""
    
    def __init__(self):
        super().__init__()
        self.early_stopping_wait = 0
        self.lr_plateau_wait = 0
        self.last_lr = None
        
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Track early stopping and LR plateau progress."""
        # Track early stopping patience
        for callback in trainer.callbacks:
            if hasattr(callback, 'wait_count'):  # EarlyStopping callback
                if callback.wait_count != self.early_stopping_wait:
                    self.early_stopping_wait = callback.wait_count
                    if self.early_stopping_wait > 0:
                        remaining = callback.patience - self.early_stopping_wait
                        logger.info(f"Early Stopping: {self.early_stopping_wait}/{callback.patience} "
                                  f"(stopping in {remaining} epochs if no improvement)")
                    else:
                        logger.info("Early Stopping: Validation loss improved - counter reset")
        
        # Track learning rate changes
        current_lr = trainer.optimizers[0].param_groups[0]['lr']
        if self.last_lr is not None and current_lr != self.last_lr:
            logger.info(f"Learning Rate Reduced: {self.last_lr:.2e} → {current_lr:.2e}")
        self.last_lr = current_lr
        
        # Track LR scheduler patience (if ReduceLROnPlateau)
        for scheduler_config in trainer.lr_scheduler_configs:
            scheduler = scheduler_config.scheduler
            if hasattr(scheduler, 'num_bad_epochs'):  # ReduceLROnPlateau
                if scheduler.num_bad_epochs != self.lr_plateau_wait:
                    self.lr_plateau_wait = scheduler.num_bad_epochs
                    if self.lr_plateau_wait > 0:
                        remaining = scheduler.patience - self.lr_plateau_wait
                        logger.info(f"LR Plateau: {self.lr_plateau_wait}/{scheduler.patience} "
                                  f"(reducing in {remaining} epochs if no improvement)")
                    else:
                        logger.info("LR Plateau: Validation loss improved - counter reset")