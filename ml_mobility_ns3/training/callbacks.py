import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import torch
import json
from pathlib import Path
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class BestMetricsTracker(Callback):
    """Track best metrics when validation loss improves - supports multiple VAE architectures."""
    
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
            
            # Core metrics (always available)
            core_metrics = [
                'val_loss', 'val_recon_loss', 'val_mse', 'val_speed_mae', 
                'val_distance_mae', 'val_total_distance_mae', 'val_bird_distance_mae'
            ]
            
            # Standard VAE metrics
            standard_vae_metrics = [
                'val_kl_loss', 'val_weighted_kl_loss'
            ]
            
            # Hierarchical VAE metrics
            hierarchical_vae_metrics = [
                'val_local_kl_loss', 'val_global_kl_loss', 
                'val_weighted_local_kl', 'val_weighted_global_kl'
            ]
            
            # Loss-specific metrics
            distance_aware_metrics = [
                'val_distance_loss', 'val_weighted_distance_loss'
            ]
            
            speed_aware_metrics = [
                'val_speed_loss', 'val_smoothness_loss', 'val_weighted_speed_loss'
            ]
            
            # Combine all possible metrics
            all_possible_metrics = (core_metrics + standard_vae_metrics + 
                                  hierarchical_vae_metrics + distance_aware_metrics + 
                                  speed_aware_metrics)
            
            # Initialize best metrics dict
            self.best_metrics = {
                'epoch': trainer.current_epoch,
                'step': trainer.global_step,
                'val_loss': val_loss
            }
            
            # Add metrics that actually exist
            for key in all_possible_metrics:
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
    
    def _detect_model_type(self) -> str:
        """Detect which type of VAE we're using based on available metrics."""
        if 'val_local_kl_loss' in self.best_metrics and 'val_global_kl_loss' in self.best_metrics:
            return 'hierarchical'
        elif 'val_kl_loss' in self.best_metrics:
            return 'standard'
        else:
            return 'unknown'
    
    def _log_formatted_metrics(self):
        """Log metrics in a clear, formatted way based on model type."""
        model_type = self._detect_model_type()
        
        logger.info("=" * 70)
        logger.info(f"BEST VALIDATION METRICS ({model_type.upper()} VAE)")
        logger.info("=" * 70)
        logger.info(f"Epoch: {self.best_metrics.get('epoch', 'N/A'):>8}")
        logger.info(f"Step:  {self.best_metrics.get('step', 'N/A'):>8}")
        logger.info("-" * 70)
        
        # Core loss metrics
        if 'val_loss' in self.best_metrics:
            logger.info(f"Total Validation Loss:   {self.best_metrics['val_loss']:>10.6f}")
        if 'val_recon_loss' in self.best_metrics:
            logger.info(f"Reconstruction Loss:     {self.best_metrics['val_recon_loss']:>10.6f}")
        
        # Model-specific KL losses
        if model_type == 'hierarchical':
            self._log_hierarchical_kl_losses()
        elif model_type == 'standard':
            self._log_standard_kl_losses()
        
        # Additional loss components
        self._log_additional_losses()
        
        logger.info("-" * 70)
        # Evaluation metrics (in interpretable units)
        self._log_evaluation_metrics()
        logger.info("=" * 70)
    
    def _log_hierarchical_kl_losses(self):
        """Log KL losses for hierarchical VAE."""
        logger.info("KL Divergence Components:")
        if 'val_local_kl_loss' in self.best_metrics:
            logger.info(f"  Local KL (segments):   {self.best_metrics['val_local_kl_loss']:>10.6f}")
        if 'val_global_kl_loss' in self.best_metrics:
            logger.info(f"  Global KL (trajectory):{self.best_metrics['val_global_kl_loss']:>10.6f}")
        if 'val_weighted_local_kl' in self.best_metrics:
            logger.info(f"  Weighted Local KL:     {self.best_metrics['val_weighted_local_kl']:>10.6f}")
        if 'val_weighted_global_kl' in self.best_metrics:
            logger.info(f"  Weighted Global KL:    {self.best_metrics['val_weighted_global_kl']:>10.6f}")
    
    def _log_standard_kl_losses(self):
        """Log KL losses for standard VAE."""
        if 'val_kl_loss' in self.best_metrics:
            logger.info(f"KL Divergence:           {self.best_metrics['val_kl_loss']:>10.6f}")
        if 'val_weighted_kl_loss' in self.best_metrics:
            logger.info(f"Weighted KL Loss:        {self.best_metrics['val_weighted_kl_loss']:>10.6f}")
    
    def _log_additional_losses(self):
        """Log additional loss components based on loss function type."""
        # Distance-aware loss components
        if 'val_distance_loss' in self.best_metrics:
            logger.info(f"Distance Preservation:   {self.best_metrics['val_distance_loss']:>10.6f}")
        if 'val_weighted_distance_loss' in self.best_metrics:
            logger.info(f"Weighted Distance Loss:  {self.best_metrics['val_weighted_distance_loss']:>10.6f}")
        
        # Speed-aware loss components
        if 'val_speed_loss' in self.best_metrics:
            logger.info(f"Speed Consistency:       {self.best_metrics['val_speed_loss']:>10.6f}")
        if 'val_smoothness_loss' in self.best_metrics:
            logger.info(f"Speed Smoothness:        {self.best_metrics['val_smoothness_loss']:>10.6f}")
        if 'val_weighted_speed_loss' in self.best_metrics:
            logger.info(f"Weighted Speed Loss:     {self.best_metrics['val_weighted_speed_loss']:>10.6f}")
    
    def _log_evaluation_metrics(self):
        """Log evaluation metrics in interpretable units."""
        logger.info("Evaluation Metrics:")
        if 'val_mse' in self.best_metrics:
            logger.info(f"  MSE (scaled):          {self.best_metrics['val_mse']:>10.6f}")
        if 'val_speed_mae' in self.best_metrics:
            logger.info(f"  Speed MAE (km/h):      {self.best_metrics['val_speed_mae']:>10.3f}")
        if 'val_distance_mae' in self.best_metrics:
            logger.info(f"  Point Distance MAE:    {self.best_metrics['val_distance_mae']:>10.3f} km")
        if 'val_total_distance_mae' in self.best_metrics:
            logger.info(f"  Total Distance MAE:    {self.best_metrics['val_total_distance_mae']:>10.3f} km")
        if 'val_bird_distance_mae' in self.best_metrics:
            logger.info(f"  Bird Distance MAE:     {self.best_metrics['val_bird_distance_mae']:>10.3f} km")
    
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
            
            # Extract key metrics for quick access (include both standard and hierarchical)
            key_metrics = {
                'best_val_loss': self.best_metrics.get('val_loss'),
                'best_mse': self.best_metrics.get('val_mse'),
                'best_speed_mae': self.best_metrics.get('val_speed_mae'),
                'best_distance_mae': self.best_metrics.get('val_distance_mae'),
                'best_total_distance_mae': self.best_metrics.get('val_total_distance_mae'),
                'best_bird_distance_mae': self.best_metrics.get('val_bird_distance_mae'),
                'best_epoch': self.best_metrics.get('epoch')
            }
            
            # Add model-specific KL metrics
            if 'val_kl_loss' in self.best_metrics:
                # Standard VAE
                key_metrics['best_kl_loss'] = self.best_metrics.get('val_kl_loss')
            
            if 'val_local_kl_loss' in self.best_metrics:
                # Hierarchical VAE
                key_metrics['best_local_kl_loss'] = self.best_metrics.get('val_local_kl_loss')
                key_metrics['best_global_kl_loss'] = self.best_metrics.get('val_global_kl_loss')
            
            # Add loss-specific metrics if they exist
            if 'val_distance_loss' in self.best_metrics:
                key_metrics['best_distance_loss'] = self.best_metrics.get('val_distance_loss')
            if 'val_speed_loss' in self.best_metrics:
                key_metrics['best_speed_loss'] = self.best_metrics.get('val_speed_loss')
            
            model_info['key_metrics'] = key_metrics
            
            with open(model_info_path, "w") as f:
                json.dump(model_info, f, indent=2)


class EarlyStoppingTracker(Callback):
    """Track early stopping and learning rate reduction events - enhanced for different model types."""
    
    def __init__(self):
        super().__init__()
        self.early_stopping_wait = 0
        self.lr_plateau_wait = 0
        self.last_lr = None
        self.last_log_epoch = -1
        
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Track early stopping and LR plateau progress."""
        current_epoch = trainer.current_epoch
        
        # Only log detailed info every 5 epochs or when something changes
        should_log_detailed = (current_epoch % 5 == 0 or 
                             current_epoch != self.last_log_epoch)
        
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
                elif should_log_detailed and self.early_stopping_wait > 0:
                    # Periodic update
                    remaining = callback.patience - self.early_stopping_wait
                    logger.info(f"Early Stopping Progress: {self.early_stopping_wait}/{callback.patience} "
                              f"epochs without improvement")
        
        # Track learning rate changes
        current_lr = trainer.optimizers[0].param_groups[0]['lr']
        if self.last_lr is not None and current_lr != self.last_lr:
            logger.info(f"Learning Rate Reduced: {self.last_lr:.2e} → {current_lr:.2e}")
        elif should_log_detailed:
            logger.info(f"Current Learning Rate: {current_lr:.2e}")
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
                elif should_log_detailed and self.lr_plateau_wait > 0:
                    # Periodic update
                    remaining = scheduler.patience - self.lr_plateau_wait
                    logger.info(f"LR Plateau Progress: {self.lr_plateau_wait}/{scheduler.patience} "
                              f"epochs without improvement")
        
        self.last_log_epoch = current_epoch


class HierarchicalVAEMetricsLogger(Callback):
    """Special callback for logging hierarchical VAE metrics during training."""
    
    def __init__(self):
        super().__init__()
        self.epoch_local_kl = []
        self.epoch_global_kl = []
        
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Log hierarchical-specific metrics."""
        metrics = trainer.callback_metrics
        current_epoch = trainer.current_epoch
        
        # Only for hierarchical VAE
        if 'val_local_kl_loss' in metrics and 'val_global_kl_loss' in metrics:
            local_kl = metrics['val_local_kl_loss']
            global_kl = metrics['val_global_kl_loss']
            
            if isinstance(local_kl, torch.Tensor):
                local_kl = local_kl.item()
            if isinstance(global_kl, torch.Tensor):
                global_kl = global_kl.item()
            
            self.epoch_local_kl.append(local_kl)
            self.epoch_global_kl.append(global_kl)
            
            # Log balance between local and global KL
            kl_ratio = local_kl / (global_kl + 1e-8)
            
            # Log every 10 epochs
            if current_epoch % 10 == 0:
                logger.info(f"\nHierarchical VAE Analysis (Epoch {current_epoch}):")
                logger.info(f"  Local KL:  {local_kl:.6f}")
                logger.info(f"  Global KL: {global_kl:.6f}")
                logger.info(f"  Ratio (L/G): {kl_ratio:.3f}")
                
                # Provide interpretation
                if kl_ratio > 3.0:
                    logger.info("  → Local patterns dominating (consider reducing beta_local)")
                elif kl_ratio < 0.3:
                    logger.info("  → Global structure dominating (consider reducing beta_global)")
                else:
                    logger.info("  → Good balance between local and global representations")
    
    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Save hierarchical analysis."""
        if self.epoch_local_kl and self.epoch_global_kl:
            # Save KL evolution
            analysis = {
                'local_kl_evolution': self.epoch_local_kl,
                'global_kl_evolution': self.epoch_global_kl,
                'final_local_kl': self.epoch_local_kl[-1],
                'final_global_kl': self.epoch_global_kl[-1],
                'final_ratio': self.epoch_local_kl[-1] / (self.epoch_global_kl[-1] + 1e-8)
            }
            
            # Find experiment directory through the trainer
            exp_dir = None
            for callback in trainer.callbacks:
                if isinstance(callback, BestMetricsTracker):
                    exp_dir = callback.experiment_dir
                    break
            
            if exp_dir:
                analysis_file = exp_dir / "hierarchical_analysis.json"
                with open(analysis_file, "w") as f:
                    json.dump(analysis, f, indent=2)
                logger.info(f"Hierarchical analysis saved to: {analysis_file}")