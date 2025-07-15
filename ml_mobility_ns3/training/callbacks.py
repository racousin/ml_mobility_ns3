import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import torch
import json
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from typing import Dict, Any, List
import numpy as np

logger = logging.getLogger(__name__)


class BestMetricsTracker(Callback):
    """Track best metrics when validation loss improves - supports multiple VAE architectures."""
    
    def __init__(self, experiment_dir: Path, monitor: str = 'val_loss'):
        super().__init__()
        self.experiment_dir = experiment_dir
        self.monitor = monitor
        self.best_metrics = {}
        self.best_monitor_value = float('inf')
        
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Capture all metrics when monitored metric improves."""
        metrics = trainer.callback_metrics
        
        # Get current monitored metric value
        monitor_value = metrics.get(self.monitor, float('inf'))
        if isinstance(monitor_value, torch.Tensor):
            monitor_value = monitor_value.item()
        
        # If monitored metric improved, capture all metrics
        if monitor_value < self.best_monitor_value:
            self.best_monitor_value = monitor_value
            
            # Core metrics (always available)
            core_metrics = [
                'val_loss', 'val_recon_loss', 'val_mse', 'val_speed_mae', 
                'val_distance_mae', 'val_total_distance_mae', 'val_bird_distance_mae'
            ]
            
            # Standard VAE metrics
            standard_vae_metrics = [
                'val_kl_loss', 'val_weighted_kl_loss'
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
                                 distance_aware_metrics + 
                                  speed_aware_metrics)
            
            # Initialize best metrics dict
            self.best_metrics = {
                'epoch': trainer.current_epoch,
                'step': trainer.global_step,
                'monitored_metric': self.monitor,
                'best_monitor_value': monitor_value
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
            logger.info(f"New best {self.monitor}: {monitor_value:.6f} at epoch {trainer.current_epoch}")
            self._log_formatted_metrics()
        
    def _detect_model_type(self) -> str:
        """Detect which type of VAE we're using based on available metrics."""
        if 'val_kl_loss' in self.best_metrics:
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

        elif model_type == 'standard':
            self._log_standard_kl_losses()
        
        # Additional loss components
        self._log_additional_losses()
        
        logger.info("-" * 70)
        # Evaluation metrics (in interpretable units)
        self._log_evaluation_metrics()
        logger.info("=" * 70)
    
    
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
            
            # Extract key metrics for quick access (include standard)
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

class BetaSchedulingMonitor(Callback):
    """Monitor and visualize beta scheduling during training."""
    
    def __init__(self, experiment_dir: Path, log_frequency: int = 10):
        super().__init__()
        self.experiment_dir = experiment_dir
        self.log_frequency = log_frequency
        
        # Track beta values over time
        self.beta_history = []
        self.epoch_history = []
        self.step_history = []
        
        # Track KL and reconstruction losses
        self.kl_loss_history = []
        self.recon_loss_history = []
        self.total_loss_history = []
        
        
    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, 
                           outputs: Dict, batch: Any, batch_idx: int):
        """Track beta values during training."""
        # Only log at specified frequency
        if trainer.global_step % self.log_frequency != 0:
            return
        
        # Get metrics from trainer
        metrics = trainer.callback_metrics
        
        # Track epoch and step
        self.epoch_history.append(trainer.current_epoch)
        self.step_history.append(trainer.global_step)
        
        # Track beta values
        if 'train_beta' in metrics:
            beta = metrics['train_beta'].item() if isinstance(metrics['train_beta'], torch.Tensor) else metrics['train_beta']
            self.beta_history.append(beta)
        # Track losses
        if 'train_kl_loss' in metrics:
            kl_loss = metrics['train_kl_loss'].item() if isinstance(metrics['train_kl_loss'], torch.Tensor) else metrics['train_kl_loss']
            self.kl_loss_history.append(kl_loss)
        
        if 'train_recon_loss' in metrics:
            recon_loss = metrics['train_recon_loss'].item() if isinstance(metrics['train_recon_loss'], torch.Tensor) else metrics['train_recon_loss']
            self.recon_loss_history.append(recon_loss)
        
        if 'train_loss' in metrics:
            total_loss = metrics['train_loss'].item() if isinstance(metrics['train_loss'], torch.Tensor) else metrics['train_loss']
            self.total_loss_history.append(total_loss)
        
    
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Log beta scheduling progress at validation time."""
        current_epoch = trainer.current_epoch
        
        # Log every 10 epochs
        if current_epoch % 10 == 0 and self.beta_history:
            logger.info(f"\n{'='*60}")
            logger.info(f"BETA SCHEDULING STATUS (Epoch {current_epoch})")
            logger.info(f"{'='*60}")
            
            if self.beta_history:
                current_beta = self.beta_history[-1]
                logger.info(f"Current Beta: {current_beta:.4f}")
                logger.info(f"Beta range: [{min(self.beta_history):.4f}, {max(self.beta_history):.4f}]")
            # Check if beta scheduling is complete
            if len(self.beta_history) > 10:
                recent_betas = self.beta_history[-10:]
                if max(recent_betas) - min(recent_betas) < 0.001:
                    logger.info("✓ Beta scheduling appears to be complete (stable values)")
                else:
                    logger.info("→ Beta scheduling still in progress")
            
            logger.info(f"{'='*60}\n")
    
    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Create visualization of beta scheduling and save data."""
        if not self.step_history:
            return
        
        # Save raw data
        beta_data = {
            'steps': self.step_history,
            'epochs': self.epoch_history,
            'beta': self.beta_history,
            'kl_loss': self.kl_loss_history,
            'recon_loss': self.recon_loss_history,
            'total_loss': self.total_loss_history,
        }
        
        data_file = self.experiment_dir / "beta_scheduling_data.json"
        with open(data_file, "w") as f:
            json.dump(beta_data, f, indent=2)
        
        # Create visualizations
        self._create_beta_plots()
        
        logger.info(f"Beta scheduling data saved to: {data_file}")
        logger.info(f"Beta scheduling plots saved to: {self.experiment_dir / 'beta_scheduling.png'}")
    
    def _create_beta_plots(self):
        """Create visualization plots for beta scheduling."""
        if not self.step_history:
            return
        
        # Determine number of subplots based on available data
        n_plots = 1  # At least beta plot
        if self.kl_loss_history:
            n_plots += 1  # Add loss plot
        
        fig, axes = plt.subplots(n_plots, 1, figsize=(10, 4 * n_plots))
        if n_plots == 1:
            axes = [axes]
        
        plot_idx = 0
        
        # Plot 1: Beta scheduling
        ax = axes[plot_idx]
        if self.beta_history:
            ax.plot(self.step_history[:len(self.beta_history)], self.beta_history, 
                   'b-', linewidth=2, label='Beta')
            ax.set_ylabel('Beta Value', fontsize=12)
            ax.set_title('Beta Scheduling During Training', fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.legend()

        ax.set_xlabel('Training Step', fontsize=12)
        
        plot_idx += 1
        
        # Plot 2: Loss components (if available)
        if self.kl_loss_history and plot_idx < len(axes):
            ax = axes[plot_idx]
            
            # Normalize losses for better visualization
            steps = self.step_history[:len(self.total_loss_history)]
            ax.plot(steps, self.total_loss_history, 'k-', linewidth=2, label='Total Loss')
            ax.plot(steps[:len(self.recon_loss_history)], self.recon_loss_history, 
                   'b--', linewidth=2, label='Recon Loss')
            
            if self.beta_history and self.kl_loss_history:
                # Plot weighted KL loss
                weighted_kl = [beta * kl for beta, kl in zip(self.beta_history[:len(self.kl_loss_history)], 
                                                            self.kl_loss_history)]
                ax.plot(steps[:len(weighted_kl)], weighted_kl, 
                       'r--', linewidth=2, label='Weighted KL Loss')
            
            ax.set_xlabel('Training Step', fontsize=12)
            ax.set_ylabel('Loss', fontsize=12)
            ax.set_title('Loss Components During Training', fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_yscale('log')  # Log scale for better visualization
            
            plot_idx += 1
        
        
        plt.tight_layout()
        plt.savefig(self.experiment_dir / 'beta_scheduling.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create a summary plot of beta evolution by epoch
        if self.epoch_history and self.beta_history:
            self._create_epoch_summary_plot()
    
    def _create_epoch_summary_plot(self):
        """Create a summary plot showing beta evolution by epoch."""
        # Group by epoch
        epoch_betas = {}
        for epoch, beta in zip(self.epoch_history[:len(self.beta_history)], self.beta_history):
            if epoch not in epoch_betas:
                epoch_betas[epoch] = []
            epoch_betas[epoch].append(beta)
        
        # Calculate mean beta per epoch
        epochs = sorted(epoch_betas.keys())
        mean_betas = [np.mean(epoch_betas[e]) for e in epochs]
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, mean_betas, 'b-', linewidth=3, marker='o', markersize=6)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Mean Beta Value', fontsize=12)
        plt.title('Beta Evolution by Epoch', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # Add annotations for key milestones
        if len(mean_betas) > 0:
            # Mark when beta reaches 50% of max
            max_beta = max(mean_betas)
            half_beta_idx = next((i for i, b in enumerate(mean_betas) if b >= 0.5 * max_beta), None)
            if half_beta_idx is not None:
                plt.annotate(f'50% of max\n(epoch {epochs[half_beta_idx]})', 
                           xy=(epochs[half_beta_idx], mean_betas[half_beta_idx]),
                           xytext=(epochs[half_beta_idx] + 5, mean_betas[half_beta_idx] - 0.1),
                           arrowprops=dict(arrowstyle='->', color='red'),
                           fontsize=10)
            
            # Mark when beta reaches 90% of max
            near_max_idx = next((i for i, b in enumerate(mean_betas) if b >= 0.9 * max_beta), None)
            if near_max_idx is not None and near_max_idx != half_beta_idx:
                plt.annotate(f'90% of max\n(epoch {epochs[near_max_idx]})', 
                           xy=(epochs[near_max_idx], mean_betas[near_max_idx]),
                           xytext=(epochs[near_max_idx] + 5, mean_betas[near_max_idx] + 0.05),
                           arrowprops=dict(arrowstyle='->', color='green'),
                           fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.experiment_dir / 'beta_evolution_by_epoch.png', dpi=300, bbox_inches='tight')
        plt.close()


class FreeBitsMonitor(Callback):
    """Monitor free bits constraint effectiveness."""
    
    def __init__(self, experiment_dir: Path):
        super().__init__()
        self.experiment_dir = experiment_dir
        self.free_bits_violations = []
        self.epochs = []
        
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Check if free bits constraint is being satisfied."""
        metrics = trainer.callback_metrics
        current_epoch = trainer.current_epoch
        
        # Check if free bits is being used
        loss_config = pl_module.config.training.loss.get('params', {})
        free_bits_config = loss_config.get('free_bits', {})
        
        if not free_bits_config.get('enabled', False):
            return
        
        lambda_free_bits = free_bits_config.get('lambda_free_bits', 2.0)
        
        # Calculate if constraint is violated
        if 'val_kl_loss' in metrics:
            kl_loss = metrics['val_kl_loss'].item() if isinstance(metrics['val_kl_loss'], torch.Tensor) else metrics['val_kl_loss']
            latent_dim = pl_module.model.latent_dim
            kl_per_dim = kl_loss / latent_dim
            
            if kl_per_dim < lambda_free_bits:
                violation = lambda_free_bits - kl_per_dim
                self.free_bits_violations.append(violation)
                self.epochs.append(current_epoch)
                
                if current_epoch % 10 == 0:
                    logger.info(f"Free bits constraint active: KL/dim = {kl_per_dim:.3f} < {lambda_free_bits}")
                    logger.info(f"Constraint violation: {violation:.3f} nats")
    
    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Save free bits monitoring data."""
        if self.free_bits_violations:
            data = {
                'epochs': self.epochs,
                'violations': self.free_bits_violations,
                'lambda_free_bits': pl_module.config.training.loss.params.get('free_bits', {}).get('lambda_free_bits', 2.0)
            }
            
            with open(self.experiment_dir / "free_bits_monitoring.json", "w") as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Free bits monitoring data saved to: {self.experiment_dir / 'free_bits_monitoring.json'}")