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