# ml_mobility_ns3/training/lightning_module_cleaned.py
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple
from hydra.utils import instantiate
from omegaconf import OmegaConf
import logging

from ml_mobility_ns3.metrics.diff_metrics import DiffMetrics
from .losses import create_loss

logger = logging.getLogger(__name__)


class TrajectoryLightningModule(pl.LightningModule):
    """Lightning module for trajectory generation models."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        
        # Initialize model
        self._init_model()
        
        # Initialize loss function
        self._init_loss()
        
        # Initialize metrics
        self.metrics = DiffMetrics()
        
        # For validation epoch aggregation
        self._validation_outputs = []
        
    def _init_model(self):
        """Initialize the model with error handling."""
        try:
            self.model = instantiate(self.config.model)
            logger.info(f"Successfully instantiated model: {self.config.model.name}")
            
            # Verify model has parameters
            param_count = sum(p.numel() for p in self.model.parameters())
            if param_count == 0:
                raise ValueError(f"Model {self.config.model.name} has no learnable parameters")
            logger.info(f"Model has {param_count:,} parameters")
            
        except Exception as e:
            logger.error(f"Failed to instantiate model: {e}")
            raise
    
    def _init_loss(self):
        """Initialize loss function from config."""
        loss_config = self.config.training.loss
        self.loss_fn = create_loss(OmegaConf.to_container(loss_config))
        logger.info(f"Using loss function: {loss_config.type}")
    
    def forward(self, x, transport_mode, length, mask=None):
        """Forward pass through the model."""
        return self.model(x, transport_mode, length, mask)
    
    def _prepare_batch(self, batch: Tuple) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare batch data and ensure mask has correct dimensions."""
        x, mask, transport_mode, length = batch
        
        # Fix mask dimensions if needed
        if mask.dim() == 1:
            batch_size, seq_len, _ = x.shape
            new_mask = torch.zeros(batch_size, seq_len, device=x.device)
            for i in range(batch_size):
                valid_len = int(mask[i].item())
                new_mask[i, :valid_len] = 1.0
            mask = new_mask
        
        return x, mask, transport_mode, length
    
    def _compute_loss(self, outputs: Dict, batch: Tuple) -> Tuple[torch.Tensor, Dict]:
        """Compute loss using configured loss function."""
        x, mask, transport_mode, length = self._prepare_batch(batch)
        
        # Prepare targets dict
        targets = {
            'x': x,
            'transport_mode': transport_mode,
            'length': length
        }
        
        # Compute loss
        loss_dict = self.loss_fn(outputs, targets, mask)
        
        return loss_dict['total'], loss_dict
    
    def _compute_metrics(self, outputs: Dict, batch: Tuple) -> Dict[str, torch.Tensor]:
        """Compute standardized metrics."""
        x, mask, _, _ = self._prepare_batch(batch)
        return self.metrics.compute_comprehensive_metrics(outputs['recon'], x, mask)
    
    def _log_metrics(self, loss: torch.Tensor, loss_components: Dict, 
                    metrics: Dict, prefix: str = 'train'):
        """Log all metrics with proper prefix - supports hierarchical VAE."""
        # Log main loss
        self.log(f'{prefix}_loss', loss, prog_bar=True)
        
        # Log loss components - handle different VAE types
        for key, value in loss_components.items():
            if key != 'total':
                self.log(f'{prefix}_{key}', value)
        
        # Log standardized metrics
        for key, value in metrics.items():
            self.log(f'{prefix}_{key}', value)
        
        # Special handling for hierarchical VAE progress bar
        if 'local_kl_loss' in loss_components and 'global_kl_loss' in loss_components:
            # Show key hierarchical metrics in progress bar for training
            if prefix == 'train':
                self.log(f'{prefix}_local_kl', loss_components['local_kl_loss'], prog_bar=True)
                self.log(f'{prefix}_global_kl', loss_components['global_kl_loss'], prog_bar=True)
            
            # Calculate and log the local/global KL ratio for analysis
            local_kl = loss_components['local_kl_loss']
            global_kl = loss_components['global_kl_loss']
            kl_ratio = local_kl / (global_kl + 1e-8)
            self.log(f'{prefix}_kl_ratio', kl_ratio)
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        x, mask, transport_mode, length = self.on_validation_epoch_end(batch)
        
        # Forward pass
        outputs = self.forward(x, transport_mode, length, mask)
        
        # Compute loss
        loss, loss_components = self._compute_loss(outputs, batch)
        
        # Compute metrics
        metrics = self._compute_metrics(outputs, batch)
        
        # Log everything
        self._log_metrics(loss, loss_components, metrics, prefix='train')
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        x, mask, transport_mode, length = self._prepare_batch(batch)
        
        # Forward pass
        outputs = self.forward(x, transport_mode, length, mask)
        
        # Compute loss
        loss, loss_components = self._compute_loss(outputs, batch)
        
        # Compute metrics
        metrics = self._compute_metrics(outputs, batch)
        
        # Log everything
        self._log_metrics(loss, loss_components, metrics, prefix='val')
        
        # Store for epoch end aggregation
        self._validation_outputs.append({
            'loss': loss,
            'metrics': metrics
        })
        
        return loss
    
    def on_validation_epoch_end(self):
        """Aggregate validation metrics at epoch end - enhanced for hierarchical VAE."""
        if not self._validation_outputs:
            return
        
        # Define key metrics to aggregate - include hierarchical VAE metrics
        key_metrics = ['mse', 'speed_mae', 'distance_mae', 'total_distance_mae', 'bird_distance_mae']
        
        # Aggregate metrics
        avg_metrics = {}
        for key in key_metrics:
            values = [out['metrics'][key].item() for out in self._validation_outputs 
                    if key in out['metrics']]
            if values:
                avg_metrics[key] = sum(values) / len(values)
        
        # Log epoch summary with better formatting
        if 'mse' in avg_metrics:
            self.log('val_epoch_mse', avg_metrics['mse'], prog_bar=True)
        
        # Log additional epoch metrics for better visibility
        if 'speed_mae' in avg_metrics:
            self.log('val_epoch_speed_mae', avg_metrics['speed_mae'], prog_bar=False)
        if 'total_distance_mae' in avg_metrics:
            self.log('val_epoch_total_dist_mae', avg_metrics['total_distance_mae'], prog_bar=False)
        
        # Enhanced logging for hierarchical VAE
        current_epoch = self.current_epoch
        if current_epoch % 5 == 0 or current_epoch == 0:
            logger.info(f"\n{'='*50}")
            logger.info(f"EPOCH {current_epoch} VALIDATION SUMMARY")
            logger.info(f"{'='*50}")
            for metric_name, value in avg_metrics.items():
                if metric_name == 'mse':
                    logger.info(f"MSE (scaled):        {value:>10.6f}")
                elif metric_name == 'speed_mae':
                    logger.info(f"Speed MAE (km/h):    {value:>10.3f}")
                elif metric_name == 'distance_mae':
                    logger.info(f"Point Dist MAE (km): {value:>10.3f}")
                elif metric_name == 'total_distance_mae':
                    logger.info(f"Total Dist MAE (km): {value:>10.3f}")
                elif metric_name == 'bird_distance_mae':
                    logger.info(f"Bird Dist MAE (km):  {value:>10.3f}")
            logger.info(f"{'='*50}\n")
        
        # Clear outputs for next epoch
        self._validation_outputs.clear()
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        # Check if model has parameters
        params = list(self.model.parameters())
        if not params:
            raise ValueError("Model has no parameters to optimize")
        
        # Create optimizer
        optimizer = torch.optim.AdamW(
            params,
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.get('weight_decay', 1e-5)
        )
        
        # Create scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=self.config.training.lr_scheduler_patience,
            factor=self.config.training.lr_scheduler_factor,
            min_lr=self.config.training.get('lr_min', 1e-6)
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }