# ml_mobility_ns3/training/lightning_module.py
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from typing import Dict, Any
from hydra.utils import instantiate
from omegaconf import OmegaConf
import logging

from ..metrics.trajectory_metrics import TrajectoryMetrics
from .losses import create_loss

logger = logging.getLogger(__name__)


class TrajectoryLightningModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Save the model config specifically
        self.model_config = OmegaConf.to_container(config.model)
        
        # Instantiate model with error handling
        try:
            self.model = instantiate(config.model)
            logger.info(f"Successfully instantiated model: {config.model.name}")
            
            # Verify model has parameters
            param_count = sum(p.numel() for p in self.model.parameters())
            if param_count == 0:
                raise ValueError(f"Model {config.model.name} has no learnable parameters")
            logger.info(f"Model has {param_count:,} parameters")
            
        except Exception as e:
            logger.error(f"Failed to instantiate model: {e}")
            raise
        
        # Create loss function from config - now from training.loss
        loss_config = config.training.loss
        self.loss_fn = create_loss(OmegaConf.to_container(loss_config))
        logger.info(f"Using loss function: {loss_config.type}")
        
        # Backward compatibility (can be removed if not needed)
        self.beta = config.training.get('beta', 1.0)
        self.lambda_dist = config.training.get('lambda_dist', 0.5)
        
        self.metrics = TrajectoryMetrics()
        self.save_hyperparameters()
        
    def forward(self, x, transport_mode, length, mask=None):
        return self.model(x, transport_mode, length, mask)
    
    def compute_loss(self, batch, outputs):
        x, mask, transport_mode, length = batch
        
        if mask.dim() == 1:
            batch_size, seq_len, _ = x.shape
            new_mask = torch.zeros(batch_size, seq_len, device=x.device)
            for i in range(batch_size):
                valid_len = int(mask[i].item())
                new_mask[i, :valid_len] = 1.0
            mask = new_mask
        # Prepare targets dict
        targets = {
            'x': x,
            'transport_mode': transport_mode,
            'length': length
        }
        
        # Compute loss using configured loss function
        loss_dict = self.loss_fn(outputs, targets, mask)
        
        return loss_dict['total'], loss_dict
    
    def compute_standard_metrics(self, pred, target, mask):
        """Compute standardized metrics for all models."""
        return self.metrics.compute_comprehensive_metrics(pred, target, mask)
    
    def training_step(self, batch, batch_idx):
        x, mask, transport_mode, length = batch
        outputs = self.forward(x, transport_mode, length, mask)
        loss, loss_components = self.compute_loss(batch, outputs)
        
        # Compute standardized metrics
        std_metrics = self.compute_standard_metrics(outputs['recon'], batch[0], batch[1])
        
        # Log loss components
        self.log('train_loss', loss, prog_bar=True)
        for key, value in loss_components.items():
            if key != 'total':
                self.log(f'train_{key}', value)
        
        # Log standardized metrics
        for key, value in std_metrics.items():
            self.log(f'train_{key}', value)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        
        x, mask, transport_mode, length = batch
        outputs = self.forward(x, transport_mode, length, mask)
        loss, loss_components = self.compute_loss(batch, outputs)
        
        # Compute standardized metrics
        std_metrics = self.compute_standard_metrics(outputs['recon'], batch[0], batch[1])
        
        # Log loss components
        self.log('val_loss', loss, prog_bar=True)
        for key, value in loss_components.items():
            if key != 'total':
                self.log(f'val_{key}', value)
        
        # Log standardized metrics - these are the key metrics for comparison
        for key, value in std_metrics.items():
            self.log(f'val_{key}', value)
        
        # Store metrics for epoch end summary
        self.validation_step_outputs = getattr(self, 'validation_step_outputs', [])
        self.validation_step_outputs.append({
            'loss': loss,
            'metrics': std_metrics
        })
        
        return loss
    
    def on_validation_epoch_end(self):
        """Log epoch-level summaries of key metrics."""
        if hasattr(self, 'validation_step_outputs') and self.validation_step_outputs:
            # Aggregate key metrics
            avg_metrics = {}
            metric_keys = ['mse', 'speed_mse', 'total_distance_mae', 'bird_distance_mae']
            
            for key in metric_keys:
                values = [out['metrics'][key].item() for out in self.validation_step_outputs 
                         if key in out['metrics']]
                if values:
                    avg_metrics[key] = sum(values) / len(values)
            
            # Log summary
            self.log('val_epoch_mse', avg_metrics.get('mse', 0), prog_bar=True)
            
            # Clear outputs
            self.validation_step_outputs.clear()
    
    def configure_optimizers(self):
        # Check if model has parameters
        params = list(self.model.parameters())
        if not params:
            raise ValueError("Model has no parameters to optimize")
        
        optimizer = torch.optim.AdamW(
            params,
            lr=self.config.training.learning_rate,
            weight_decay=1e-5
        )
        
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