# ml_mobility_ns3/training/lightning_module.py
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from typing import Dict, Any
from hydra.utils import instantiate
from omegaconf import OmegaConf  # Add this import
import logging

from ..metrics.trajectory_metrics import TrajectoryMetrics

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
        
        self.beta = config.training.beta
        self.lambda_dist = config.training.get('lambda_dist', 0.5)
        self.metrics = TrajectoryMetrics()
        self.save_hyperparameters()
        
    def forward(self, x, transport_mode, length, mask=None):
        return self.model(x, transport_mode, length, mask)
    
    def compute_loss(self, batch, outputs):
        x, mask, transport_mode, length = batch
        recon = outputs['recon']
        mu = outputs['mu']
        logvar = outputs['logvar']
        
        # Ensure numerical stability
        logvar = torch.clamp(logvar, min=-10, max=10)
        
        # Reconstruction loss
        mask_expanded = mask.unsqueeze(-1).expand_as(x)
        diff = (recon - x) ** 2
        masked_diff = diff * mask_expanded
        num_valid = mask_expanded.sum()
        recon_loss = masked_diff.sum() / (num_valid + 1e-8)
        
        # Distance preservation loss
        metrics_pred = self.metrics.compute_gps_metrics_torch(recon, mask)
        metrics_target = self.metrics.compute_gps_metrics_torch(x, mask)
        
        distance_loss = F.mse_loss(metrics_pred['total_distance'], metrics_target['total_distance'])
        
        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss
        loss = recon_loss + self.beta * kl_loss + self.lambda_dist * distance_loss
        
        metrics_dict = {
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'distance_loss': distance_loss,
            'total_loss': loss
        }
        
        return loss, metrics_dict
    
    def training_step(self, batch, batch_idx):
        outputs = self.forward(*batch)
        loss, metrics = self.compute_loss(batch, outputs)
        
        # Compute trajectory metrics
        traj_metrics = self.metrics.compute_trajectory_mae(
            outputs['recon'], batch[0], batch[1]
        )
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_recon_loss', metrics['recon_loss'])
        self.log('train_kl_loss', metrics['kl_loss'])
        self.log('train_distance_loss', metrics['distance_loss'])
        
        for key, value in traj_metrics.items():
            self.log(f'train_{key}', value)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self.forward(*batch)
        loss, metrics = self.compute_loss(batch, outputs)
        
        # Compute trajectory metrics
        traj_metrics = self.metrics.compute_trajectory_mae(
            outputs['recon'], batch[0], batch[1]
        )
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_recon_loss', metrics['recon_loss'])
        self.log('val_kl_loss', metrics['kl_loss'])
        self.log('val_distance_loss', metrics['distance_loss'])
        
        for key, value in traj_metrics.items():
            self.log(f'val_{key}', value)
        
        return loss
    
    def configure_optimizers(self):
        # Check if model has parameters
        params = list(self.model.parameters())
        if not params:
            raise ValueError("Model has no parameters to optimize")
        
        optimizer = torch.optim.AdamW(
            params,  # Use explicit params list
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