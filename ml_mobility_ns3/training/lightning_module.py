# ml_mobility_ns3/training/lightning_module_cleaned.py
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple
from hydra.utils import instantiate
from omegaconf import OmegaConf
import logging

from ml_mobility_ns3.metrics.diff_metrics import DiffMetrics
from .losses import create_loss, BaseLoss

logger = logging.getLogger(__name__)


class TrajectoryLightningModule(pl.LightningModule):
    """Lightning module for trajectory generation models."""
    
    def __init__(self, config, skip_loss_init=False):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        
        # Initialize model
        self._init_model()
        
        # Load pretrained weights if specified
        self._load_pretrained_weights()
        
        # Initialize loss function only if not skipping (for evaluation mode)
        if not skip_loss_init:
            self._init_loss()
        else:
            self.loss_fn = None
            logger.info("Skipping loss initialization (evaluation mode)")
        
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
    
    def _load_pretrained_weights(self):
        """Load pretrained weights from a previous experiment if specified."""
        if hasattr(self.config.training, 'pretrained_checkpoint') and self.config.training.pretrained_checkpoint:
            checkpoint_path = self.config.training.pretrained_checkpoint
            logger.info(f"Loading pretrained weights from: {checkpoint_path}")
            
            try:
                # Load only the model weights, not the full Lightning checkpoint
                # This avoids loading training configurations we don't want
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                
                # Extract only the model state dict from Lightning checkpoint
                if 'state_dict' in checkpoint:
                    # Lightning checkpoint format - extract only model weights
                    full_state_dict = checkpoint['state_dict']
                    # Filter to get only model parameters (remove Lightning module overhead)
                    state_dict = {}
                    for key, value in full_state_dict.items():
                        if key.startswith('model.'):
                            # Remove 'model.' prefix to match the actual model structure
                            new_key = key[6:]  # Remove 'model.' prefix
                            state_dict[new_key] = value
                else:
                    # Direct state dict
                    state_dict = checkpoint
                
                # Load the state dict
                missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
                
                if missing_keys:
                    logger.warning(f"Missing keys when loading pretrained weights: {missing_keys}")
                if unexpected_keys:
                    logger.warning(f"Unexpected keys when loading pretrained weights: {unexpected_keys}")
                
                logger.info(f"Successfully loaded pretrained weights from {checkpoint_path}")
                
                # Log epoch info if available
                if 'epoch' in checkpoint:
                    logger.info(f"Checkpoint was from epoch: {checkpoint['epoch']}")
                if 'global_step' in checkpoint:
                    logger.info(f"Checkpoint was from global step: {checkpoint['global_step']}")
                    
            except FileNotFoundError:
                logger.error(f"Checkpoint file not found: {checkpoint_path}")
                raise
            except Exception as e:
                logger.error(f"Failed to load pretrained weights: {e}")
                raise
    
    def _init_loss(self):
        """Initialize loss function from config."""
        loss_config = self.config.training.loss
        self.loss_fn = create_loss(OmegaConf.to_container(loss_config))
        logger.info(f"Using loss function: {loss_config.type}")
        
        # Log beta scheduling info if applicable
        loss_params = loss_config.get('params', {})
        if 'beta' in loss_params and isinstance(loss_params['beta'], dict):
            logger.info(f"Beta scheduling: {loss_params['beta']['type']}")
        
        # Log free bits info if enabled
        if 'free_bits' in loss_params and loss_params['free_bits'].get('enabled', False):
            logger.info(f"Free bits enabled with lambda={loss_params['free_bits'].get('lambda_free_bits', 2.0)}")
    
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
        
        # Update loss function with current step/epoch for beta scheduling
        self.loss_fn.update_step(self.global_step, self.current_epoch)
        
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
        """Log all metrics with proper prefix - beta values."""
        # Log main loss
        self.log(f'{prefix}_loss', loss, prog_bar=True)
        
        # Log loss components - handle different VAE types
        for key, value in loss_components.items():
            if key != 'total':
                self.log(f'{prefix}_{key}', value)
        
        # Log standardized metrics
        for key, value in metrics.items():
            self.log(f'{prefix}_{key}', value)
        
        # Special handling for beta values (for monitoring annealing)
        if 'beta' in loss_components:
            self.log(f'{prefix}_beta', loss_components['beta'], prog_bar=False)
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        x, mask, transport_mode, length = self._prepare_batch(batch)
        
        # Forward pass
        outputs = self.forward(x, transport_mode, length, mask)
        
        # Compute loss
        loss, loss_components = self._compute_loss(outputs, batch)
        
        # Compute metrics
        metrics = self._compute_metrics(outputs, batch)
        
        # Log everything
        self._log_metrics(loss, loss_components, metrics, prefix='train')
        
        # Log adaptive scheduler status if available
        if 'epochs_without_improvement' in loss_components:
            self.log('train_epochs_without_improvement', 
                    loss_components['epochs_without_improvement'], prog_bar=False)
        if 'scheduler_converged' in loss_components:
            self.log('train_scheduler_converged', 
                    float(loss_components['scheduler_converged']), prog_bar=False)
        
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
            'metrics': metrics,
            'loss_components': loss_components  # Store for beta tracking
        })
        
        return loss
    
    def on_validation_epoch_end(self):
        """Aggregate validation metrics at epoch end - enhanced for beta scheduling."""
        if not self._validation_outputs:
            return
        
        # Calculate average validation loss for adaptive scheduler
        avg_val_loss = sum(out['loss'].item() for out in self._validation_outputs) / len(self._validation_outputs)
        
        # Update adaptive scheduler with epoch-level validation loss (same as what model checkpointing uses)
        if hasattr(self.loss_fn, 'update_adaptive_scheduler_epoch'):
            # Use the actual validation loss (weighted) for consistency with model checkpointing
            self.loss_fn.update_adaptive_scheduler_epoch(self.current_epoch, avg_val_loss)
            
            # Log additional info if components are available
            if 'loss_components' in self._validation_outputs[0]:
                if 'recon_loss' in self._validation_outputs[0]['loss_components'] and 'kl_loss' in self._validation_outputs[0]['loss_components']:
                    avg_recon_loss = sum(out['loss_components']['recon_loss'].item() for out in self._validation_outputs) / len(self._validation_outputs)
                    avg_kl_loss = sum(out['loss_components']['kl_loss'].item() for out in self._validation_outputs) / len(self._validation_outputs)
                    complete_loss = avg_recon_loss + avg_kl_loss
                    
                    # Get detailed reconstruction breakdown if available
                    recon_breakdown = []
                    if 'recon_weighted_coord' in self._validation_outputs[0]['loss_components']:
                        avg_coord = sum(out['loss_components']['recon_weighted_coord'].item() for out in self._validation_outputs) / len(self._validation_outputs)
                        recon_breakdown.append(f"coord:{avg_coord:.3f}")
                    if 'recon_weighted_point' in self._validation_outputs[0]['loss_components']:
                        avg_point = sum(out['loss_components']['recon_weighted_point'].item() for out in self._validation_outputs) / len(self._validation_outputs)
                        recon_breakdown.append(f"point:{avg_point:.3f}")
                    if 'recon_weighted_consecutive' in self._validation_outputs[0]['loss_components']:
                        avg_consec = sum(out['loss_components']['recon_weighted_consecutive'].item() for out in self._validation_outputs) / len(self._validation_outputs)
                        recon_breakdown.append(f"consec:{avg_consec:.3f}")
                    if 'recon_weighted_cumulative' in self._validation_outputs[0]['loss_components']:
                        avg_cumul = sum(out['loss_components']['recon_weighted_cumulative'].item() for out in self._validation_outputs) / len(self._validation_outputs)
                        recon_breakdown.append(f"cumul:{avg_cumul:.3f}")
                    
                    if recon_breakdown:
                        breakdown_str = " + ".join(recon_breakdown)
                        logger.info(f"Updated adaptive scheduler with epoch {self.current_epoch} weighted val_loss: {avg_val_loss:.6f}")
                        logger.info(f"  Reconstruction: {avg_recon_loss:.6f} = ({breakdown_str})")
                        logger.info(f"  KL: {avg_kl_loss:.6f}, Complete: {complete_loss:.6f}")
                    else:
                        logger.info(f"Updated adaptive scheduler with epoch {self.current_epoch} weighted val_loss: {avg_val_loss:.6f} (recon: {avg_recon_loss:.6f}, kl: {avg_kl_loss:.6f}, complete: {complete_loss:.6f})")
                else:
                    logger.info(f"Updated adaptive scheduler with epoch {self.current_epoch} val_loss: {avg_val_loss:.6f}")
        
        # Define key metrics to aggregate - VAE metrics
        key_metrics = ['mse', 'speed_mae', 'distance_mae', 'total_distance_mae', 'bird_distance_mae', 'consecutive_distance_mae']
        
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
        
        # Log beta values at epoch end (for monitoring annealing progress)
        if self._validation_outputs and 'loss_components' in self._validation_outputs[-1]:
            last_components = self._validation_outputs[-1]['loss_components']
            if 'beta' in last_components:
                logger.info(f"Current beta: {last_components['beta']:.4f}")

        # Enhanced logging for VAE
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
                elif metric_name == 'consecutive_distance_mae':
                    logger.info(f"Consec Dist MAE (km):{value:>10.3f}")
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
        
        # Check if learning rate scheduling is enabled
        if self.config.training.get('lr_scheduler_enabled', True):
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
                    'monitor': self.config.training.lr_scheduler_monitor
                }
            }
        else:
            # No scheduler
            return optimizer