import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Union
from abc import ABC, abstractmethod
import logging
import math

logger = logging.getLogger(__name__)


class BetaScheduler(ABC):
    """Base class for beta scheduling strategies."""
    
    @abstractmethod
    def get_beta(self, step: int, epoch: int) -> float:
        """Get beta value for current step/epoch."""
        pass


class ConstantBeta(BetaScheduler):
    """Constant beta value."""
    
    def __init__(self, value: float = 1.0):
        self.value = value
    
    def get_beta(self, step: int, epoch: int) -> float:
        return self.value


class LinearAnnealingBeta(BetaScheduler):
    """Linear annealing from start to end value over specified steps/epochs."""
    
    def __init__(self, start: float = 0.0, end: float = 1.0, 
                 anneal_steps: Optional[int] = None, anneal_epochs: Optional[int] = None):
        self.start = start
        self.end = end
        self.anneal_steps = anneal_steps
        self.anneal_epochs = anneal_epochs
        
        if anneal_steps is None and anneal_epochs is None:
            raise ValueError("Either anneal_steps or anneal_epochs must be specified")
    
    def get_beta(self, step: int, epoch: int) -> float:
        if self.anneal_steps is not None:
            progress = min(1.0, step / self.anneal_steps)
        else:
            progress = min(1.0, epoch / self.anneal_epochs)
        
        return self.start + (self.end - self.start) * progress


class ExponentialAnnealingBeta(BetaScheduler):
    """Exponential annealing from start to end value."""
    
    def __init__(self, start: float = 0.0, end: float = 1.0, 
                 anneal_steps: Optional[int] = None, anneal_epochs: Optional[int] = None,
                 rate: float = 0.999):
        self.start = start
        self.end = end
        self.anneal_steps = anneal_steps
        self.anneal_epochs = anneal_epochs
        self.rate = rate
        
        if anneal_steps is None and anneal_epochs is None:
            raise ValueError("Either anneal_steps or anneal_epochs must be specified")
    
    def get_beta(self, step: int, epoch: int) -> float:
        if self.anneal_steps is not None:
            progress = min(1.0, step / self.anneal_steps)
        else:
            progress = min(1.0, epoch / self.anneal_epochs)
        
        # Exponential interpolation
        if progress >= 1.0:
            return self.end
        
        # Use exponential decay formula
        beta = self.end - (self.end - self.start) * (self.rate ** (progress * 100))
        return max(self.start, min(self.end, beta))


class CyclicalBeta(BetaScheduler):
    """Cyclical beta scheduling (useful for preventing posterior collapse)."""
    
    def __init__(self, min_beta: float = 0.0, max_beta: float = 1.0, 
                 cycle_length: int = 10, mode: str = 'triangle'):
        self.min_beta = min_beta
        self.max_beta = max_beta
        self.cycle_length = cycle_length
        self.mode = mode  # 'triangle' or 'cosine'
    
    def get_beta(self, step: int, epoch: int) -> float:
        cycle_progress = (epoch % self.cycle_length) / self.cycle_length
        
        if self.mode == 'triangle':
            if cycle_progress < 0.5:
                # Ascending
                beta = self.min_beta + (self.max_beta - self.min_beta) * (2 * cycle_progress)
            else:
                # Descending
                beta = self.max_beta - (self.max_beta - self.min_beta) * (2 * (cycle_progress - 0.5))
        else:  # cosine
            beta = self.min_beta + (self.max_beta - self.min_beta) * \
                   (1 + math.cos(math.pi * (1 + cycle_progress))) / 2
        
        return beta


def create_beta_scheduler(config: Dict[str, Any]) -> BetaScheduler:
    """Create beta scheduler from config."""
    scheduler_type = config.get('type', 'constant')
    params = config.get('params', {})
    
    if scheduler_type == 'constant':
        return ConstantBeta(**params)
    elif scheduler_type == 'linear_annealing':
        return LinearAnnealingBeta(**params)
    elif scheduler_type == 'exponential_annealing':
        return ExponentialAnnealingBeta(**params)
    elif scheduler_type == 'cyclical':
        return CyclicalBeta(**params)
    else:
        raise ValueError(f"Unknown beta scheduler type: {scheduler_type}")


class FreeBits:
    """Free bits constraint for KL divergence."""
    
    def __init__(self, lambda_free_bits: float = 2.0):
        """
        Args:
            lambda_free_bits: Target free bits per latent dimension (in nats)
        """
        self.lambda_free_bits = lambda_free_bits
    
    def apply(self, kl_loss: torch.Tensor, latent_dim: int) -> torch.Tensor:
        """Apply free bits constraint to KL loss."""
        # Convert total KL to per-dimension KL
        kl_per_dim = kl_loss / latent_dim
        
        # Apply free bits: max(KL, lambda)
        constrained_kl_per_dim = torch.maximum(kl_per_dim, 
                                               torch.tensor(self.lambda_free_bits, device=kl_loss.device))
        
        # Convert back to total KL
        return constrained_kl_per_dim * latent_dim


class BaseLoss(ABC):
    """Base class for all loss functions."""
    
    def __init__(self):
        self.current_step = 0
        self.current_epoch = 0
    
    def update_step(self, step: int, epoch: int):
        """Update current training step/epoch for schedulers."""
        self.current_step = step
        self.current_epoch = epoch
    
    @abstractmethod
    def __call__(self, outputs: Dict[str, torch.Tensor], 
                 targets: Dict[str, torch.Tensor], 
                 mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute loss.
        
        Returns:
            Dict with 'total' key and component losses
        """
        pass


class SimpleVAELoss(BaseLoss):
    """Simple VAE loss with beta scheduling and free bits support."""
    
    def __init__(self, beta: Union[float, Dict[str, Any]] = 1.0, 
                 free_bits: Optional[Dict[str, Any]] = None):
        super().__init__()
        
        # Setup beta scheduler
        if isinstance(beta, (int, float)):
            self.beta_scheduler = ConstantBeta(float(beta))
        else:
            self.beta_scheduler = create_beta_scheduler(beta)
        
        # Setup free bits
        self.free_bits = None
        if free_bits is not None and free_bits.get('enabled', False):
            self.free_bits = FreeBits(free_bits.get('lambda_free_bits', 2.0))
    
    def __call__(self, outputs: Dict[str, torch.Tensor], 
                 targets: Dict[str, torch.Tensor], 
                 mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        
        recon = outputs['recon']
        x = targets['x']
        mu = outputs['mu']
        logvar = outputs['logvar']

        # Get current beta value
        beta = self.beta_scheduler.get_beta(self.current_step, self.current_epoch)
        
        # Ensure numerical stability
        logvar = torch.clamp(logvar, min=-10, max=10)
        
        # Reconstruction loss (MSE)
        mask_expanded = mask.unsqueeze(-1).expand_as(x)
        valid_positions = mask_expanded.bool()
        valid_recon = recon[valid_positions]
        valid_x = x[valid_positions] 
        recon_loss = F.mse_loss(valid_recon, valid_x)
        
        # KL divergence
        kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        
        # Apply free bits if enabled
        if self.free_bits is not None:
            latent_dim = mu.shape[1]
            kl_loss = self.free_bits.apply(kl_loss, latent_dim)
        
        # Total loss
        total = recon_loss + beta * kl_loss
        
        return {
            'total': total,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'weighted_kl_loss': beta * kl_loss,
            'beta': beta  # Include current beta value for logging
        }


class DistanceAwareVAELoss(BaseLoss):
    """VAE loss with distance preservation and beta scheduling."""
    
    def __init__(self, beta: Union[float, Dict[str, Any]] = 1.0, 
                 lambda_dist: float = 0.5,
                 free_bits: Optional[Dict[str, Any]] = None):
        super().__init__()
        
        # Setup beta scheduler
        if isinstance(beta, (int, float)):
            self.beta_scheduler = ConstantBeta(float(beta))
        else:
            self.beta_scheduler = create_beta_scheduler(beta)
        
        self.lambda_dist = lambda_dist
        
        # Setup free bits
        self.free_bits = None
        if free_bits is not None and free_bits.get('enabled', False):
            self.free_bits = FreeBits(free_bits.get('lambda_free_bits', 2.0))
    
    def __call__(self, outputs: Dict[str, torch.Tensor], 
                 targets: Dict[str, torch.Tensor], 
                 mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        
        # Get current beta
        beta = self.beta_scheduler.get_beta(self.current_step, self.current_epoch)
        
        # First compute simple VAE loss components with current beta
        simple_loss = SimpleVAELoss(beta=beta, free_bits=self.free_bits.__dict__ if self.free_bits else None)
        simple_loss.update_step(self.current_step, self.current_epoch)
        simple_loss_dict = simple_loss(outputs, targets, mask)
        
        # Add distance preservation loss
        recon = outputs['recon']
        x = targets['x']
        
        # Compute pairwise distances for sequences
        def compute_sequence_distances(seq, mask):
            coords = seq[:, :, :2]  # (batch, seq_len, 2)
            diff = coords[:, 1:] - coords[:, :-1]  # (batch, seq_len-1, 2)
            distances = torch.norm(diff, dim=-1)  # (batch, seq_len-1)
            pair_mask = mask[:, 1:] * mask[:, :-1]
            masked_distances = distances * pair_mask
            return masked_distances.sum(dim=1) / (pair_mask.sum(dim=1) + 1e-8)
        
        pred_distances = compute_sequence_distances(recon, mask)
        target_distances = compute_sequence_distances(x, mask)
        distance_loss = F.mse_loss(pred_distances, target_distances)
        
        # Update total loss
        total = simple_loss_dict['total'] + self.lambda_dist * distance_loss
        
        return {
            'total': total,
            'recon_loss': simple_loss_dict['recon_loss'],
            'kl_loss': simple_loss_dict['kl_loss'],
            'weighted_kl_loss': simple_loss_dict['weighted_kl_loss'],
            'distance_loss': distance_loss,
            'weighted_distance_loss': self.lambda_dist * distance_loss,
            'beta': beta
        }


class SpeedAwareVAELoss(BaseLoss):
    """VAE loss with speed consistency and beta scheduling."""
    
    def __init__(self, beta: Union[float, Dict[str, Any]] = 1.0, 
                 lambda_speed: float = 0.3,
                 free_bits: Optional[Dict[str, Any]] = None):
        super().__init__()
        
        # Setup beta scheduler
        if isinstance(beta, (int, float)):
            self.beta_scheduler = ConstantBeta(float(beta))
        else:
            self.beta_scheduler = create_beta_scheduler(beta)
        
        self.lambda_speed = lambda_speed
        
        # Setup free bits
        self.free_bits = None
        if free_bits is not None and free_bits.get('enabled', False):
            self.free_bits = FreeBits(free_bits.get('lambda_free_bits', 2.0))
    
    def __call__(self, outputs: Dict[str, torch.Tensor], 
                 targets: Dict[str, torch.Tensor], 
                 mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        
        # Get current beta
        beta = self.beta_scheduler.get_beta(self.current_step, self.current_epoch)
        
        # First compute simple VAE loss components
        simple_loss = SimpleVAELoss(beta=beta, free_bits=self.free_bits.__dict__ if self.free_bits else None)
        simple_loss.update_step(self.current_step, self.current_epoch)
        simple_loss_dict = simple_loss(outputs, targets, mask)
        
        # Add speed consistency loss
        recon = outputs['recon']
        x = targets['x']
        
        pred_speeds = recon[:, :, 2]
        target_speeds = x[:, :, 2]
        
        speed_diff = (pred_speeds - target_speeds) ** 2
        masked_speed_diff = speed_diff * mask
        speed_loss = masked_speed_diff.sum() / (mask.sum() + 1e-8)
        
        def compute_speed_smoothness(speeds, mask):
            speed_diff = torch.abs(speeds[:, 1:] - speeds[:, :-1])
            pair_mask = mask[:, 1:] * mask[:, :-1]
            return (speed_diff * pair_mask).sum() / (pair_mask.sum() + 1e-8)
        
        pred_smoothness = compute_speed_smoothness(pred_speeds, mask)
        target_smoothness = compute_speed_smoothness(target_speeds, mask)
        smoothness_loss = torch.abs(pred_smoothness - target_smoothness)
        
        speed_aware_loss = speed_loss + 0.1 * smoothness_loss
        total = simple_loss_dict['total'] + self.lambda_speed * speed_aware_loss
        
        return {
            'total': total,
            'recon_loss': simple_loss_dict['recon_loss'],
            'kl_loss': simple_loss_dict['kl_loss'],
            'weighted_kl_loss': simple_loss_dict['weighted_kl_loss'],
            'speed_loss': speed_loss,
            'smoothness_loss': smoothness_loss,
            'weighted_speed_loss': self.lambda_speed * speed_aware_loss,
            'beta': beta
        }



# Loss factory
LOSS_REGISTRY = {
    'simple_vae': SimpleVAELoss,
    'distance_aware_vae': DistanceAwareVAELoss,
    'speed_aware_vae': SpeedAwareVAELoss,
}


def create_loss(config: Dict[str, Any]) -> BaseLoss:
    """Create loss function from config."""
    loss_type = config.get('type', 'simple_vae')
    loss_params = config.get('params', {})
    
    if loss_type not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss type: {loss_type}. Available: {list(LOSS_REGISTRY.keys())}")
    
    return LOSS_REGISTRY[loss_type](**loss_params)