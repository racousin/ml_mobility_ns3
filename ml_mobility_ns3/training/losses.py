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



# Loss factory
LOSS_REGISTRY = {
    'simple_vae': SimpleVAELoss,
}


def create_loss(config: Dict[str, Any]) -> BaseLoss:
    """Create loss function from config."""
    loss_type = config.get('type', 'simple_vae')
    loss_params = config.get('params', {})
    
    if loss_type not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss type: {loss_type}. Available: {list(LOSS_REGISTRY.keys())}")
    
    return LOSS_REGISTRY[loss_type](**loss_params)