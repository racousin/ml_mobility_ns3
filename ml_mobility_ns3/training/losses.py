import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, List, Optional, Union
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

class BaseLoss(nn.Module, ABC):
    """Base class for all loss functions."""
    def __init__(self):
        super().__init__()
        self.current_step = 0
        self.current_epoch = 0
    
    def update_step(self, step: int, epoch: int):
        self.current_step = step
        self.current_epoch = epoch

    @abstractmethod
    def __call__(self, outputs: Dict[str, torch.Tensor], 
                 targets: Dict[str, torch.Tensor], 
                 mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        pass

# --- Beta Schedulers ---

class BaseBetaScheduler(ABC):
    @abstractmethod
    def get_beta(self, step: int, epoch: int) -> float:
        pass

class ConstantBetaScheduler(BaseBetaScheduler):
    def __init__(self, value: float = 1.0):
        self.value = value
    def get_beta(self, step: int, epoch: int) -> float:
        return self.value

class CyclicalBetaScheduler(BaseBetaScheduler):
    def __init__(self, start: float = 0.0, stop: float = 1.0, 
                 n_steps: int = 10000, n_cycles: int = 4, ratio: float = 0.5):
        self.start = start
        self.stop = stop
        self.period = n_steps // n_cycles
        self.warmup = int(self.period * ratio)

    def get_beta(self, step: int, epoch: int) -> float:
        step_in_cycle = step % self.period
        if step_in_cycle < self.warmup:
            return self.start + (self.stop - self.start) * step_in_cycle / self.warmup
        return self.stop

class AdaptiveSlowAnnealingBeta(BaseBetaScheduler):
    """Slowly increases beta only when loss stabilizes."""
    def __init__(self, start_beta=0.001, max_beta=0.5, 
                 inc_factor=1.5, patience=10, threshold=0.01):
        self.beta = start_beta
        self.max_beta = max_beta
        self.inc_factor = inc_factor
        self.patience = patience
        self.threshold = threshold
        self.best_loss = float('inf')
        self.epochs_without_improvement = 0
        self.converged = False

    def update_epoch_loss(self, epoch, loss):
        if self.converged: return
        if loss < self.best_loss * (1 - self.threshold):
            self.best_loss = loss
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
        
        if self.epochs_without_improvement >= self.patience:
            old_beta = self.beta
            self.beta = min(self.beta * self.inc_factor, self.max_beta)
            self.epochs_without_improvement = 0
            logger.info(f"Beta annealing: increased beta from {old_beta:.4f} to {self.beta:.4f}")
            if self.beta >= self.max_beta:
                self.converged = True
                logger.info("Beta annealing: reached max_beta, annealing finished.")

    def get_beta(self, step: int, epoch: int) -> float:
        return self.beta

    def get_status(self):
        return {'beta': self.beta, 'epochs_without_improvement': self.epochs_without_improvement, 'converged': self.converged}

def create_beta_scheduler(config: Union[float, Dict[str, Any]]) -> BaseBetaScheduler:
    if isinstance(config, (int, float)):
        return ConstantBetaScheduler(float(config))
    
    stype = config.get('type', 'constant')
    params = config.get('params', {})
    
    if stype == 'constant':
        return ConstantBetaScheduler(config.get('value', 1.0))
    elif stype == 'cyclical':
        return CyclicalBetaScheduler(**params)
    elif stype == 'adaptive':
        return AdaptiveSlowAnnealingBeta(**params)
    return ConstantBetaScheduler(1.0)

# --- KL Components ---

class FreeBits(nn.Module):
    def __init__(self, lambda_free_bits: float = 2.0):
        super().__init__()
        self.lambda_free_bits = lambda_free_bits

    def apply(self, kl_loss: torch.Tensor, latent_dim: int) -> torch.Tensor:
        return torch.max(kl_loss, torch.tensor(self.lambda_free_bits, device=kl_loss.device))

# --- Loss Implementations ---

class SimpleVAELoss(BaseLoss):
    def __init__(self, beta: Any = 1.0, free_bits: Optional[Dict] = None, **kwargs):
        super().__init__()
        self.beta_scheduler = create_beta_scheduler(beta)
        self.free_bits = None
        if free_bits and free_bits.get('enabled', False):
            self.free_bits = FreeBits(free_bits.get('lambda_free_bits', 2.0))

    def __call__(self, outputs: Dict[str, torch.Tensor], 
                 targets: Dict[str, torch.Tensor], 
                 mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        recon, mu, logvar = outputs['recon'], outputs['mu'], outputs['logvar']
        
        recon_loss = F.mse_loss(recon, targets['x'], reduction='none')
        recon_loss = (recon_loss * mask.unsqueeze(-1)).sum() / mask.sum()
        
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        if self.free_bits:
            kl_loss = self.free_bits.apply(kl_loss, mu.shape[-1])
            
        beta = self.beta_scheduler.get_beta(self.current_step, self.current_epoch)
        total_loss = recon_loss + beta * kl_loss
        
        return {'total': total_loss, 'recon_loss': recon_loss, 'kl_loss': kl_loss, 'beta': torch.tensor(beta)}

class DiffusionLoss(BaseLoss):
    def __init__(self, **kwargs):
        super().__init__()
    def __call__(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        noise_loss = F.mse_loss(outputs['recon'], outputs['target_noise'])
        return {'total': noise_loss, 'noise_loss': noise_loss}

class DiffusionGANLoss(BaseLoss):
    def __init__(self, **kwargs):
        super().__init__()
    def __call__(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], mask: torch.Tensor, mode: str = 'generator') -> Dict[str, torch.Tensor]:
        if mode == 'generator':
            noise_loss = F.mse_loss(outputs['recon'], outputs['target_noise'])
            return {'total': noise_loss, 'noise_loss': noise_loss}
        return {'total': torch.tensor(0.0)}

# --- Factory ---

def create_loss(config: Dict[str, Any]) -> BaseLoss:
    from ml_mobility_ns3.training.distance_aware_loss import DistanceVAELoss
    
    LOSS_REGISTRY = {
        'simple_vae': SimpleVAELoss,
        'distance_vae': DistanceVAELoss,
        'diffusion': DiffusionLoss,
        'diffusion_gan': DiffusionGANLoss,
    }
    
    loss_type = config.get('type', 'simple_vae')
    loss_params = config.get('params', {})
    if loss_type not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss type: {loss_type}")
    return LOSS_REGISTRY[loss_type](**loss_params)