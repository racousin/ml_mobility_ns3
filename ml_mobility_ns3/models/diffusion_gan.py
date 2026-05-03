import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Tuple, List
from .base import BaseTrajectoryModel

class DiffusionGANDiscriminator(nn.Module):
    """Discriminator that looks at (xt, xt-1, t, condition)."""
    def __init__(self, input_dim: int, condition_dim: int, hidden_dim: int = 256):
        super().__init__()
        # Input: current x, previous x, time, and conditions
        self.net = nn.Sequential(
            nn.Linear(input_dim * 2 + condition_dim + 1, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x_t, x_start, t, cond):
        # Discriminator distinguishes between (x_t, real_x0) and (x_t, pred_x0)
        B, T, D = x_t.shape
        t_embed = t.view(B, 1, 1).expand(-1, T, -1)
        cond_embed = cond.unsqueeze(1).expand(-1, T, -1)
        
        combined = torch.cat([x_t, x_start, t_embed, cond_embed], dim=-1)
        logits = self.net(combined) # (B, T, 1)
        return torch.mean(logits, dim=1) # (B, 1) average over sequence

class TrajectoryDiffusionGAN(BaseTrajectoryModel):
    """
    Diffusion-GAN for trajectory generation.
    Combines the denoising power of Diffusion with the sharpness of GANs.
    """
    def __init__(self, input_dim=3, base_channels=128, condition_dim=64, 
                 timesteps=100, schedule_type='cosine', num_transport_modes=5,
                 sequence_length=2000, **kwargs):
        config = locals()
        config.pop('self')
        config.pop('kwargs')
        super().__init__(config)

        self.input_dim = input_dim
        self.timesteps = timesteps
        
        # Generator: Reuse the diffusion architecture logic (UNet-like)
        # For simplicity in this example, we use a shared denoising backbone
        from .diffusion import DenoisingUNet1D
        # DenoisingUNet1D expects condition_dim to be the size of EACH component (it multiplies by 2 internally)
        self.generator = DenoisingUNet1D(input_dim, base_channels, condition_dim=condition_dim // 2)
        
        # Discriminator
        self.discriminator = DiffusionGANDiscriminator(input_dim, condition_dim)
        
        # Diffusion parameters
        self.register_buffer('betas', self._get_beta_schedule(schedule_type, timesteps))
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        
        # Condition embeddings
        self.transport_embedding = nn.Embedding(num_transport_modes, condition_dim // 2)
        self.length_proj = nn.Linear(1, condition_dim // 2)
        
        # Register buffers for posterior sampling
        self.register_buffer('alphas_cumprod_prev', F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0))
        self.register_buffer('posterior_variance', 
                            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        
        # Coefficients for posterior mean
        self.register_buffer('posterior_mean_coef1',
                            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod))

    def _get_beta_schedule(self, schedule_type, timesteps):
        if schedule_type == 'linear':
            return torch.linspace(1e-4, 0.02, timesteps)
        elif schedule_type == 'cosine':
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps)
            alphas_cumprod = torch.cos(((x / timesteps) + 0.008) / (1 + 0.008) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clip(betas, 0.0001, 0.9999)
        return torch.linspace(1e-4, 0.02, timesteps)

    def get_conditions(self, transport_mode, length):
        mode_embed = self.transport_embedding(transport_mode)
        length_norm = length.unsqueeze(-1).float() / 2000.0
        len_embed = self.length_proj(length_norm)
        return torch.cat([mode_embed, len_embed], dim=-1)

    
    def q_posterior_sample(self, x_t, x_0, t):
        """Sample x_{t-1} ~ q(x_{t-1} | x_t, x_0)."""
        coef1 = self.posterior_mean_coef1[t].view(-1, 1, 1)
        coef2 = self.posterior_mean_coef2[t].view(-1, 1, 1)
        mu = coef1 * x_0 + coef2 * x_t
        var = self.posterior_variance[t].view(-1, 1, 1)
        noise = torch.randn_like(x_t)
        return mu + torch.sqrt(var) * noise

    def q_sample(self, x_0, t, noise=None):
        """Forward diffusion: add noise to data."""
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alphas_cumprod_t = torch.sqrt(self.alphas_cumprod[t]).view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1.0 - self.alphas_cumprod[t]).view(-1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise

    def predict_x0(self, x_t, t, noise_pred):
        """Estimate x_0 from x_t and predicted noise."""
        sqrt_inv_alphas_cumprod_t = torch.sqrt(1.0 / self.alphas_cumprod[t]).view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_inv_t = torch.sqrt(1.0 / self.alphas_cumprod[t] - 1.0).view(-1, 1, 1)
        
        return sqrt_inv_alphas_cumprod_t * x_t - sqrt_one_minus_alphas_cumprod_inv_t * noise_pred

    def forward(self, x, transport_mode, length, mask=None) -> Dict[str, torch.Tensor]:
        """
        Training forward pass. 
        In Diffusion-GAN, we need to return both the noise prediction 
        and the data needed for the adversarial loss.
        """
        B, T, D = x.shape
        device = x.device
        t = torch.randint(0, self.timesteps, (B,), device=device).long()
        cond = self.get_conditions(transport_mode, length)
        
        noise = torch.randn_like(x)
        x_t = self.q_sample(x, t, noise)
        
        # Predict noise: transpose to (B, D, T) for Conv1D backbone
        pred_noise = self.generator(x_t.transpose(1, 2), t, cond)
        # Transpose back to (B, T, D)
        pred_noise = pred_noise.transpose(1, 2)
        
        # Estimate x_0 from predicted noise
        x_0_pred = self.predict_x0(x_t, t, pred_noise)
        
        return {
            'recon': pred_noise,      # Noise prediction (for MSE loss)
            'x_0_pred': x_0_pred,     # Estimated trajectory (for metrics)
            'x_t': x_t,
            't': t,
            'cond': cond,
            'target_noise': noise,
            'x_0': x                  # Original data
        }

    @torch.no_grad()
    def generate(self, conditions, n_samples, target_length=None):
        # Standard DDPM sampling
        device = next(self.parameters()).device
        mode = conditions['transport_mode']
        length = conditions['length']
        cond = self.get_conditions(mode, length)
        
        x = torch.randn(n_samples, target_length or 2000, self.input_dim, device=device)
        
        for i in reversed(range(0, self.timesteps)):
            t = torch.full((n_samples,), i, device=device, dtype=torch.long)
            # Transpose to (B, D, T) for backbone
            pred_noise = self.generator(x.transpose(1, 2), t, cond).transpose(1, 2)
            
            alpha = self.alphas[i]
            alpha_cumprod = self.alphas_cumprod[i]
            beta = self.betas[i]
            
            if i > 0:
                noise = torch.randn_like(x)
            else:
                noise = 0
                
            x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_cumprod))) * pred_noise) + torch.sqrt(beta) * noise
            
        return x
