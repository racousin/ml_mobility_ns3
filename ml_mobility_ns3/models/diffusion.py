import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional

from .base import BaseTrajectoryModel
from .vae_cnn import ConvBlock


def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class DenoisingUNet1D(nn.Module):
    """
    1D U-Net architecture for denoising trajectories
    """
    def __init__(self, input_dim=3, base_channels=64, time_dim=256, condition_dim=32, use_bn=True):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        
        # Initial projection
        # We need to process conditionals as well, so we project them and add them
        self.init_conv = nn.Conv1d(input_dim, base_channels, kernel_size=7, padding=3)
        self.cond_proj = nn.Linear(condition_dim * 2, base_channels)
        
        # Downsampling blocks
        self.down1 = ConvBlock(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1, use_bn=use_bn)
        self.down2 = ConvBlock(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1, use_bn=use_bn)
        
        # Middle block
        self.mid1 = ConvBlock(base_channels * 4, base_channels * 4, kernel_size=3, padding=1, use_bn=use_bn, use_residual=True)
        self.mid2 = ConvBlock(base_channels * 4, base_channels * 4, kernel_size=3, padding=1, use_bn=use_bn, use_residual=True)
        
        # Upsampling blocks (using ConvTranspose1d)
        self.up1 = nn.ConvTranspose1d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.up1_conv = ConvBlock(base_channels * 4, base_channels * 2, kernel_size=3, padding=1, use_bn=use_bn)
        
        self.up2 = nn.ConvTranspose1d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1)
        self.up2_conv = ConvBlock(base_channels * 2, base_channels, kernel_size=3, padding=1, use_bn=use_bn)
        
        # Time embeddings projections
        self.time_emb_down1 = nn.Linear(time_dim, base_channels * 2)
        self.time_emb_down2 = nn.Linear(time_dim, base_channels * 4)
        self.time_emb_mid = nn.Linear(time_dim, base_channels * 4)
        self.time_emb_up1 = nn.Linear(time_dim, base_channels * 2)
        self.time_emb_up2 = nn.Linear(time_dim, base_channels)
        
        # Final block
        self.final_conv = nn.Conv1d(base_channels, input_dim, kernel_size=1)
        
    def forward(self, x, time, cond):
        # x: (batch, input_dim, seq_len)
        # cond: (batch, cond_dim*2)
        
        t = self.time_mlp(time)
        
        # Prepare condition embedding for full sequence
        c = self.cond_proj(cond) # (batch, base_channels)
        c = c.unsqueeze(-1) # (batch, base_channels, 1)
        
        # Initial projection
        x_init = self.init_conv(x) + c
        
        # Down 1
        h1 = self.down1(x_init)
        t_down1 = self.time_emb_down1(t).unsqueeze(-1)
        h1 = h1 + t_down1
        
        # Down 2
        h2 = self.down2(h1)
        t_down2 = self.time_emb_down2(t).unsqueeze(-1)
        h2 = h2 + t_down2
        
        # Middle
        h_mid = self.mid1(h2)
        t_mid = self.time_emb_mid(t).unsqueeze(-1)
        h_mid = h_mid + t_mid
        h_mid = self.mid2(h_mid)
        
        # Up 1
        h_up1 = self.up1(h_mid)
        
        # Match lengths if necessary (due to padding/stride effects for odd lengths)
        if h_up1.size(2) != h1.size(2):
            h_up1 = F.interpolate(h_up1, size=h1.size(2), mode='nearest')
            
        h_up1 = torch.cat([h_up1, h1], dim=1) # skip connection
        h_up1 = self.up1_conv(h_up1)
        t_up1 = self.time_emb_up1(t).unsqueeze(-1)
        h_up1 = h_up1 + t_up1
        
        # Up 2
        h_up2 = self.up2(h_up1)
        
        if h_up2.size(2) != x_init.size(2):
            h_up2 = F.interpolate(h_up2, size=x_init.size(2), mode='nearest')
            
        h_up2 = torch.cat([h_up2, x_init], dim=1) # skip connection
        h_up2 = self.up2_conv(h_up2)
        t_up2 = self.time_emb_up2(t).unsqueeze(-1)
        h_up2 = h_up2 + t_up2
        
        # Final projection
        return self.final_conv(h_up2)


class TrajectoryDiffusionModel(BaseTrajectoryModel):
    def __init__(self, input_dim=3, base_channels=64, condition_dim=32, 
                 timesteps=1000, schedule_type="cosine", num_transport_modes=5, 
                 sequence_length=2000, **kwargs):
        config = locals()
        config.pop('self')
        config.pop('kwargs')
        super().__init__(config)
        
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.condition_dim = condition_dim
        self.timesteps = timesteps
        self.schedule_type = schedule_type
        
        # Condition embeddings
        self.transport_mode_embedding = nn.Embedding(num_transport_modes, condition_dim)
        self.length_projection = nn.Linear(1, condition_dim)
        
        # U-Net model
        self.model = DenoisingUNet1D(
            input_dim=input_dim, 
            base_channels=base_channels, 
            condition_dim=condition_dim
        )
        
        # Setup noise schedule (we register buffers so they are moved to right device, but not trained)
        if schedule_type == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            betas = linear_beta_schedule(timesteps)
            
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        
        # For forward diffusion q(x_t | x_{t-1})
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1. - alphas_cumprod))
        
        # For backward diffusion p(x_{t-1} | x_t)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("sqrt_recip_alphas", sqrt_recip_alphas)
        self.register_buffer("betas", betas)
        
    def get_conditions(self, transport_mode: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
        mode_embed = self.transport_mode_embedding(transport_mode)
        length_normalized = length.unsqueeze(-1).float() / self.sequence_length
        length_embed = self.length_projection(length_normalized)
        return torch.cat([mode_embed, length_embed], dim=-1)
        
    def extract(self, a, t, x_shape):
        """Extract matched parameters for a given batch of timesteps."""
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
        
    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion process - adds noise to data."""
        if noise is None:
            noise = torch.randn_like(x_start)
            
        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        
    def forward(self, x: torch.Tensor, transport_mode: torch.Tensor, 
                length: torch.Tensor, mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        
        batch_size = x.shape[0]
        device = x.device
        
        # x is (batch, seq_len, input_dim), U-Net needs (batch, input_dim, seq_len)
        x = x.transpose(1, 2)
        
        # Sample random timesteps
        t = torch.randint(0, self.timesteps, (batch_size,), device=device).long()
        
        # Get conditions
        cond = self.get_conditions(transport_mode, length)
        
        # Sample noise
        noise = torch.randn_like(x)
        
        # Add noise to original data forward diffusion 
        x_noisy = self.q_sample(x_start=x, t=t, noise=noise)
        
        # Predict noise using U-Net
        predicted_noise = self.model(x_noisy, t, cond)
        
        return {
            'predicted_noise': predicted_noise,
            'target_noise': noise,
            'cond': cond,
            't': t,
            # Returning dummy recon for backward compatibility with metrics evaluation
            # although meaningful metrics should be evaluated based on generated samples
            'recon': x.transpose(1, 2), # Dummy original x 
            'mask': mask
        }
    
    @torch.no_grad()
    def p_sample(self, x, t, t_index, cond):
        """Reverse diffusion step (denoising)."""
        betas_t = self.extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = self.extract(self.sqrt_recip_alphas, t, x.shape)
        
        # Predict noise
        predicted_noise = self.model(x, t, cond)
        
        # Compute mean
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
        )
        
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self.extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
            
    @torch.no_grad()
    def generate(self, conditions: Dict[str, torch.Tensor], n_samples: int, target_length: int = None) -> torch.Tensor:
        """Generate samples using reverse diffusion."""
        transport_mode = conditions['transport_mode']
        length = conditions['length']
        device = transport_mode.device
        target_len = target_length or self.sequence_length
        
        cond = self.get_conditions(transport_mode, length)
        
        # Start from pure noise
        x = torch.randn((n_samples, self.input_dim, target_len), device=device)
        
        # Reverse process
        for i in reversed(range(0, self.timesteps)):
            t = torch.full((n_samples,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, t, i, cond)
            
        # Back to (batch, seq_len, input_dim)
        return x.transpose(1, 2)
