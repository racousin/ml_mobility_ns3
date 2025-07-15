import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict
from .base import BaseTrajectoryModel


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding=0, use_bn=True, use_residual=False):
        super().__init__()
        self.use_residual = use_residual and (stride == 1) and (in_channels == out_channels)
        
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                             stride=stride, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels) if use_bn else nn.Identity()
        self.activation = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        identity = x
        out = self.conv(x)
        out = self.bn(out)
        out = self.activation(out)
        
        if self.use_residual:
            out = out + identity
        return out


class ConvTransposeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding=0, output_padding=0, use_bn=True):
        super().__init__()
        
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size,
                                      stride=stride, padding=padding, 
                                      output_padding=output_padding)
        self.bn = nn.BatchNorm1d(out_channels) if use_bn else nn.Identity()
        self.activation = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.activation(out)
        return out


class ConditionalTrajectoryVAECNN(BaseTrajectoryModel):
    
    def __init__(self, input_dim=3, base_channels=64, latent_dim=32, 
                 condition_dim=32, dropout=0.1, num_transport_modes=5, 
                 sequence_length=2000, use_batch_norm=True, **kwargs):
        config = locals()
        config.pop('self')
        config.pop('kwargs')
        super().__init__(config)
        
        self.input_dim = input_dim
        self.base_channels = base_channels
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.sequence_length = sequence_length
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        
        # Transport mode and length embeddings
        self.transport_mode_embedding = nn.Embedding(num_transport_modes, condition_dim)
        self.length_projection = nn.Linear(1, condition_dim)
        
        # Calculate architecture dimensions
        # We'll downsample by factor of 8 (2^3) through 3 strided convolutions
        self.downsample_factor = 8
        self.encoded_length = sequence_length // self.downsample_factor
        
        # Encoder architecture
        encoder_channels = [base_channels, base_channels * 2, base_channels * 4]
        
        self.encoder = nn.ModuleList([
            # Initial projection
            ConvBlock(input_dim, base_channels, kernel_size=7, stride=1, 
                     padding=3, use_bn=use_batch_norm),
            
            # Downsampling blocks
            ConvBlock(base_channels, encoder_channels[0], kernel_size=4, 
                     stride=2, padding=1, use_bn=use_batch_norm),
            ConvBlock(encoder_channels[0], encoder_channels[1], kernel_size=4, 
                     stride=2, padding=1, use_bn=use_batch_norm),
            ConvBlock(encoder_channels[1], encoder_channels[2], kernel_size=4, 
                     stride=2, padding=1, use_bn=use_batch_norm),
            
            # Additional processing blocks with residual connections
            ConvBlock(encoder_channels[2], encoder_channels[2], kernel_size=3, 
                     stride=1, padding=1, use_bn=use_batch_norm, use_residual=True),
            ConvBlock(encoder_channels[2], encoder_channels[2], kernel_size=3, 
                     stride=1, padding=1, use_bn=use_batch_norm, use_residual=True),
        ])
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Latent projections (include conditions)
        total_condition_dim = condition_dim * 2
        encoder_output_dim = encoder_channels[2]
        
        self.fc_mu = nn.Linear(encoder_output_dim + total_condition_dim, latent_dim)
        self.fc_logvar = nn.Linear(encoder_output_dim + total_condition_dim, latent_dim)
        
        # Decoder initial projection
        self.fc_decode = nn.Linear(latent_dim + total_condition_dim, 
                                  encoder_channels[2] * self.encoded_length)
        
        # Decoder architecture
        self.decoder = nn.ModuleList([
            # Process at encoded resolution
            ConvBlock(encoder_channels[2], encoder_channels[2], kernel_size=3, 
                     stride=1, padding=1, use_bn=use_batch_norm, use_residual=True),
            ConvBlock(encoder_channels[2], encoder_channels[2], kernel_size=3, 
                     stride=1, padding=1, use_bn=use_batch_norm, use_residual=True),
            
            # Upsampling blocks
            ConvTransposeBlock(encoder_channels[2], encoder_channels[1], 
                              kernel_size=4, stride=2, padding=1, 
                              output_padding=0, use_bn=use_batch_norm),
            ConvTransposeBlock(encoder_channels[1], encoder_channels[0], 
                              kernel_size=4, stride=2, padding=1, 
                              output_padding=0, use_bn=use_batch_norm),
            ConvTransposeBlock(encoder_channels[0], base_channels, 
                              kernel_size=4, stride=2, padding=1, 
                              output_padding=0, use_bn=use_batch_norm),
            
            # Final projection
            ConvBlock(base_channels, base_channels, kernel_size=3, 
                     stride=1, padding=1, use_bn=use_batch_norm),
        ])
        
        # Output layer
        self.conv_out = nn.Conv1d(base_channels, input_dim, kernel_size=1)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def get_conditions(self, transport_mode: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
        mode_embed = self.transport_mode_embedding(transport_mode)
        length_normalized = length.unsqueeze(-1).float() / self.sequence_length
        length_embed = self.length_projection(length_normalized)
        return torch.cat([mode_embed, length_embed], dim=-1)
    
    def encode(self, x: torch.Tensor, conditions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x shape: (batch_size, seq_len, input_dim)
        # Transpose to (batch_size, input_dim, seq_len) for Conv1d
        x = x.transpose(1, 2)
        
        # Pass through encoder
        h = x
        for layer in self.encoder:
            h = layer(h)
            h = self.dropout_layer(h)
        
        # Global pooling
        h_pooled = self.global_pool(h).squeeze(-1)  # (batch_size, channels)
        
        # Concatenate with conditions
        h_conditioned = torch.cat([h_pooled, conditions], dim=-1)
        
        # Project to latent parameters
        mu = self.fc_mu(h_conditioned)
        logvar = self.fc_logvar(h_conditioned)
        
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor, conditions: torch.Tensor, 
               target_length: int = None) -> torch.Tensor:
        batch_size = z.size(0)
        target_len = target_length or self.sequence_length
        
        # Concatenate latent with conditions
        z_conditioned = torch.cat([z, conditions], dim=-1)
        
        # Project and reshape
        h = self.fc_decode(z_conditioned)
        h = h.view(batch_size, -1, self.encoded_length)  # (batch, channels, length)
        
        # Pass through decoder
        for layer in self.decoder:
            h = layer(h)
            h = self.dropout_layer(h)
        
        # Final output projection
        out = self.conv_out(h)  # (batch_size, input_dim, seq_len)
        
        # Transpose back to (batch_size, seq_len, input_dim)
        out = out.transpose(1, 2)
        
        # Adjust to target length if needed
        if out.size(1) != target_len:
            # Use interpolation to match target length
            out = out.transpose(1, 2)  # Back to (batch, channels, length)
            out = F.interpolate(out, size=target_len, mode='linear', align_corners=False)
            out = out.transpose(1, 2)  # Back to (batch, length, channels)
        
        return out
    
    def forward(self, x: torch.Tensor, transport_mode: torch.Tensor, 
                length: torch.Tensor, mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        conditions = self.get_conditions(transport_mode, length)
        mu, logvar = self.encode(x, conditions)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, conditions, x.size(1))
        
        return {
            'recon': recon, 
            'mu': mu, 
            'logvar': logvar, 
            'z': z,
            'conditions': conditions
        }
    
    def generate(self, conditions: Dict[str, torch.Tensor], n_samples: int, 
                 target_length: int = None) -> torch.Tensor:
        transport_mode = conditions['transport_mode']
        length = conditions['length']
        
        device = transport_mode.device
        z = torch.randn(n_samples, self.latent_dim, device=device)
        conditions_embed = self.get_conditions(transport_mode, length)
        
        with torch.no_grad():
            trajectories = self.decode(z, conditions_embed, target_length)
        
        return trajectories
    
    def get_model_size(self) -> Dict[str, int]:
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'downsample_factor': self.downsample_factor,
            'encoded_length': self.encoded_length,
            'model_type': 'cnn_vae'
        }