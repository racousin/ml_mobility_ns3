import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict

from .base import BaseTrajectoryModel
from .vae_cnn import ConvBlock, ConvTransposeBlock


class VectorQuantizer(nn.Module):
    """
    Vector Quantizer module for VQ-VAE.
    Discrete bottleneck that maps continuous embeddings to nearest discrete codes.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # Codebook
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        # Initialize codebook uniformly
        self.embedding.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)
        
    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # inputs shape: (batch_size, seq_len, embedding_dim)
        
        # Flatten inputs for distance computation
        flat_inputs = inputs.reshape(-1, self.embedding_dim)
        
        # Calculate distances to embeddings
        # (x - y)^2 = x^2 + y^2 - 2xy
        distances = (torch.sum(flat_inputs**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_inputs, self.embedding.weight.t()))
        
        # Find nearest embeddings
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        
        # Create one-hot encodings
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Get quantized continuous vectors
        quantized = torch.matmul(encodings, self.embedding.weight).reshape(inputs.shape)
        
        # Compute losses
        # Loss 1: Pull codebook towards encoder outputs (embedding loss)
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        # Loss 2: Pull encoder outputs towards codebook (commitment loss)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Straight-through estimator for gradients
        # Use quantized during forward, but gradients flow through inputs
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return quantized, loss, perplexity


class ConditionalTrajectoryVQVAE(BaseTrajectoryModel):
    """
    VQ-VAE implementation for trajectory data.
    Uses CNN blocks for encoding/decoding and VectorQuantizer in the bottleneck.
    """
    def __init__(self, input_dim=3, base_channels=64, num_embeddings=512, 
                 embedding_dim=64, condition_dim=32, commitment_cost=0.25, 
                 dropout=0.1, num_transport_modes=5, sequence_length=2000, 
                 use_batch_norm=True, **kwargs):
        config = locals()
        config.pop('self')
        config.pop('kwargs')
        super().__init__(config)
        
        self.input_dim = input_dim
        self.base_channels = base_channels
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.condition_dim = condition_dim
        self.sequence_length = sequence_length
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        
        # Conditions embedding
        self.transport_mode_embedding = nn.Embedding(num_transport_modes, condition_dim)
        self.length_projection = nn.Linear(1, condition_dim)
        
        # Calculate architectural dimensions
        self.downsample_factor = 8  # Three stride-2 convolutions
        self.encoded_length = sequence_length // self.downsample_factor
        
        encoder_channels = [base_channels, base_channels * 2, base_channels * 4]
        
        self.encoder = nn.ModuleList([
            # Input block
            ConvBlock(input_dim, base_channels, kernel_size=7, stride=1, 
                     padding=3, use_bn=use_batch_norm),
            
            # Downsampling
            ConvBlock(base_channels, encoder_channels[0], kernel_size=4, 
                     stride=2, padding=1, use_bn=use_batch_norm),
            ConvBlock(encoder_channels[0], encoder_channels[1], kernel_size=4, 
                     stride=2, padding=1, use_bn=use_batch_norm),
            ConvBlock(encoder_channels[1], encoder_channels[2], kernel_size=4, 
                     stride=2, padding=1, use_bn=use_batch_norm),
            
            # Processing with residuals
            ConvBlock(encoder_channels[2], encoder_channels[2], kernel_size=3, 
                     stride=1, padding=1, use_bn=use_batch_norm, use_residual=True),
            ConvBlock(encoder_channels[2], encoder_channels[2], kernel_size=3, 
                     stride=1, padding=1, use_bn=use_batch_norm, use_residual=True),
        ])
        
        total_condition_dim = condition_dim * 2
        encoder_output_dim = encoder_channels[2]
        
        # Project encoder output and conditions to bottleneck embedding dimension
        self.pre_quant_conv = nn.Conv1d(encoder_output_dim + total_condition_dim, 
                                       embedding_dim, kernel_size=1)
        
        # Vector Quantizer
        self.quantizer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        
        # Upsampling for decoder
        self.post_quant_conv = nn.Conv1d(embedding_dim + total_condition_dim, 
                                        encoder_channels[2], kernel_size=1)
        
        # Decoder architecture
        self.decoder = nn.ModuleList([
            # Initial process
            ConvBlock(encoder_channels[2], encoder_channels[2], kernel_size=3, 
                     stride=1, padding=1, use_bn=use_batch_norm, use_residual=True),
            ConvBlock(encoder_channels[2], encoder_channels[2], kernel_size=3, 
                     stride=1, padding=1, use_bn=use_batch_norm, use_residual=True),
            
            # Upsampling
            ConvTransposeBlock(encoder_channels[2], encoder_channels[1], 
                              kernel_size=4, stride=2, padding=1, use_bn=use_batch_norm),
            ConvTransposeBlock(encoder_channels[1], encoder_channels[0], 
                              kernel_size=4, stride=2, padding=1, use_bn=use_batch_norm),
            ConvTransposeBlock(encoder_channels[0], base_channels, 
                              kernel_size=4, stride=2, padding=1, use_bn=use_batch_norm),
            
            # Final processing
            ConvBlock(base_channels, base_channels, kernel_size=3, 
                     stride=1, padding=1, use_bn=use_batch_norm),
        ])
        
        self.conv_out = nn.Conv1d(base_channels, input_dim, kernel_size=1)
        self.dropout_layer = nn.Dropout(dropout)
        
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
                    
    def get_conditions(self, transport_mode: torch.Tensor, length: torch.Tensor, seq_len: int) -> torch.Tensor:
        # Generate conditioned vector
        mode_embed = self.transport_mode_embedding(transport_mode)
        length_normalized = length.unsqueeze(-1).float() / self.sequence_length
        length_embed = self.length_projection(length_normalized)
        
        # Cat and expand to specified sequence length
        cond = torch.cat([mode_embed, length_embed], dim=-1)  # (batch, cond_dim * 2)
        cond = cond.unsqueeze(2).expand(-1, -1, seq_len)  # (batch, cond_dim*2, seq_len)
        return cond
        
    def encode(self, x: torch.Tensor, conditions_spatial: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        x = x.transpose(1, 2)  # (batch, input_dim, seq_len)
        
        h = x
        for layer in self.encoder:
            h = layer(h)
            h = self.dropout_layer(h)
            
        # h is now (batch, channels, encoded_seq_len)
        # Concat conditions (need to match downsampled seq length)
        cond = F.interpolate(conditions_spatial, size=h.size(2), mode='nearest')
        h_cond = torch.cat([h, cond], dim=1)
        
        # Project to embedding dim
        z_e = self.pre_quant_conv(h_cond)
        return z_e
        
    def decode(self, z_q: torch.Tensor, conditions_spatial: torch.Tensor, target_length: int) -> torch.Tensor:
        # z_q: (batch, embedding_dim, encoded_seq_len)
        
        # Concat conditions to quantized code
        cond = F.interpolate(conditions_spatial, size=z_q.size(2), mode='nearest')
        z_q_cond = torch.cat([z_q, cond], dim=1)
        
        # Project up
        h = self.post_quant_conv(z_q_cond)
        
        # Decoder passes
        for layer in self.decoder:
            h = layer(h)
            h = self.dropout_layer(h)
            
        out = self.conv_out(h)
        out = out.transpose(1, 2)  # (batch, target_len, input_dim)
        
        if out.size(1) != target_length:
            out = out.transpose(1, 2)
            out = F.interpolate(out, size=target_length, mode='linear', align_corners=False)
            out = out.transpose(1, 2)
            
        return out

    def forward(self, x: torch.Tensor, transport_mode: torch.Tensor, 
                length: torch.Tensor, mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        
        # 1. Conditions
        # Note: conditions are generated for full spatial sequence here
        cond_spatial = self.get_conditions(transport_mode, length, x.size(1))
        
        # 2. Encode
        z_e = self.encode(x, cond_spatial)  # (batch, embedding_dim, encoded_len)
        
        # 3. Quantize
        # VectorQuantizer expects (batch, seq_len, dim), so transpose
        z_e_transposed = z_e.transpose(1, 2)
        z_q_transposed, vq_loss, perplexity = self.quantizer(z_e_transposed)
        z_q = z_q_transposed.transpose(1, 2)  # Back to (batch, dim, len)
        
        # 4. Decode
        recon = self.decode(z_q, cond_spatial, x.size(1))
        
        return {
            'recon': recon,
            'vq_loss': vq_loss,
            'perplexity': perplexity,
            # Returning mu and logvar as dummies to not break simple_vae loss if used incorrectly
            # A specialized VQVAELoss should be written
            'mu': torch.zeros_like(z_q_transposed),
            'logvar': torch.zeros_like(z_q_transposed),
        }
        
    def generate(self, conditions: Dict[str, torch.Tensor], n_samples: int, 
                 target_length: int = None) -> torch.Tensor:
        transport_mode = conditions['transport_mode']
        length = conditions['length']
        device = transport_mode.device
        target_len = target_length or self.sequence_length
        encoded_len = target_len // self.downsample_factor
        
        # Generate conditions
        cond_spatial = self.get_conditions(transport_mode, length, target_len)
        
        # Sample random discrete latent codes
        indices = torch.randint(0, self.num_embeddings, (n_samples, encoded_len), device=device)
        
        # Retrieve continuous embeddings
        # self.quantizer.embedding(indices) -> (batch, encoded_len, embedding_dim)
        z_q = self.quantizer.embedding(indices)
        z_q = z_q.transpose(1, 2)  # (batch, embedding_dim, encoded_len)
        
        with torch.no_grad():
            trajectories = self.decode(z_q, cond_spatial, target_len)
            
        return trajectories