# ml_mobility_ns3/models/vae.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import math


class MultiHeadAttention(nn.Module):
    """Multi-head attention module with optional masking."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        
        # Linear transformations and reshape
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided
        if mask is not None:
            # Expand mask for heads: (batch, 1, seq_len, seq_len)
            mask = mask.unsqueeze(1).unsqueeze(3)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)
        
        # Reshape and final linear transformation
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(context)
        
        return output


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with attention and feed-forward."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual connection
        attn_out = self.attention(x, mask)
        x = self.norm1(x + attn_out)
        
        # Feed-forward with residual connection
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        
        return x


class ConditionalTrajectoryVAE(nn.Module):
    """Conditional VAE for trajectory generation with transport mode and length conditioning."""
    
    def __init__(
        self,
        input_dim: int = 3,  # lat, lon, speed
        sequence_length: int = 2000,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        num_layers: int = 2,
        num_transport_modes: int = 10,  # number of transport modes + multimodal
        condition_dim: int = 32,  # dimension for condition embeddings
        architecture: str = 'lstm',  # 'lstm' or 'attention'
        attention_params: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.num_transport_modes = num_transport_modes
        self.num_layers = num_layers
        self.architecture = architecture
        
        # Default attention parameters
        if attention_params is None:
            attention_params = {}
        self.n_heads = attention_params.get('n_heads', 8)
        self.d_ff = attention_params.get('d_ff', hidden_dim * 4)
        self.dropout = attention_params.get('dropout', 0.1)
        self.use_causal_mask = attention_params.get('use_causal_mask', False)
        self.pooling = attention_params.get('pooling', 'mean')  # 'mean', 'max', or 'cls'
        
        # Condition embeddings
        self.transport_mode_embedding = nn.Embedding(num_transport_modes, condition_dim)
        self.length_projection = nn.Linear(1, condition_dim)  # project normalized length
        
        # Total condition dimension (transport mode + length)
        total_condition_dim = condition_dim * 2
        
        # Input projection for attention architecture
        if self.architecture == 'attention':
            self.input_projection = nn.Linear(input_dim, hidden_dim)
            self.positional_encoding = self._create_positional_encoding()
            
            # CLS token for 'cls' pooling
            if self.pooling == 'cls':
                self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # Encoder
        if self.architecture == 'lstm':
            self.encoder_lstm = nn.LSTM(
                input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True
            )
            encoder_output_dim = hidden_dim * 2
        elif self.architecture == 'attention':
            self.encoder_layers = nn.ModuleList([
                TransformerEncoderLayer(hidden_dim, self.n_heads, self.d_ff, self.dropout)
                for _ in range(num_layers)
            ])
            encoder_output_dim = hidden_dim
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        # Include conditions in latent space projection
        self.fc_mu = nn.Linear(encoder_output_dim + total_condition_dim, latent_dim)
        self.fc_logvar = nn.Linear(encoder_output_dim + total_condition_dim, latent_dim)
        
        # Decoder
        self.fc_latent = nn.Linear(latent_dim + total_condition_dim, hidden_dim)
        
        if self.architecture == 'lstm':
            self.decoder_lstm = nn.LSTM(
                hidden_dim, hidden_dim, num_layers, batch_first=True
            )
            self.fc_out = nn.Linear(hidden_dim, input_dim)
        elif self.architecture == 'attention':
            self.decoder_layers = nn.ModuleList([
                TransformerEncoderLayer(hidden_dim, self.n_heads, self.d_ff, self.dropout)
                for _ in range(num_layers)
            ])
            self.fc_out = nn.Linear(hidden_dim, input_dim)
    
    def _create_positional_encoding(self) -> torch.Tensor:
        """Create sinusoidal positional encoding."""
        pe = torch.zeros(self.sequence_length, self.hidden_dim)
        position = torch.arange(0, self.sequence_length).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, self.hidden_dim, 2).float() * 
                            -(math.log(10000.0) / self.hidden_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)
    
    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal mask for attention."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask == 0
        
    def get_conditions(self, transport_mode: torch.Tensor, trip_length: torch.Tensor) -> torch.Tensor:
        """Create condition embeddings from transport mode and trip length."""
        # Transport mode embedding
        mode_embed = self.transport_mode_embedding(transport_mode)  # (batch, condition_dim)
        
        # Length embedding (normalize length and project)
        length_normalized = trip_length.unsqueeze(-1).float() / self.sequence_length  # normalize to [0,1]
        length_embed = self.length_projection(length_normalized)  # (batch, condition_dim)
        
        # Concatenate conditions
        conditions = torch.cat([mode_embed, length_embed], dim=-1)  # (batch, condition_dim * 2)
        return conditions
    
    def encode_lstm(self, x: torch.Tensor, conditions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode using LSTM architecture."""
        _, (h, _) = self.encoder_lstm(x)
        # Concatenate forward and backward hidden states
        h = torch.cat([h[-2], h[-1]], dim=1)  # (batch, hidden_dim * 2)
        
        # Concatenate with conditions
        h_conditioned = torch.cat([h, conditions], dim=-1)
        
        mu = self.fc_mu(h_conditioned)
        logvar = self.fc_logvar(h_conditioned)
        return mu, logvar
    
    def encode_attention(self, x: torch.Tensor, conditions: torch.Tensor, 
                        mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode using attention architecture."""
        batch_size = x.size(0)
        
        # Project input to hidden dimension
        x = self.input_projection(x)  # (batch, seq_len, hidden_dim)
        
        # Add positional encoding
        x = x + self.positional_encoding[:, :x.size(1), :]
        
        # Add CLS token if using CLS pooling
        if self.pooling == 'cls':
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
            if mask is not None:
                # Add mask for CLS token
                cls_mask = torch.ones(batch_size, 1, device=mask.device, dtype=mask.dtype)
                mask = torch.cat([cls_mask, mask], dim=1)
        
        # Create attention mask
        attn_mask = mask
        if self.use_causal_mask:
            causal_mask = self._create_causal_mask(x.size(1), x.device)
            if attn_mask is not None:
                # Combine padding mask with causal mask
                attn_mask = attn_mask.unsqueeze(1) & causal_mask.unsqueeze(0)
            else:
                attn_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Pass through transformer layers
        for layer in self.encoder_layers:
            x = layer(x, attn_mask)
        
        # Pool to get sequence representation
        if self.pooling == 'cls':
            h = x[:, 0, :]  # Use CLS token
        elif self.pooling == 'mean':
            if mask is not None:
                # Masked mean pooling
                if self.pooling == 'cls':
                    # Remove CLS token from mask for mean calculation
                    mask_for_mean = mask[:, 1:]
                    x_for_mean = x[:, 1:, :]
                else:
                    mask_for_mean = mask
                    x_for_mean = x
                mask_expanded = mask_for_mean.unsqueeze(-1).expand_as(x_for_mean)
                h = (x_for_mean * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
            else:
                h = x.mean(dim=1)
        elif self.pooling == 'max':
            h, _ = x.max(dim=1)
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")
        
        # Concatenate with conditions
        h_conditioned = torch.cat([h, conditions], dim=-1)
        
        mu = self.fc_mu(h_conditioned)
        logvar = self.fc_logvar(h_conditioned)
        return mu, logvar
        
    def encode(self, x: torch.Tensor, conditions: torch.Tensor, 
               mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode trajectory to latent distribution parameters."""
        if self.architecture == 'lstm':
            return self.encode_lstm(x, conditions)
        elif self.architecture == 'attention':
            return self.encode_attention(x, conditions, mask)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode_lstm(self, z: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        """Decode using LSTM architecture."""
        batch_size = z.size(0)
        
        # Concatenate latent with conditions
        z_conditioned = torch.cat([z, conditions], dim=-1)
        
        # Project to hidden
        h = self.fc_latent(z_conditioned)
        h = h.unsqueeze(1).repeat(1, self.sequence_length, 1)
        
        # Decode sequence
        out, _ = self.decoder_lstm(h)
        out = self.fc_out(out)
        
        return out
    
    def decode_attention(self, z: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        """Decode using attention architecture."""
        batch_size = z.size(0)
        
        # Concatenate latent with conditions
        z_conditioned = torch.cat([z, conditions], dim=-1)
        
        # Project to hidden and create sequence
        h = self.fc_latent(z_conditioned)
        h = h.unsqueeze(1).repeat(1, self.sequence_length, 1)
        
        # Add positional encoding
        h = h + self.positional_encoding
        
        # Create causal mask for autoregressive decoding if specified
        attn_mask = None
        if self.use_causal_mask:
            attn_mask = self._create_causal_mask(self.sequence_length, h.device)
            attn_mask = attn_mask.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Pass through transformer layers
        for layer in self.decoder_layers:
            h = layer(h, attn_mask)
        
        # Project to output dimension
        out = self.fc_out(h)
        
        return out
    
    def decode(self, z: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        """Decode from latent space to trajectory."""
        if self.architecture == 'lstm':
            return self.decode_lstm(z, conditions)
        elif self.architecture == 'attention':
            return self.decode_attention(z, conditions)
    
    def forward(
        self, 
        x: torch.Tensor, 
        transport_mode: torch.Tensor, 
        trip_length: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass."""
        conditions = self.get_conditions(transport_mode, trip_length)
        mu, logvar = self.encode(x, conditions, mask)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, conditions)
        return recon, mu, logvar
    
    def generate(
        self, 
        transport_mode: torch.Tensor, 
        trip_length: torch.Tensor, 
        n_samples: Optional[int] = None,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """Generate new trajectories given conditions."""
        if n_samples is None:
            n_samples = transport_mode.size(0)
            
        conditions = self.get_conditions(transport_mode, trip_length)
        z = torch.randn(n_samples, self.latent_dim).to(device)
        
        with torch.no_grad():
            trajectories = self.decode(z, conditions)
        return trajectories
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        config = {
            'input_dim': self.input_dim,
            'sequence_length': self.sequence_length,
            'hidden_dim': self.hidden_dim,
            'latent_dim': self.latent_dim,
            'num_layers': self.num_layers,
            'num_transport_modes': self.num_transport_modes,
            'condition_dim': self.condition_dim,
            'architecture': self.architecture,
        }
        
        if self.architecture == 'attention':
            config['attention_params'] = {
                'n_heads': self.n_heads,
                'd_ff': self.d_ff,
                'dropout': self.dropout,
                'use_causal_mask': self.use_causal_mask,
                'pooling': self.pooling,
            }
        
        return config


def masked_vae_loss(
    recon: torch.Tensor, 
    x: torch.Tensor, 
    mu: torch.Tensor, 
    logvar: torch.Tensor, 
    mask: torch.Tensor,
    beta: float = 1.0
) -> Tuple[torch.Tensor, dict]:
    """
    VAE loss function with masking for variable-length sequences.
    
    Args:
        recon: Reconstructed trajectories (batch, seq_len, input_dim)
        x: Original trajectories (batch, seq_len, input_dim)
        mu: Latent mean (batch, latent_dim)
        logvar: Latent log variance (batch, latent_dim)
        mask: Binary mask indicating valid positions (batch, seq_len)
        beta: Weight for KL loss
    """
    # Reconstruction loss with masking
    # Only compute loss on valid (non-padded) positions
    diff = (recon - x) ** 2  # (batch, seq_len, input_dim)
    
    # Expand mask to match input dimensions
    mask_expanded = mask.unsqueeze(-1).expand_as(diff)  # (batch, seq_len, input_dim)
    
    # Apply mask and compute mean over valid positions
    masked_diff = diff * mask_expanded
    num_valid = mask_expanded.sum()
    recon_loss = masked_diff.sum() / (num_valid + 1e-8)  # avoid division by zero
    
    # KL divergence (not masked, applied to latent space)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    
    # Total loss
    loss = recon_loss + beta * kl_loss
    
    return loss, {
        'loss': loss.item(),
        'recon_loss': recon_loss.item(),
        'kl_loss': kl_loss.item(),
        'num_valid_points': num_valid.item()
    }