import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Dict
from .base import BaseTrajectoryModel


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for trajectory sequences."""
    
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, 
                                   mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_heads, seq_len, d_k = Q.size()
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Apply mask if provided
        if mask is not None:
            # Expand mask to match attention shape [batch_size, num_heads, seq_len, seq_len]
            if mask.dim() == 2:  # [batch_size, seq_len]
                mask = mask.unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1, seq_len]
                mask = mask.expand(-1, num_heads, seq_len, -1)
            scores.masked_fill_(mask == 0, -1e9)
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len, d_model = x.size()
        
        # Linear projections and reshape for multi-head attention
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        attn_output, _ = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        # Output projection
        output = self.w_o(attn_output)
        
        return output


class TransformerBlock(nn.Module):
    """Transformer block with self-attention and feed-forward network."""
    
    def __init__(self, d_model: int, num_heads: int = 8, d_ff: int = None, dropout: float = 0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Self-attention with residual connection
        attn_output = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class PositionalEncoding(nn.Module):
    """Positional encoding for sequence positions."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :].transpose(0, 1)


class ConditionalTrajectoryVAEAttention(BaseTrajectoryModel):
    """Pure attention-based VAE for trajectory generation (no LSTM)."""
    
    def __init__(self, input_dim=3, hidden_dim=256, latent_dim=32, 
                 condition_dim=32, dropout=0.1, num_transport_modes=5, 
                 sequence_length=2000, num_attention_heads=8, 
                 num_attention_layers=2, **kwargs):
        config = locals()
        config.pop('self')
        config.pop('kwargs')
        super().__init__(config)
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.sequence_length = sequence_length
        self.num_attention_heads = num_attention_heads
        self.num_attention_layers = num_attention_layers
        
        # Transport mode embedding
        self.transport_mode_embedding = nn.Embedding(num_transport_modes, condition_dim)
        self.length_projection = nn.Linear(1, condition_dim)
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim, sequence_length)
        
        # Encoder transformer blocks
        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_attention_heads, None, dropout)
            for _ in range(num_attention_layers)
        ])
        
        # Global pooling for encoding (to get fixed-size representation)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Latent projections
        self.fc_mu = nn.Linear(hidden_dim + condition_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim + condition_dim * 2, latent_dim)
        
        # Decoder
        self.latent_projection = nn.Linear(latent_dim + condition_dim * 2, hidden_dim)
        
        # Decoder transformer blocks
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_attention_heads, None, dropout)
            for _ in range(num_attention_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, input_dim)
        
        # Learnable position embeddings for decoder
        self.decoder_pos_embed = nn.Parameter(torch.randn(1, sequence_length, hidden_dim) * 0.02)
        
    def get_conditions(self, transport_mode: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
        """Get condition embeddings."""
        mode_embed = self.transport_mode_embedding(transport_mode)
        length_normalized = length.unsqueeze(-1).float() / self.sequence_length
        length_embed = self.length_projection(length_normalized)
        return torch.cat([mode_embed, length_embed], dim=-1)
    
    def encode(self, x: torch.Tensor, conditions: torch.Tensor, mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input sequence with pure attention."""
        batch_size, seq_len, _ = x.shape
        
        # Project input to hidden dimension
        x_proj = self.input_projection(x)
        
        # Add positional encoding
        x_pos = self.pos_encoding(x_proj)
        
        # Apply transformer blocks
        for block in self.encoder_blocks:
            x_pos = block(x_pos, mask)
        
        # Global pooling to get fixed-size representation
        pooled = self.global_pool(x_pos.transpose(1, 2)).squeeze(-1)  # [batch_size, hidden_dim]
        
        # Combine with conditions
        h_cond = torch.cat([pooled, conditions], dim=-1)
        
        # Latent projections
        mu = self.fc_mu(h_cond)
        logvar = self.fc_logvar(h_cond)
        
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor, conditions: torch.Tensor, 
               target_length: int = None, mask: torch.Tensor = None) -> torch.Tensor:
        """Decode latent representation with pure attention."""
        batch_size = z.size(0)
        seq_len = target_length or self.sequence_length
        
        # Combine latent with conditions and project
        z_cond = torch.cat([z, conditions], dim=-1)
        z_proj = self.latent_projection(z_cond)  # [batch_size, hidden_dim]
        
        # Expand to sequence length and add positional embeddings
        z_seq = z_proj.unsqueeze(1).expand(-1, seq_len, -1)  # [batch_size, seq_len, hidden_dim]
        decoder_input = z_seq + self.decoder_pos_embed[:, :seq_len, :]
        
        # Apply transformer blocks
        for block in self.decoder_blocks:
            decoder_input = block(decoder_input, mask)
        
        # Project to output dimension
        output = self.output_projection(decoder_input)
        
        return output
    
    def forward(self, x: torch.Tensor, transport_mode: torch.Tensor, 
                length: torch.Tensor, mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """Forward pass through the model."""
        conditions = self.get_conditions(transport_mode, length)
        mu, logvar = self.encode(x, conditions, mask)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, conditions, x.size(1), mask)
        
        return {
            'recon': recon, 
            'mu': mu, 
            'logvar': logvar, 
            'z': z,
            'conditions': conditions
        }
    
    def generate(self, conditions: Dict[str, torch.Tensor], n_samples: int, 
                 target_length: int = None) -> torch.Tensor:
        """Generate new trajectories."""
        transport_mode = conditions['transport_mode']
        length = conditions['length']
        
        device = transport_mode.device
        z = torch.randn(n_samples, self.latent_dim, device=device)
        conditions_embed = self.get_conditions(transport_mode, length)
        
        with torch.no_grad():
            trajectories = self.decode(z, conditions_embed, target_length)
        
        return trajectories