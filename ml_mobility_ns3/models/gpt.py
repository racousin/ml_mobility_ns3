import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Tuple
from .base import BaseTrajectoryModel


class CausalSelfAttention(nn.Module):
    """Multi-head causal (masked) self-attention."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, max_len: int = 2048):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Causal mask
        self.register_buffer(
            'causal_mask',
            torch.tril(torch.ones(max_len, max_len)).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
        B, T, C = x.size()

        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_k).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply causal mask
        scores = scores.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))

        # Apply padding mask if provided
        if padding_mask is not None:
            # padding_mask: (B, T) with 1 for valid, 0 for padding
            pad_mask = padding_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
            scores = scores.masked_fill(pad_mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        out = (attn @ v).transpose(1, 2).contiguous().reshape(B, T, C)
        return self.resid_dropout(self.out_proj(out))


class GPTBlock(nn.Module):
    """Transformer block with causal self-attention."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1, max_len: int = 2048):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout, max_len)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), padding_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class ConditionalTrajectoryGPT(BaseTrajectoryModel):
    """GPT-like autoregressive model for trajectory generation.

    Predicts the next trajectory point given all previous points,
    conditioned on transport mode and trajectory length.
    """

    def __init__(self, input_dim=3, d_model=256, n_heads=8, n_layers=6,
                 d_ff=1024, dropout=0.1, condition_dim=32,
                 num_transport_modes=5, sequence_length=2000, **kwargs):
        config = locals()
        config.pop('self')
        config.pop('kwargs')
        super().__init__(config)

        self.input_dim = input_dim
        self.d_model = d_model
        self.sequence_length = sequence_length

        # Condition embeddings
        self.transport_mode_embedding = nn.Embedding(num_transport_modes, condition_dim)
        self.length_projection = nn.Linear(1, condition_dim)

        # Input projection: input_dim + condition_dim*2 -> d_model
        # Conditions are concatenated to each input token
        self.input_projection = nn.Linear(input_dim + condition_dim * 2, d_model)

        # Learnable positional embeddings
        self.pos_embedding = nn.Embedding(sequence_length, d_model)

        # Start token (learnable)
        self.start_token = nn.Parameter(torch.randn(1, 1, input_dim) * 0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            GPTBlock(d_model, n_heads, d_ff, dropout, sequence_length)
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, input_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights following GPT-2 conventions."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def get_conditions(self, transport_mode: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
        mode_embed = self.transport_mode_embedding(transport_mode)
        length_normalized = length.unsqueeze(-1).float() / self.sequence_length
        length_embed = self.length_projection(length_normalized)
        return torch.cat([mode_embed, length_embed], dim=-1)  # (B, condition_dim*2)

    def _forward_transformer(self, x_input: torch.Tensor, conditions: torch.Tensor,
                              padding_mask: torch.Tensor = None) -> torch.Tensor:
        """Run the transformer on input tokens with conditions.

        Args:
            x_input: (B, T, input_dim) - input trajectory points
            conditions: (B, condition_dim*2) - condition embeddings
            padding_mask: (B, T) optional mask

        Returns:
            (B, T, input_dim) - predicted next points
        """
        B, T, _ = x_input.shape

        # Expand conditions to match sequence length and concatenate
        cond_expanded = conditions.unsqueeze(1).expand(-1, T, -1)  # (B, T, cond_dim*2)
        x_cond = torch.cat([x_input, cond_expanded], dim=-1)  # (B, T, input_dim + cond_dim*2)

        # Project to d_model
        h = self.input_projection(x_cond)

        # Add positional embeddings
        positions = torch.arange(T, device=x_input.device).unsqueeze(0)
        h = h + self.pos_embedding(positions)

        # Apply transformer blocks
        for block in self.blocks:
            h = block(h, padding_mask)

        h = self.ln_f(h)
        return self.output_head(h)

    def forward(self, x: torch.Tensor, transport_mode: torch.Tensor,
                length: torch.Tensor, mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """Forward pass with teacher forcing.

        Input x is shifted right: model sees x[:, :-1] and predicts x[:, 1:].
        A learned start token is prepended.
        """
        B, T, D = x.shape
        conditions = self.get_conditions(transport_mode, length)

        # Prepend start token to input, drop last position
        start = self.start_token.expand(B, -1, -1)  # (B, 1, input_dim)
        x_input = torch.cat([start, x[:, :-1, :]], dim=1)  # (B, T, input_dim)

        # Build padding mask for the shifted sequence
        padding_mask = mask  # same shape (B, T)

        # Forward through transformer
        predictions = self._forward_transformer(x_input, conditions, padding_mask)

        # Return format compatible with VAE loss pipeline
        # mu/logvar/z are zeros so KL = 0
        latent_dim = 1
        dummy = torch.zeros(B, latent_dim, device=x.device)

        return {
            'recon': predictions,
            'mu': dummy,
            'logvar': dummy,
            'z': dummy,
            'conditions': conditions,
        }

    def generate(self, conditions: Dict[str, torch.Tensor], n_samples: int,
                 target_length: int = None) -> torch.Tensor:
        """Autoregressive trajectory generation."""
        transport_mode = conditions['transport_mode']
        length = conditions['length']
        device = transport_mode.device
        seq_len = target_length or self.sequence_length

        cond_embed = self.get_conditions(transport_mode, length)

        # Start with the learned start token
        current_input = self.start_token.expand(n_samples, -1, -1).to(device)  # (n, 1, input_dim)
        trajectory = []

        with torch.no_grad():
            for t in range(seq_len):
                preds = self._forward_transformer(current_input, cond_embed)
                next_point = preds[:, -1:, :]  # (n, 1, input_dim)
                trajectory.append(next_point)
                current_input = torch.cat([current_input, next_point], dim=1)

        return torch.cat(trajectory, dim=1)  # (n, seq_len, input_dim)
