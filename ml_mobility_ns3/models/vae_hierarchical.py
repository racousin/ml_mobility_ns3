import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Dict, List
from .base import BaseTrajectoryModel


class SegmentEncoder(nn.Module):
    """Encodes individual trajectory segments."""
    
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, 
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # LSTM for segment encoding
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True, dropout=dropout
        )
        
        # Projections to latent space
        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch_size, segment_length, input_dim]
            mask: [batch_size, segment_length]
        Returns:
            mu, logvar: [batch_size, latent_dim]
        """
        # LSTM encoding
        output, (h_n, _) = self.lstm(x)
        
        # Use final hidden states from both directions
        # h_n: [num_layers * 2, batch_size, hidden_dim]
        forward_h = h_n[-2]  # Last layer, forward direction
        backward_h = h_n[-1]  # Last layer, backward direction
        
        # Concatenate bidirectional hidden states
        h_combined = torch.cat([forward_h, backward_h], dim=1)
        
        # Project to latent space
        mu = self.fc_mu(h_combined)
        logvar = self.fc_logvar(h_combined)
        
        return mu, logvar


class GlobalEncoder(nn.Module):
    """Encodes sequence of segment representations."""
    
    def __init__(self, segment_latent_dim: int, hidden_dim: int, 
                 global_latent_dim: int, condition_dim: int, 
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.global_latent_dim = global_latent_dim
        
        # LSTM for global encoding
        self.lstm = nn.LSTM(
            segment_latent_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True, dropout=dropout
        )
        
        # Global latent projections (conditioned)
        self.fc_mu = nn.Linear(hidden_dim * 2 + condition_dim, global_latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2 + condition_dim, global_latent_dim)
        
    def forward(self, segment_z: torch.Tensor, conditions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            segment_z: [batch_size, num_segments, segment_latent_dim]
            conditions: [batch_size, condition_dim]
        Returns:
            global_mu, global_logvar: [batch_size, global_latent_dim]
        """
        # Global LSTM encoding
        output, (h_n, _) = self.lstm(segment_z)
        
        # Combine bidirectional hidden states
        forward_h = h_n[-2]
        backward_h = h_n[-1]
        h_combined = torch.cat([forward_h, backward_h], dim=1)
        
        # Add conditions
        h_cond = torch.cat([h_combined, conditions], dim=1)
        
        # Project to global latent space
        global_mu = self.fc_mu(h_cond)
        global_logvar = self.fc_logvar(h_cond)
        
        return global_mu, global_logvar


class SegmentDecoder(nn.Module):
    """Decodes individual segments from local and global latents."""
    
    def __init__(self, segment_latent_dim: int, global_latent_dim: int,
                 hidden_dim: int, output_dim: int, segment_length: int,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.segment_length = segment_length
        self.num_layers = num_layers
        
        # Combine local and global latents
        self.latent_combiner = nn.Linear(
            segment_latent_dim + global_latent_dim, hidden_dim
        )
        
        # Hidden state initialization
        self.fc_h0 = nn.Linear(hidden_dim, num_layers * hidden_dim)
        self.fc_c0 = nn.Linear(hidden_dim, num_layers * hidden_dim)
        
        # Decoder LSTM
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout
        )
        
        # Output projection
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        
        # Learnable start token
        self.start_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.01)
        
    def forward(self, segment_z: torch.Tensor, global_z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            segment_z: [batch_size, segment_latent_dim]
            global_z: [batch_size, global_latent_dim]
        Returns:
            segment_recon: [batch_size, segment_length, output_dim]
        """
        batch_size = segment_z.size(0)
        
        # Combine latents
        combined_z = torch.cat([segment_z, global_z], dim=1)
        z_proj = self.latent_combiner(combined_z)
        
        # Initialize hidden states
        h0 = self.fc_h0(z_proj).view(self.num_layers, batch_size, self.hidden_dim)
        c0 = self.fc_c0(z_proj).view(self.num_layers, batch_size, self.hidden_dim)
        
        # Prepare decoder input
        decoder_input = self.start_token.repeat(batch_size, self.segment_length, 1)
        
        # Decode
        output, _ = self.lstm(decoder_input, (h0, c0))
        segment_recon = self.fc_out(output)
        
        return segment_recon


class HierarchicalTrajectoryVAE(BaseTrajectoryModel):
    """Hierarchical VAE for long trajectory sequences."""
    
    def __init__(self, input_dim=3, hidden_dim=256, segment_latent_dim=32, 
                 global_latent_dim=64, condition_dim=32, segment_length=100,
                 num_layers=2, dropout=0.1, num_transport_modes=5, 
                 sequence_length=2000, **kwargs):
        config = locals()
        config.pop('self')
        config.pop('kwargs')
        super().__init__(config)
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.segment_latent_dim = segment_latent_dim
        self.global_latent_dim = global_latent_dim
        self.condition_dim = condition_dim
        self.segment_length = segment_length
        self.sequence_length = sequence_length
        self.num_segments = sequence_length // segment_length
        
        # Condition embeddings
        self.transport_mode_embedding = nn.Embedding(num_transport_modes, condition_dim)
        self.length_projection = nn.Linear(1, condition_dim)
        
        # Hierarchical components
        self.segment_encoder = SegmentEncoder(
            input_dim, hidden_dim, segment_latent_dim, num_layers, dropout
        )
        
        self.global_encoder = GlobalEncoder(
            segment_latent_dim, hidden_dim, global_latent_dim, 
            condition_dim * 2, num_layers, dropout
        )
        
        self.segment_decoder = SegmentDecoder(
            segment_latent_dim, global_latent_dim, hidden_dim, 
            input_dim, segment_length, num_layers, dropout
        )
        
        # Global to local latent mapping
        self.global_to_local = nn.Sequential(
            nn.Linear(global_latent_dim + condition_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, segment_latent_dim * self.num_segments)
        )
        
    def get_conditions(self, transport_mode: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
        """Get condition embeddings."""
        mode_embed = self.transport_mode_embedding(transport_mode)
        length_normalized = length.unsqueeze(-1).float() / self.sequence_length
        length_embed = self.length_projection(length_normalized)
        return torch.cat([mode_embed, length_embed], dim=-1)
    
    def segment_trajectory(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Segment trajectory into chunks.
        
        Args:
            x: [batch_size, sequence_length, input_dim]
            mask: [batch_size, sequence_length]
        Returns:
            segments: [batch_size, num_segments, segment_length, input_dim]
            segment_masks: [batch_size, num_segments, segment_length]
        """
        batch_size, seq_len, input_dim = x.shape
        
        # Reshape into segments
        segments = x[:, :self.num_segments * self.segment_length].view(
            batch_size, self.num_segments, self.segment_length, input_dim
        )
        
        segment_masks = mask[:, :self.num_segments * self.segment_length].view(
            batch_size, self.num_segments, self.segment_length
        )
        
        return segments, segment_masks
    
    def reassemble_trajectory(self, segments: torch.Tensor) -> torch.Tensor:
        """
        Reassemble segments back into full trajectory.
        
        Args:
            segments: [batch_size, num_segments, segment_length, input_dim]
        Returns:
            trajectory: [batch_size, sequence_length, input_dim]
        """
        batch_size, num_segments, segment_length, input_dim = segments.shape
        return segments.view(batch_size, num_segments * segment_length, input_dim)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def encode(self, x: torch.Tensor, conditions: torch.Tensor, 
               mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """Hierarchical encoding."""
        batch_size = x.size(0)
        
        # Segment the trajectory
        segments, segment_masks = self.segment_trajectory(x, mask)
        
        # Encode each segment
        segment_mus = []
        segment_logvars = []
        segment_zs = []
        
        for i in range(self.num_segments):
            seg_mu, seg_logvar = self.segment_encoder(
                segments[:, i], segment_masks[:, i]
            )
            seg_z = self.reparameterize(seg_mu, seg_logvar)
            
            segment_mus.append(seg_mu)
            segment_logvars.append(seg_logvar)
            segment_zs.append(seg_z)
        
        # Stack segment representations
        segment_mus = torch.stack(segment_mus, dim=1)  # [batch_size, num_segments, segment_latent_dim]
        segment_logvars = torch.stack(segment_logvars, dim=1)
        segment_zs = torch.stack(segment_zs, dim=1)
        
        # Global encoding
        global_mu, global_logvar = self.global_encoder(segment_zs, conditions)
        global_z = self.reparameterize(global_mu, global_logvar)
        
        return {
            'segment_mus': segment_mus,
            'segment_logvars': segment_logvars,
            'segment_zs': segment_zs,
            'global_mu': global_mu,
            'global_logvar': global_logvar,
            'global_z': global_z
        }
    
    def decode(self, encoding_dict: Dict[str, torch.Tensor], 
               conditions: torch.Tensor) -> torch.Tensor:
        """Hierarchical decoding."""
        global_z = encoding_dict['global_z']
        batch_size = global_z.size(0)
        
        # Generate local latents from global context
        global_cond = torch.cat([global_z, conditions], dim=1)
        local_latents = self.global_to_local(global_cond)
        local_latents = local_latents.view(
            batch_size, self.num_segments, self.segment_latent_dim
        )
        
        # Decode each segment
        decoded_segments = []
        for i in range(self.num_segments):
            segment_recon = self.segment_decoder(
                local_latents[:, i], global_z
            )
            decoded_segments.append(segment_recon)
        
        # Stack and reassemble
        segments = torch.stack(decoded_segments, dim=1)
        return self.reassemble_trajectory(segments)
    
    def forward(self, x: torch.Tensor, transport_mode: torch.Tensor, 
                length: torch.Tensor, mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """Forward pass through hierarchical VAE."""
        conditions = self.get_conditions(transport_mode, length)
        
        # Encode
        encoding_dict = self.encode(x, conditions, mask)
        
        # Decode
        recon = self.decode(encoding_dict, conditions)
        
        # Return all necessary components for loss computation
        return {
            'recon': recon,
            'segment_mus': encoding_dict['segment_mus'],
            'segment_logvars': encoding_dict['segment_logvars'],
            'global_mu': encoding_dict['global_mu'],
            'global_logvar': encoding_dict['global_logvar'],
            'global_z': encoding_dict['global_z'],
            'conditions': conditions
        }
    
    def generate(self, conditions: Dict[str, torch.Tensor], n_samples: int) -> torch.Tensor:
        """Generate new trajectories."""
        transport_mode = conditions['transport_mode']
        length = conditions['length']
        
        device = transport_mode.device
        conditions_embed = self.get_conditions(transport_mode, length)
        
        with torch.no_grad():
            # Sample global latent
            global_z = torch.randn(n_samples, self.global_latent_dim, device=device)
            
            # Create encoding dict for decoding
            encoding_dict = {'global_z': global_z}
            
            # Decode
            trajectories = self.decode(encoding_dict, conditions_embed)
        
        return trajectories
