import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict
from .base import BaseTrajectoryModel


class ConditionalTrajectoryVAEDense(BaseTrajectoryModel):
    """Simple dense VAE for trajectory generation with fully connected layers."""
    
    def __init__(self, input_dim=3, hidden_dim=256, latent_dim=32, 
                 condition_dim=32, dropout=0.1, num_transport_modes=5, 
                 sequence_length=2000, num_hidden_layers=2, **kwargs):
        config = locals()
        config.pop('self')
        config.pop('kwargs')
        super().__init__(config)
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.sequence_length = sequence_length
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        
        # Calculate flattened input size
        self.flattened_size = sequence_length * input_dim
        
        # Transport mode embedding
        self.transport_mode_embedding = nn.Embedding(num_transport_modes, condition_dim)
        self.length_projection = nn.Linear(1, condition_dim)
        
        # Encoder layers
        encoder_layers = []
        input_size = self.flattened_size + condition_dim * 2
        
        # First layer
        encoder_layers.extend([
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        ])
        
        # Hidden layers
        for _ in range(num_hidden_layers - 1):
            encoder_layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent projections
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder layers
        decoder_layers = []
        decoder_input_size = latent_dim + condition_dim * 2
        
        # First decoder layer
        decoder_layers.extend([
            nn.Linear(decoder_input_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        ])
        
        # Hidden layers
        for _ in range(num_hidden_layers - 1):
            decoder_layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
        
        # Output layer
        decoder_layers.append(nn.Linear(hidden_dim, self.flattened_size))
        
        self.decoder = nn.Sequential(*decoder_layers)
        
    def get_conditions(self, transport_mode: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
        """Get condition embeddings."""
        mode_embed = self.transport_mode_embedding(transport_mode)
        length_normalized = length.unsqueeze(-1).float() / self.sequence_length
        length_embed = self.length_projection(length_normalized)
        return torch.cat([mode_embed, length_embed], dim=-1)
    
    def encode(self, x: torch.Tensor, conditions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input sequence."""
        batch_size = x.size(0)
        
        # Flatten the sequence
        x_flat = x.view(batch_size, -1)
        
        # Concatenate with conditions
        x_cond = torch.cat([x_flat, conditions], dim=-1)
        
        # Pass through encoder
        hidden = self.encoder(x_cond)
        
        # Get latent parameters
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor, conditions: torch.Tensor, 
               target_length: int = None) -> torch.Tensor:
        """Decode latent representation."""
        batch_size = z.size(0)
        
        # Concatenate latent with conditions
        z_cond = torch.cat([z, conditions], dim=-1)
        
        # Pass through decoder
        output_flat = self.decoder(z_cond)
        
        # Reshape to sequence format
        seq_len = target_length or self.sequence_length
        output = output_flat.view(batch_size, seq_len, self.input_dim)
        
        return output
    
    def forward(self, x: torch.Tensor, transport_mode: torch.Tensor, 
                length: torch.Tensor, mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """Forward pass through the model."""
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
        """Generate new trajectories."""
        transport_mode = conditions['transport_mode']
        length = conditions['length']
        
        device = transport_mode.device
        z = torch.randn(n_samples, self.latent_dim, device=device)
        conditions_embed = self.get_conditions(transport_mode, length)
        
        with torch.no_grad():
            trajectories = self.decode(z, conditions_embed, target_length)
        
        return trajectories
    
    def get_model_size(self) -> Dict[str, int]:
        """Get model size information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'flattened_input_size': self.flattened_size,
            'model_type': 'dense_vae'
        }