import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict
from .base import BaseTrajectoryModel


class ConditionalTrajectoryVAE(BaseTrajectoryModel):
    def __init__(self, input_dim=3, hidden_dim=256, latent_dim=32, 
                 num_layers=2, condition_dim=32, dropout=0.1, 
                 num_transport_modes=5, sequence_length=2000, **kwargs):
        config = locals()
        config.pop('self')
        config.pop('kwargs')
        super().__init__(config)
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.condition_dim = condition_dim
        self.sequence_length = sequence_length
        self.dropout = dropout
        
        # Transport mode embedding
        self.transport_mode_embedding = nn.Embedding(num_transport_modes, condition_dim)
        self.length_projection = nn.Linear(1, condition_dim)
        
        # Total condition dimension (transport mode + length)
        total_condition_dim = condition_dim * 2
        
        # Encoder - LSTM with bidirectional processing
        self.encoder_lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Latent projections (include conditions)
        encoder_output_dim = hidden_dim * 2  # bidirectional
        self.fc_mu = nn.Linear(encoder_output_dim + total_condition_dim, latent_dim)
        self.fc_logvar = nn.Linear(encoder_output_dim + total_condition_dim, latent_dim)
        
        # Decoder - following first implementation approach
        self.fc_latent = nn.Linear(latent_dim + total_condition_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(
            hidden_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.fc_out = nn.Linear(hidden_dim, input_dim)
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.uniform_(param, -0.1, 0.1)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def get_conditions(self, transport_mode: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
        # Transport mode embedding
        mode_embed = self.transport_mode_embedding(transport_mode)  # (batch, condition_dim)
        
        # Length embedding (normalize length and project)
        length_normalized = length.unsqueeze(-1).float() / self.sequence_length  # normalize to [0,1]
        length_embed = self.length_projection(length_normalized)  # (batch, condition_dim)
        
        # Concatenate conditions
        conditions = torch.cat([mode_embed, length_embed], dim=-1)  # (batch, condition_dim * 2)
        return conditions
    
    def encode(self, x: torch.Tensor, conditions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # LSTM encoding
        _, (h, _) = self.encoder_lstm(x)
        # Concatenate forward and backward hidden states from last layer
        h = torch.cat([h[-2], h[-1]], dim=1)  # (batch, hidden_dim * 2)
        h = self.dropout_layer(h)
        
        # Concatenate with conditions
        h_conditioned = torch.cat([h, conditions], dim=-1)
        
        mu = self.fc_mu(h_conditioned)
        logvar = self.fc_logvar(h_conditioned)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        batch_size = z.size(0)
        
        # Concatenate latent with conditions
        z_conditioned = torch.cat([z, conditions], dim=-1)
        
        h = self.fc_latent(z_conditioned)
        h = torch.tanh(h)  # Add non-linearity
        h = h.unsqueeze(1).repeat(1, self.sequence_length, 1)
        
        # Decode sequence
        out, _ = self.decoder_lstm(h)
        out = self.dropout_layer(out)
        out = self.fc_out(out)
        
        return out
    
    def forward(self, x: torch.Tensor, transport_mode: torch.Tensor, 
                length: torch.Tensor, mask: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        conditions = self.get_conditions(transport_mode, length)
        mu, logvar = self.encode(x, conditions)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, conditions)
        return {'recon': recon, 'mu': mu, 'logvar': logvar, 'z': z}
    
    def generate(self, conditions: Dict[str, torch.Tensor], n_samples: int) -> torch.Tensor:
        transport_mode = conditions['transport_mode']
        length = conditions['length']
        
        device = transport_mode.device
        z = torch.randn(n_samples, self.latent_dim, device=device)
        conditions_embed = self.get_conditions(transport_mode, length)
        
        with torch.no_grad():
            trajectories = self.decode(z, conditions_embed)
        return trajectories