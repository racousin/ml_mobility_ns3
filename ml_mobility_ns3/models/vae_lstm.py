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
        
        # Transport mode embedding
        self.transport_mode_embedding = nn.Embedding(num_transport_modes, condition_dim)
        self.length_projection = nn.Linear(1, condition_dim)
        
        # Encoder
        self.encoder_lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True, dropout=dropout
        )
        
        # Latent projections
        self.fc_mu = nn.Linear(hidden_dim * 2 + condition_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2 + condition_dim * 2, latent_dim)
        
        # Decoder
        self.fc_latent_h = nn.Linear(latent_dim + condition_dim * 2, hidden_dim * num_layers)
        self.fc_latent_c = nn.Linear(latent_dim + condition_dim * 2, hidden_dim * num_layers)
        self.decoder_start_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.01)
        
        self.decoder_lstm = nn.LSTM(
            hidden_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout
        )
        self.fc_out = nn.Linear(hidden_dim, input_dim)
        
    def get_conditions(self, transport_mode: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
        mode_embed = self.transport_mode_embedding(transport_mode)
        length_normalized = length.unsqueeze(-1).float() / self.sequence_length
        length_embed = self.length_projection(length_normalized)
        return torch.cat([mode_embed, length_embed], dim=-1)
    
    def encode(self, x: torch.Tensor, conditions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, (h, _) = self.encoder_lstm(x)
        h = torch.cat([h[-2], h[-1]], dim=1)
        h_cond = torch.cat([h, conditions], dim=-1)
        mu = self.fc_mu(h_cond)
        logvar = self.fc_logvar(h_cond)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        batch_size = z.size(0)
        z_cond = torch.cat([z, conditions], dim=-1)
        
        h_0 = self.fc_latent_h(z_cond).view(self.num_layers, batch_size, self.hidden_dim)
        c_0 = self.fc_latent_c(z_cond).view(self.num_layers, batch_size, self.hidden_dim)
        
        decoder_input = self.decoder_start_token.repeat(batch_size, self.sequence_length, 1)
        out, _ = self.decoder_lstm(decoder_input, (h_0, c_0))
        return self.fc_out(out)
    
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