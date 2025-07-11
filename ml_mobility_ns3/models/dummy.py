import torch
import torch.nn as nn
from .base import BaseTrajectoryModel


class DummyModel(BaseTrajectoryModel):
    def __init__(self, input_dim=3, sequence_length=2000, **kwargs):
        config = locals()
        config.pop('self')
        config.pop('kwargs')
        super().__init__(config)
        
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        
        # Add at least one learnable parameter to avoid empty parameter list
        self.dummy_param = nn.Parameter(torch.ones(1))
        
    def forward(self, x: torch.Tensor, *args, **kwargs):
        # Simple pass-through with dummy parameter to ensure gradients flow
        return {
            'recon': x * self.dummy_param, 
            'mu': torch.zeros(x.size(0), 1), 
            'logvar': torch.zeros(x.size(0), 1)
        }
    
    def generate(self, conditions, n_samples):
        # Generate random trajectories
        return torch.randn(n_samples, self.sequence_length, self.input_dim) * self.dummy_param