import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple


class BaseTrajectoryModel(nn.Module, ABC):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
    @abstractmethod
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        pass
    
    @abstractmethod
    def generate(self, conditions: Dict[str, torch.Tensor], n_samples: int) -> torch.Tensor:
        pass
    
    def get_config(self) -> Dict[str, Any]:
        return dict(self.config)