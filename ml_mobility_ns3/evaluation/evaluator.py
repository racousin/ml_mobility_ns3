# ml_mobility_ns3/evaluation/evaluator.py
import torch
from pathlib import Path
import numpy as np
from typing import Dict, List
import logging
from tqdm import tqdm

from ..metrics.trajectory_metrics import TrajectoryMetrics

logger = logging.getLogger(__name__)


class TrajectoryEvaluator:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = config.device if config.device != 'cuda' else 'cpu'  # Use CPU for evaluation
        self.model = self.model.to(self.device)
        self.metrics = TrajectoryMetrics()
        
    def evaluate_reconstruction(self, dataloader) -> Dict:
        """Evaluate reconstruction quality."""
        self.model.eval()
        all_metrics = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating reconstruction"):
                # Move batch to device
                x, mask, transport_mode, length = [b.to(self.device) for b in batch]
                
                # Get model outputs
                outputs = self.model(x, transport_mode, length, mask)
                
                # Compute metrics
                metrics = self.metrics.compute_trajectory_mae(outputs['recon'], x, mask)
                all_metrics.append(metrics)
        
        # Aggregate metrics
        aggregated = {}
        if all_metrics:
            for key in all_metrics[0].keys():
                values = [m[key] for m in all_metrics]
                aggregated[f'mean_{key}'] = np.mean(values)
                aggregated[f'std_{key}'] = np.std(values)
        
        return aggregated
    
    def evaluate_generation(self, n_samples_per_mode: int = 100) -> Dict:
        """Evaluate generation quality."""
        self.model.eval()
        
        # Get number of transport modes from config
        if hasattr(self.config.model, 'num_transport_modes'):
            n_transport_modes = self.config.model.num_transport_modes
        else:
            n_transport_modes = 5  # Default
        
        generation_metrics = {
            'n_samples_per_mode': n_samples_per_mode,
            'n_transport_modes': n_transport_modes
        }
        
        # For each transport mode
        for mode in range(n_transport_modes):
            mode_name = f'mode_{mode}'
            
            # Generate samples
            conditions = {
                'transport_mode': torch.tensor([mode] * n_samples_per_mode).to(self.device),
                'length': torch.tensor([500] * n_samples_per_mode).to(self.device)  # Fixed length
            }
            
            with torch.no_grad():
                trajectories = self.model.generate(conditions, n_samples_per_mode)
            
            # Compute generation statistics
            # Create a dummy mask for the generated trajectories
            mask = torch.ones(n_samples_per_mode, trajectories.shape[1], dtype=torch.bool).to(self.device)
            
            # Compute metrics on generated data
            gen_metrics = self.metrics.compute_gps_metrics_torch(trajectories, mask)
            
            # Store aggregated metrics
            generation_metrics[f'{mode_name}_avg_speed'] = gen_metrics['avg_speed'].mean().item()
            generation_metrics[f'{mode_name}_avg_bird_distance'] = gen_metrics['bird_distance'].mean().item()
            generation_metrics[f'{mode_name}_avg_total_distance'] = gen_metrics['total_distance'].mean().item()
        
        return generation_metrics