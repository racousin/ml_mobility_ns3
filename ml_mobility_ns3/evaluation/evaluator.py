# ml_mobility_ns3/evaluation/evaluator.py
import torch
from pathlib import Path
import numpy as np
from typing import Dict, List
import logging
from tqdm import tqdm
import inspect

from ml_mobility_ns3.metrics.trajectory_metrics import TrajectoryMetrics

logger = logging.getLogger(__name__)


class TrajectoryEvaluator:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = config.device if config.device != 'cuda' else 'cpu'  # Use CPU for evaluation
        self.model = self.model.to(self.device)
        self.metrics = TrajectoryMetrics()
        
        # Check what arguments the model's forward method accepts
        self.model_forward_sig = inspect.signature(self.model.forward)
        self.model_params = list(self.model_forward_sig.parameters.keys())
        
    def evaluate_reconstruction(self, dataloader) -> Dict:
        """Evaluate reconstruction quality with standardized metrics."""
        self.model.eval()
        all_metrics = []
        all_std_metrics = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating reconstruction"):
                # Move batch to device
                x, mask, transport_mode, length = [b.to(self.device) for b in batch]
                
                # Prepare arguments based on what the model accepts
                model_args = {'x': x}
                if 'transport_mode' in self.model_params:
                    model_args['transport_mode'] = transport_mode
                if 'length' in self.model_params:
                    model_args['length'] = length
                if 'mask' in self.model_params:
                    model_args['mask'] = mask
                
                # Get model outputs
                outputs = self.model(**model_args)
                
                # Compute legacy metrics for backward compatibility
                metrics = self.metrics.compute_trajectory_mae(outputs['recon'], x, mask)
                all_metrics.append(metrics)
                
                # Compute standardized metrics
                std_metrics = self.metrics.compute_comprehensive_metrics(outputs['recon'], x, mask)
                # Convert tensors to floats
                std_metrics_dict = {k: v.item() if torch.is_tensor(v) else v 
                                   for k, v in std_metrics.items()}
                all_std_metrics.append(std_metrics_dict)
        
        # Aggregate metrics
        aggregated = {}
        
        # Legacy metrics
        if all_metrics:
            for key in all_metrics[0].keys():
                values = [m[key] for m in all_metrics]
                aggregated[f'mean_{key}'] = np.mean(values)
                aggregated[f'std_{key}'] = np.std(values)
        
        # Standardized metrics
        if all_std_metrics:
            std_keys = ['mse', 'speed_mse', 'total_distance_mae', 
                       'bird_distance_mae', 'frechet_distance']
            for key in std_keys:
                values = [m[key] for m in all_std_metrics if key in m]
                if values:
                    aggregated[f'{key}_mean'] = np.mean(values)
                    aggregated[f'{key}_std'] = np.std(values)
        
        return aggregated
    
    def evaluate_generation(self, n_samples_per_mode: int = 100) -> Dict:
        """Evaluate generation quality."""
        self.model.eval()
        
        # Get number of transport modes from config or model
        if hasattr(self.config.model, 'num_transport_modes'):
            n_transport_modes = self.config.model.num_transport_modes
        elif hasattr(self.model, 'num_transport_modes'):
            n_transport_modes = self.model.num_transport_modes
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