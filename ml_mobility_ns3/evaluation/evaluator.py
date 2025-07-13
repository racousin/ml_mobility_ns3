# ml_mobility_ns3/evaluation/evaluator.py
import torch
import numpy as np
from typing import Dict
import logging
from tqdm import tqdm
import pickle
from pathlib import Path

from ml_mobility_ns3.metrics.stat_metrics import StatMetrics

logger = logging.getLogger(__name__)


class TrajectoryEvaluator:
    """Simplified evaluator for trajectory generation models."""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = config.device if config.device != 'cuda' else 'cpu'
        self.model = self.model.to(self.device)
        self.stat_metrics = StatMetrics()
        
        # Load category encoder
        self.category_encoder = self._load_category_encoder()
        
        # Load real data statistics
        self.real_stats = self._load_real_statistics()
    
    def _load_category_encoder(self):
        """Load category encoder from scalers."""
        scaler_path = Path(self.config.data.output_dir) / 'scalers.pkl'
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                scalers = pickle.load(f)
                return scalers.get('category_encoder')
        return None
    
    def _load_real_statistics(self) -> Dict:
        """Load real data statistics from preprocessing."""
        report_path = Path(self.config.data.output_dir) / 'preprocessing_report.pkl'
        if report_path.exists():
            try:
                with open(report_path, 'rb') as f:
                    report = pickle.load(f)
                    return report.get('category_statistics', {})
            except Exception as e:
                logger.warning(f"Could not load preprocessing stats: {e}")
        return {}
    
    def evaluate_generation(self, dataloader) -> Dict:
        """
        Generate trajectories for entire dataset and compute statistics.
        
        Args:
            dataloader: DataLoader with all trajectories
            
        Returns:
            Dictionary with per-category statistics
        """
        self.model.eval()
        
        all_generated = []
        all_masks = []
        all_categories = []
        all_lengths = []
        
        logger.info("Generating trajectories for entire dataset...")
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Generating trajectories"):
                # Unpack batch
                x, mask, transport_mode, length = [b.to(self.device) for b in batch]
                batch_size = x.shape[0]
                
                # Prepare conditions from real data
                conditions = {
                    'transport_mode': transport_mode,
                    'length': length
                }
                
                # Generate trajectories
                try:
                    generated = self.model.generate(conditions, batch_size)
                    
                    # Store results
                    all_generated.append(generated.cpu())
                    all_masks.append(mask.cpu())
                    all_categories.append(transport_mode.cpu())
                    all_lengths.append(length.cpu())
                    
                except Exception as e:
                    logger.error(f"Error generating batch: {e}")
                    continue
        
        # Concatenate all results
        if not all_generated:
            logger.error("No trajectories generated")
            return {}
        
        generated_trajectories = torch.cat(all_generated, dim=0)
        masks = torch.cat(all_masks, dim=0)
        categories = torch.cat(all_categories, dim=0)
        lengths = torch.cat(all_lengths, dim=0)
        
        # Compute per-category statistics using shared function
        category_stats = self.stat_metrics.compute_category_statistics(
            generated_trajectories, masks, categories, lengths
        )
        
        return {
            'generated_stats': category_stats,
            'real_stats': self.real_stats,
            'n_generated': len(generated_trajectories)
        }