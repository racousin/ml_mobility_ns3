# ml_mobility_ns3/utils/metrics_utils.py
"""Utilities for metrics handling and visualization."""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
import math

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collect and aggregate metrics from experiments."""
    
    @staticmethod
    def load_experiment_metrics(exp_dir: Path) -> Dict[str, Any]:
        """Load all available metrics for an experiment."""
        metrics = {}
        
        # Try loading best_metrics.json first (new format)
        best_metrics_file = exp_dir / "best_metrics.json"
        if best_metrics_file.exists():
            with open(best_metrics_file, "r") as f:
                best_metrics = json.load(f)
                metrics.update(best_metrics)
        
        # Try loading from model_info.json
        model_info_path = exp_dir / "model_info.json"
        if model_info_path.exists():
            with open(model_info_path, "r") as f:
                model_info = json.load(f)
                
                # Get metrics from different possible locations
                if 'best_metrics' in model_info:
                    metrics.update(model_info['best_metrics'])
                if 'key_metrics' in model_info:
                    for k, v in model_info['key_metrics'].items():
                        if k not in metrics and v is not None:
                            metrics[k] = v
                if 'final_metrics' in model_info:
                    final = model_info['final_metrics']
                    if 'best_val_loss' in final and 'val_loss' not in metrics:
                        metrics['val_loss'] = final['best_val_loss']
        
        # Fallback to tensorboard if needed
        if not metrics:
            metrics = MetricsCollector._load_tensorboard_metrics(exp_dir)
        
        return metrics
    
    @staticmethod
    def _load_tensorboard_metrics(exp_dir: Path) -> Dict[str, Any]:
        """Load metrics from tensorboard logs if available."""
        tb_dir = exp_dir / "logs" / "tensorboard"
        metrics = {}
        
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
            
            version_dirs = [d for d in tb_dir.iterdir() if d.is_dir()] if tb_dir.exists() else []
            if not version_dirs:
                event_files = list(tb_dir.glob("events.out.tfevents.*")) if tb_dir.exists() else []
                if event_files:
                    version_dirs = [tb_dir]
            
            for version_dir in version_dirs:
                event_files = list(version_dir.glob("events.out.tfevents.*"))
                if event_files:
                    ea = EventAccumulator(str(version_dir))
                    ea.Reload()
                    
                    # Get final validation metrics
                    metric_names = [
                        'val_mse', 'val_speed_mse', 'val_total_distance_mae', 
                        'val_bird_distance_mae', 'val_kl_loss', 'val_loss'
                    ]
                    
                    for metric in metric_names:
                        if metric in ea.scalars.Keys():
                            values = ea.scalars.Items(metric)
                            if values:
                                metrics[metric] = values[-1].value
                    
                    break  # Use first valid version
        except Exception as e:
            logger.debug(f"Could not load tensorboard metrics: {e}")
        
        return metrics
    
    @staticmethod
    def aggregate_experiment_metrics(experiments_dir: Path = Path("experiments")) -> pd.DataFrame:
        """Aggregate metrics from all experiments into a DataFrame."""
        all_metrics = []
        
        for exp_dir in experiments_dir.iterdir():
            if not exp_dir.is_dir():
                continue
            
            # Load experiment info
            model_info_path = exp_dir / "model_info.json"
            if not model_info_path.exists():
                continue
            
            with open(model_info_path, "r") as f:
                model_info = json.load(f)
            
            # Load metrics
            metrics = MetricsCollector.load_experiment_metrics(exp_dir)
            
            # Combine info
            experiment_data = {
                'experiment_id': exp_dir.name,
                'model_type': model_info.get('model_type', 'unknown'),
                'status': model_info.get('status', 'unknown'),
                'created_at': model_info.get('created_at'),
                'completed_at': model_info.get('completed_at'),
                'epochs_trained': model_info.get('final_metrics', {}).get('epochs_trained'),
                **metrics
            }
            
            all_metrics.append(experiment_data)
        
        return pd.DataFrame(all_metrics)


class MetricsFormatter:
    """Format metrics for display."""
    
    @staticmethod
    def format_metric_value(value: Any, decimals: int = 4) -> str:
        """Format a metric value for display."""
        if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
            return "N/A"
        if isinstance(value, (int, float)):
            return f"{value:.{decimals}f}"
        return str(value)
    
    @staticmethod
    def format_metric_name(name: str) -> str:
        """Format metric name for display."""
        # Remove prefix
        if name.startswith('val_'):
            name = name[4:]
        elif name.startswith('train_'):
            name = name[6:]
        
        # Convert to title case
        name = name.replace('_', ' ').title()
        
        # Special cases
        replacements = {
            'Mse': 'MSE',
            'Mae': 'MAE',
            'Kl': 'KL',
            'Kmh': 'km/h',
            'Km': 'km'
        }
        
        for old, new in replacements.items():
            name = name.replace(old, new)
        
        return name
    
    @staticmethod
    def create_metrics_summary(metrics: Dict[str, Any], 
                             key_metrics: Optional[List[str]] = None) -> str:
        """Create a formatted summary of metrics."""
        if key_metrics is None:
            key_metrics = [
                'val_loss', 'val_mse', 'val_kl_loss',
                'val_total_distance_mae', 'val_bird_distance_mae',
                'val_speed_mse'
            ]
        
        lines = []
        for metric in key_metrics:
            if metric in metrics:
                name = MetricsFormatter.format_metric_name(metric)
                value = MetricsFormatter.format_metric_value(metrics[metric])
                lines.append(f"{name}: {value}")
        
        return "\n".join(lines)


class MetricsComparator:
    """Compare metrics across experiments."""
    
    @staticmethod
    def find_best_experiments(df: pd.DataFrame, 
                            metrics_to_minimize: Optional[Dict[str, str]] = None) -> Dict[str, List[Dict]]:
        """Find best experiments for each metric."""
        if metrics_to_minimize is None:
            metrics_to_minimize = {
                'val_loss': 'Validation Loss',
                'val_mse': 'MSE',
                'val_kl_loss': 'KL Loss',
                'val_total_distance_mae': 'Total Distance MAE',
                'val_bird_distance_mae': 'Bird Distance MAE',
                'val_speed_mse': 'Speed MSE'
            }
        
        best_experiments = {}
        
        for metric_key, metric_name in metrics_to_minimize.items():
            if metric_key not in df.columns:
                continue
            
            # Filter valid values
            valid_df = df[df[metric_key].notna() & 
                          df['status'].eq('completed')].copy()
            
            if valid_df.empty:
                continue
            
            # Sort by metric
            valid_df = valid_df.sort_values(metric_key)
            
            # Get top 3
            top_experiments = []
            for _, row in valid_df.head(3).iterrows():
                top_experiments.append({
                    'experiment_id': row['experiment_id'],
                    'model_type': row['model_type'],
                    'value': row[metric_key],
                    'epochs': row.get('epochs_trained', 'N/A')
                })
            
            best_experiments[metric_name] = top_experiments
        
        return best_experiments
    
    @staticmethod
    def create_comparison_table(experiments: List[str], 
                              metrics: List[str],
                              experiments_dir: Path = Path("experiments")) -> pd.DataFrame:
        """Create a comparison table for specific experiments and metrics."""
        data = []
        
        for exp_id in experiments:
            exp_dir = experiments_dir / exp_id
            if not exp_dir.exists():
                continue
            
            # Load metrics
            exp_metrics = MetricsCollector.load_experiment_metrics(exp_dir)
            
            row = {'experiment_id': exp_id}
            for metric in metrics:
                row[metric] = exp_metrics.get(metric)
            
            data.append(row)
        
        return pd.DataFrame(data)