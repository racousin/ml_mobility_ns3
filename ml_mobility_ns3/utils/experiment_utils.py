#!/usr/bin/env python
"""Clean script for listing experiments with metrics."""
import argparse
import sys
from pathlib import Path
from tabulate import tabulate
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from ml_mobility_ns3.utils.experiment_utils import ExperimentManager


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

class ExperimentLister:
    """List and display experiments."""
    
    def __init__(self, experiments_dir: Path = Path("experiments")):
        self.experiments_dir = experiments_dir
        self.exp_manager = ExperimentManager()
        self.metrics_collector = MetricsCollector()
        
    def list_experiments(self, detailed: bool = False):
        """List all experiments with metrics."""
        # Collect all experiment metrics
        df = self.metrics_collector.aggregate_experiment_metrics(self.experiments_dir)
        
        if df.empty:
            print("No experiments found. Train a model first with: python scripts/train.py")
            return
        
        # Prepare display data
        if detailed:
            self._display_detailed(df)
        else:
            self._display_summary(df)
        
        # Show best models
        if detailed and len(df[df['status'] == 'completed']) > 0:
            self._display_best_models(df)
    
    def _display_summary(self, df: pd.DataFrame):
        """Display summary table."""
        # Select columns for display
        columns = [
            'experiment_id', 'model_type', 'status', 
            'val_loss', 'epochs_trained'
        ]
        
        # Format experiment IDs
        df['experiment_id'] = df['experiment_id'].apply(
            lambda x: x[:30] + "..." if len(x) > 30 else x
        )
        
        # Format numeric columns
        numeric_cols = ['val_loss']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].apply(MetricsFormatter.format_metric_value)
        
        # Select available columns
        available_cols = [col for col in columns if col in df.columns]
        display_df = df[available_cols]
        
        # Rename columns for display
        column_names = {
            'experiment_id': 'Experiment ID',
            'model_type': 'Model',
            'status': 'Status',
            'val_loss': 'Best Loss',
            'epochs_trained': 'Epochs'
        }
        display_df = display_df.rename(columns=column_names)
        
        print("\n=== Experiments ===")
        print(tabulate(display_df, headers='keys', tablefmt='grid', showindex=False))
        print(f"\nTotal experiments: {len(df)}")
    
    def _display_detailed(self, df: pd.DataFrame):
        """Display detailed table with all metrics."""
        # Define all columns to show
        columns = [
            'experiment_id', 'model_type', 'status',
            'val_loss', 'val_mse', 'val_kl_loss',
            'val_total_distance_mae', 'val_bird_distance_mae',
            'val_speed_mse', 'epochs_trained'
        ]
        
        # Format experiment IDs
        df['experiment_id'] = df['experiment_id'].apply(
            lambda x: x[:20] + "..." if len(x) > 20 else x
        )
        
        # Format numeric columns
        numeric_cols = [
            'val_loss', 'val_mse', 'val_kl_loss',
            'val_total_distance_mae', 'val_bird_distance_mae',
            'val_speed_mse'
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].apply(MetricsFormatter.format_metric_value)
        
        # Select available columns
        available_cols = [col for col in columns if col in df.columns]
        display_df = df[available_cols]
        
        # Rename columns for display
        column_names = {
            'experiment_id': 'ID',
            'model_type': 'Model',
            'status': 'Status',
            'val_loss': 'Loss',
            'val_mse': 'MSE',
            'val_kl_loss': 'KL',
            'val_total_distance_mae': 'Dist MAE',
            'val_bird_distance_mae': 'Bird MAE',
            'val_speed_mse': 'Speed MSE',
            'epochs_trained': 'Epochs'
        }
        display_df = display_df.rename(columns=column_names)
        
        print("\n=== Experiments (Detailed) ===")
        print(tabulate(display_df, headers='keys', tablefmt='grid', showindex=False))
        print(f"\nTotal experiments: {len(df)}")
    
    def _display_best_models(self, df: pd.DataFrame):
        """Display best models by metric."""
        best_experiments = MetricsComparator.find_best_experiments(df)
        
        if not best_experiments:
            return
        
        print("\n=== Best Models by Metric ===")
        
        for metric_name, experiments in best_experiments.items():
            if not experiments:
                continue
            
            print(f"\nBest by {metric_name}:")
            for i, exp in enumerate(experiments, 1):
                exp_id = exp['experiment_id']
                if len(exp_id) > 40:
                    exp_id = exp_id[:40] + "..."
                
                value = MetricsFormatter.format_metric_value(exp['value'])
                print(f"  {i}. {exp_id}")
                print(f"     Model: {exp['model_type']}, Value: {value}, Epochs: {exp['epochs']}")
    
    def compare_experiments(self, experiment_ids: List[str], metrics: List[str] = None):
        """Compare specific experiments."""
        if metrics is None:
            metrics = [
                'val_loss', 'val_mse', 'val_kl_loss',
                'val_total_distance_mae', 'val_bird_distance_mae'
            ]
        
        comparison_df = MetricsComparator.create_comparison_table(
            experiment_ids, metrics, self.experiments_dir
        )
        
        if comparison_df.empty:
            print("No valid experiments found for comparison.")
            return
        
        # Format numeric columns
        for col in metrics:
            if col in comparison_df.columns:
                comparison_df[col] = comparison_df[col].apply(
                    MetricsFormatter.format_metric_value
                )
        
        print("\n=== Experiment Comparison ===")
        print(tabulate(comparison_df, headers='keys', tablefmt='grid', showindex=False))


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="List experiments with metrics")
    parser.add_argument("--detailed", "-d", action="store_true", 
                       help="Show detailed metrics")
    parser.add_argument("--compare", "-c", nargs="+", 
                       help="Compare specific experiments by ID")
    parser.add_argument("--metrics", "-m", nargs="+",
                       help="Metrics to compare (with --compare)")
    
    args = parser.parse_args()
    
    lister = ExperimentLister()
    
    if args.compare:
        lister.compare_experiments(args.compare, args.metrics)
    else:
        lister.list_experiments(detailed=args.detailed)


if __name__ == "__main__":
    main()