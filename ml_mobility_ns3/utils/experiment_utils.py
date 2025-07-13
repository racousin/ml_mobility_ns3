#!/usr/bin/env python
"""Clean script for listing experiments with metrics."""
import argparse
import sys
from pathlib import Path
from tabulate import tabulate
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from ml_mobility_ns3.utils.metrics_utils import MetricsCollector, MetricsFormatter, MetricsComparator
from ml_mobility_ns3.utils.experiment_utils import ExperimentManager


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