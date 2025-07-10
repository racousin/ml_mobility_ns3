#!/usr/bin/env python
# analyze_results.py
"""Comprehensive analysis script for trajectory generation experiments."""

import argparse
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Import model classes
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml_mobility_ns3.models.vae import ConditionalTrajectoryVAE
from ml_mobility_ns3.training.trainer import VAETrainer


class TrajectoryAnalyzer:
    """Comprehensive analyzer for trajectory generation results."""
    
    def __init__(self, results_dir: Path, data_path: Path, metadata_path: Optional[Path] = None):
        self.results_dir = Path(results_dir)
        self.data_path = Path(data_path)
        self.metadata_path = metadata_path or data_path.parent / "metadata.pkl"
        
        # Load original data
        self.load_original_data()
        
        # Storage for analysis results
        self.experiment_results = {}
        self.generation_results = {}
        
    def load_original_data(self):
        """Load the original dataset for comparison."""
        print("Loading original dataset...")
        
        data = np.load(self.data_path)
        with open(self.metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        self.trajectories = torch.from_numpy(data['trajectories']).float()
        self.masks = torch.from_numpy(data['masks']).bool()
        
        # Handle different possible column names
        if 'transport_modes' in data:
            self.transport_modes = torch.from_numpy(data['transport_modes']).long()
        elif 'categories' in data:
            self.transport_modes = torch.from_numpy(data['categories']).long()
        else:
            raise KeyError("Neither 'transport_modes' nor 'categories' found in data")
        
        if 'trip_lengths' in data:
            self.trip_lengths = torch.from_numpy(data['trip_lengths']).long()
        else:
            self.trip_lengths = self.masks.sum(dim=1).long()
        
        # Get transport mode names
        if 'categories' in metadata:
            self.transport_mode_names = metadata['categories']
        elif 'transport_modes' in metadata:
            self.transport_mode_names = metadata['transport_modes']
        else:
            self.transport_mode_names = [f"Mode_{i}" for i in range(self.transport_modes.max().item() + 1)]
        
        self.metadata = metadata
        print(f"Loaded {len(self.trajectories)} trajectories with {len(self.transport_mode_names)} transport modes")
    
    def load_experiment_results(self, experiment_name: str) -> Dict:
        """Load results from a specific experiment."""
        exp_dir = self.results_dir / experiment_name
        
        if not exp_dir.exists():
            raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")
        
        results = {}
        
        # Load configuration
        config_file = exp_dir / "config.json"
        if config_file.exists():
            with open(config_file, 'r') as f:
                results['config'] = json.load(f)
        
        # Load training history
        history_file = exp_dir / "history.json"
        if history_file.exists():
            with open(history_file, 'r') as f:
                results['history'] = json.load(f)
        
        # Load final results
        final_results_file = exp_dir / "final_results.json"
        if final_results_file.exists():
            with open(final_results_file, 'r') as f:
                results['final_results'] = json.load(f)
        
        # Load best model
        best_model_file = exp_dir / "best_model.pt"
        if best_model_file.exists():
            results['model_path'] = best_model_file
        
        return results
    
    def load_model(self, model_path: Path, device: str = 'cpu') -> ConditionalTrajectoryVAE:
        """Load a trained model from checkpoint."""
        checkpoint = torch.load(model_path, map_location=device)
        config = checkpoint['config']
        model = ConditionalTrajectoryVAE(**config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        return model
    
    def compute_trajectory_statistics(self, trajectories: torch.Tensor, masks: torch.Tensor, 
                                    transport_modes: torch.Tensor) -> Dict[str, Dict]:
        """Compute comprehensive trajectory statistics."""
        stats = {}
        
        for mode_idx, mode_name in enumerate(self.transport_mode_names):
            mode_mask = transport_modes == mode_idx
            if mode_mask.sum() == 0:
                continue
            
            mode_trajs = trajectories[mode_mask]
            mode_masks = masks[mode_mask]
            
            # Compute statistics for this mode
            mode_stats = self._compute_mode_statistics(mode_trajs, mode_masks)
            stats[mode_name] = mode_stats
        
        return stats
    
    def _compute_mode_statistics(self, trajectories: torch.Tensor, masks: torch.Tensor) -> Dict:
        """Compute statistics for a specific transport mode."""
        stats = {}
        
        # Speed statistics
        speeds = trajectories[:, :, 2]  # Assuming speed is the 3rd dimension
        valid_speeds = speeds[masks]
        
        stats['speed_mean'] = valid_speeds.mean().item()
        stats['speed_std'] = valid_speeds.std().item()
        stats['speed_min'] = valid_speeds.min().item()
        stats['speed_max'] = valid_speeds.max().item()
        
        # Distance statistics
        total_distances = []
        bird_distances = []
        
        for i in range(len(trajectories)):
            traj = trajectories[i]
            mask = masks[i]
            valid_length = mask.sum().item()
            
            if valid_length > 1:
                valid_traj = traj[:valid_length]
                
                # Total distance
                lat_diff = torch.diff(valid_traj[:, 0])
                lon_diff = torch.diff(valid_traj[:, 1])
                segments = torch.sqrt(lat_diff**2 + lon_diff**2) * 111  # Rough km conversion
                total_dist = segments.sum().item()
                total_distances.append(total_dist)
                
                # Bird distance
                start_point = valid_traj[0, :2]
                end_point = valid_traj[-1, :2]
                bird_dist = torch.sqrt(((end_point - start_point)**2).sum()).item() * 111
                bird_distances.append(bird_dist)
        
        if total_distances:
            stats['total_distance_mean'] = np.mean(total_distances)
            stats['total_distance_std'] = np.std(total_distances)
            stats['bird_distance_mean'] = np.mean(bird_distances)
            stats['bird_distance_std'] = np.std(bird_distances)
        else:
            stats['total_distance_mean'] = 0
            stats['total_distance_std'] = 0
            stats['bird_distance_mean'] = 0
            stats['bird_distance_std'] = 0
        
        # Length statistics
        lengths = masks.sum(dim=1).float()
        stats['length_mean'] = lengths.mean().item()
        stats['length_std'] = lengths.std().item()
        
        stats['num_trajectories'] = len(trajectories)
        
        return stats
    
    def evaluate_reconstruction(self, model: ConditionalTrajectoryVAE, device: str = 'cpu') -> Dict:
        """Evaluate reconstruction quality on the test set."""
        print("Evaluating reconstruction quality...")
        
        # Create dataloader
        dataset = TensorDataset(self.trajectories, self.masks, self.transport_modes, self.trip_lengths)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        model.eval()
        reconstruction_metrics = {
            'total_loss': 0,
            'recon_loss': 0,
            'kl_loss': 0,
            'speed_mae': 0,
            'total_distance_mae': 0,
            'bird_distance_mae': 0,
        }
        
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                x, mask, mode, length = [b.to(device) for b in batch]
                
                # Reconstruction
                recon, mu, logvar = model(x, mode, length, mask)
                
                # Compute losses (simplified - you might want to import the actual loss function)
                recon_error = ((recon - x) ** 2 * mask.unsqueeze(-1)).sum() / mask.sum()
                kl_error = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
                
                reconstruction_metrics['recon_loss'] += recon_error.item()
                reconstruction_metrics['kl_loss'] += kl_error.item()
                reconstruction_metrics['total_loss'] += (recon_error + kl_error).item()
                
                # Trajectory-specific metrics
                from ml_mobility_ns3.models.vae import compute_trajectory_metrics
                traj_metrics = compute_trajectory_metrics(recon, x, mask)
                
                for key in ['speed_mae', 'total_distance_mae', 'bird_distance_mae']:
                    reconstruction_metrics[key] += traj_metrics[key]
                
                num_batches += 1
        
        # Average the metrics
        for key in reconstruction_metrics:
            reconstruction_metrics[key] /= num_batches
        
        return reconstruction_metrics
    
    def evaluate_generation(self, model: ConditionalTrajectoryVAE, n_samples_per_mode: int = 500, 
                          device: str = 'cpu') -> Dict:
        """Evaluate generation quality by generating samples for each transport mode."""
        print(f"Evaluating generation quality with {n_samples_per_mode} samples per mode...")
        
        model.eval()
        
        # Compute statistics for original data
        original_stats = self.compute_trajectory_statistics(
            self.trajectories, self.masks, self.transport_modes
        )
        
        # Generate samples for each mode
        generated_stats = {}
        
        with torch.no_grad():
            for mode_idx, mode_name in enumerate(self.transport_mode_names):
                # Get average length for this mode
                mode_mask = self.transport_modes == mode_idx
                if mode_mask.sum() == 0:
                    continue
                
                mode_lengths = self.trip_lengths[mode_mask]
                avg_length = int(mode_lengths.float().mean().item())
                
                # Generate samples
                modes = torch.full((n_samples_per_mode,), mode_idx, dtype=torch.long, device=device)
                lengths = torch.full((n_samples_per_mode,), avg_length, dtype=torch.long, device=device)
                
                generated = model.generate(modes, lengths, device=device)
                
                # Create masks for generated data
                gen_masks = torch.zeros(n_samples_per_mode, generated.shape[1], dtype=torch.bool)
                for i in range(n_samples_per_mode):
                    gen_masks[i, :avg_length] = True
                
                # Compute statistics for generated data
                gen_mode_stats = self._compute_mode_statistics(generated.cpu(), gen_masks)
                generated_stats[mode_name] = gen_mode_stats
        
        # Compare original vs generated statistics
        comparison = {}
        for mode_name in original_stats:
            if mode_name in generated_stats:
                orig = original_stats[mode_name]
                gen = generated_stats[mode_name]
                
                comparison[mode_name] = {
                    'speed_mae': abs(orig['speed_mean'] - gen['speed_mean']),
                    'speed_std_diff': abs(orig['speed_std'] - gen['speed_std']),
                    'total_distance_mae': abs(orig['total_distance_mean'] - gen['total_distance_mean']),
                    'total_distance_std_diff': abs(orig['total_distance_std'] - gen['total_distance_std']),
                    'bird_distance_mae': abs(orig['bird_distance_mean'] - gen['bird_distance_mean']),
                    'bird_distance_std_diff': abs(orig['bird_distance_std'] - gen['bird_distance_std']),
                    'length_mae': abs(orig['length_mean'] - gen['length_mean']),
                }
        
        return {
            'original_stats': original_stats,
            'generated_stats': generated_stats,
            'comparison': comparison
        }
    
    def analyze_experiment(self, experiment_name: str, device: str = 'cpu') -> Dict:
        """Perform comprehensive analysis of a single experiment."""
        print(f"Analyzing experiment: {experiment_name}")
        
        # Load experiment results
        results = self.load_experiment_results(experiment_name)
        
        analysis = {
            'experiment_name': experiment_name,
            'config': results.get('config', {}),
            'training_history': results.get('history', {}),
        }
        
        # Load and evaluate model if available
        if 'model_path' in results:
            model = self.load_model(results['model_path'], device)
            
            # Evaluate reconstruction
            reconstruction_metrics = self.evaluate_reconstruction(model, device)
            analysis['reconstruction_evaluation'] = reconstruction_metrics
            
            # Evaluate generation
            generation_metrics = self.evaluate_generation(model, device=device)
            analysis['generation_evaluation'] = generation_metrics
        
        return analysis
    
    def compare_experiments(self, experiment_names: List[str], device: str = 'cpu') -> Dict:
        """Compare multiple experiments."""
        print(f"Comparing {len(experiment_names)} experiments...")
        
        comparisons = {}
        
        for exp_name in experiment_names:
            try:
                analysis = self.analyze_experiment(exp_name, device)
                comparisons[exp_name] = analysis
            except Exception as e:
                print(f"Error analyzing {exp_name}: {e}")
                continue
        
        return comparisons
    
    def plot_training_curves(self, experiment_names: List[str], save_path: Optional[Path] = None):
        """Plot training curves for multiple experiments."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        metrics = ['loss', 'recon_loss', 'kl_loss', 'speed_mae', 'total_distance_mae', 'bird_distance_mae']
        
        for exp_name in experiment_names:
            try:
                results = self.load_experiment_results(exp_name)
                history = results.get('history', {})
                
                for i, metric in enumerate(metrics):
                    train_key = f'train_{metric}'
                    val_key = f'val_{metric}'
                    
                    if train_key in history and val_key in history:
                        epochs = range(1, len(history[train_key]) + 1)
                        axes[i].plot(epochs, history[train_key], '--', alpha=0.7, label=f'{exp_name} (train)')
                        axes[i].plot(epochs, history[val_key], '-', label=f'{exp_name} (val)')
                        axes[i].set_title(f'{metric.replace("_", " ").title()}')
                        axes[i].set_xlabel('Epoch')
                        axes[i].legend()
                        axes[i].grid(True, alpha=0.3)
                        
            except Exception as e:
                print(f"Error plotting {exp_name}: {e}")
                continue
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training curves saved to {save_path}")
        
        plt.show()
    
    def plot_generation_comparison(self, experiment_names: List[str], save_path: Optional[Path] = None, device: str = 'cpu'):
        """Plot comparison of generation quality across experiments."""
        
        # Collect generation statistics
        comparison_data = []
        
        for exp_name in experiment_names:
            try:
                analysis = self.analyze_experiment(exp_name, device)
                gen_eval = analysis.get('generation_evaluation', {})
                comparison = gen_eval.get('comparison', {})
                
                for mode_name, metrics in comparison.items():
                    for metric_name, value in metrics.items():
                        comparison_data.append({
                            'experiment': exp_name,
                            'transport_mode': mode_name,
                            'metric': metric_name,
                            'value': value
                        })
            except Exception as e:
                print(f"Error processing {exp_name}: {e}")
                continue
        
        if not comparison_data:
            print("No comparison data available")
            return
        
        df = pd.DataFrame(comparison_data)
        
        # Create subplots for different metrics
        metrics = df['metric'].unique()
        n_metrics = len(metrics)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1 or n_cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            if i < len(axes):
                metric_data = df[df['metric'] == metric]
                sns.boxplot(data=metric_data, x='experiment', y='value', ax=axes[i])
                axes[i].set_title(f'{metric.replace("_", " ").title()}')
                axes[i].tick_params(axis='x', rotation=45)
        
        # Remove empty subplots
        for i in range(len(metrics), len(axes)):
            axes[i].remove()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Generation comparison saved to {save_path}")
        
        plt.show()
    
    def generate_report(self, experiment_names: List[str], output_dir: Path, device: str = 'cpu'):
        """Generate a comprehensive analysis report."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Generating comprehensive report for {len(experiment_names)} experiments...")
        
        # Perform comprehensive analysis
        comparisons = self.compare_experiments(experiment_names, device)
        
        # Save detailed results
        with open(output_dir / 'detailed_analysis.json', 'w') as f:
            json.dump(comparisons, f, indent=2)
        
        # Create summary table
        summary_data = []
        for exp_name, analysis in comparisons.items():
            row = {
                'experiment': exp_name,
                'latent_dim': analysis.get('config', {}).get('model_config', {}).get('latent_dim'),
                'hidden_dim': analysis.get('config', {}).get('model_config', {}).get('hidden_dim'),
                'num_layers': analysis.get('config', {}).get('model_config', {}).get('num_layers'),
                'beta': analysis.get('config', {}).get('training_config', {}).get('beta'),
            }
            
            # Add reconstruction metrics
            recon_eval = analysis.get('reconstruction_evaluation', {})
            for key, value in recon_eval.items():
                row[f'recon_{key}'] = value
            
            # Add generation metrics (average across modes)
            gen_eval = analysis.get('generation_evaluation', {})
            comparison = gen_eval.get('comparison', {})
            if comparison:
                for metric in ['speed_mae', 'total_distance_mae', 'bird_distance_mae']:
                    values = [mode_metrics.get(metric, 0) for mode_metrics in comparison.values()]
                    row[f'gen_{metric}_avg'] = np.mean(values) if values else 0
            
            summary_data.append(row)
        
        # Save summary table
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(output_dir / 'experiment_summary.csv', index=False)
        
        # Generate plots
        self.plot_training_curves(experiment_names, output_dir / 'training_curves.png')
        self.plot_generation_comparison(experiment_names, output_dir / 'generation_comparison.png', device)
        
        # Generate markdown report
        self._generate_markdown_report(summary_df, output_dir)
        
        print(f"Report generated in {output_dir}")
    
    def _generate_markdown_report(self, summary_df: pd.DataFrame, output_dir: Path):
        """Generate a markdown report."""
        
        report = f"""# Trajectory Generation Experiment Analysis Report

Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

Total experiments analyzed: {len(summary_df)}

## Best Experiments by Metric

### Best Reconstruction Loss
{summary_df.nsmallest(5, 'recon_total_loss')[['experiment', 'recon_total_loss', 'recon_speed_mae']].to_markdown(index=False)}

### Best Speed MAE (Reconstruction)
{summary_df.nsmallest(5, 'recon_speed_mae')[['experiment', 'recon_speed_mae', 'recon_total_loss']].to_markdown(index=False)}

### Best Generation Quality (Speed)
{summary_df.nsmallest(5, 'gen_speed_mae_avg')[['experiment', 'gen_speed_mae_avg', 'gen_total_distance_mae_avg']].to_markdown(index=False)}

## Configuration Analysis

### Latent Dimension Impact
{summary_df.groupby('latent_dim')['recon_total_loss'].agg(['mean', 'std', 'count']).to_markdown()}

### Hidden Dimension Impact
{summary_df.groupby('hidden_dim')['recon_total_loss'].agg(['mean', 'std', 'count']).to_markdown()}

### Beta Parameter Impact
{summary_df.groupby('beta')['recon_total_loss'].agg(['mean', 'std', 'count']).to_markdown()}

## Files Generated

- `detailed_analysis.json`: Complete analysis results
- `experiment_summary.csv`: Summary table of all experiments
- `training_curves.png`: Training curves comparison
- `generation_comparison.png`: Generation quality comparison

## Recommendations

Based on the analysis:
1. Best overall model: {summary_df.loc[summary_df['recon_total_loss'].idxmin(), 'experiment']}
2. Best for speed accuracy: {summary_df.loc[summary_df['recon_speed_mae'].idxmin(), 'experiment']}
3. Most balanced performance: {summary_df.loc[(summary_df['recon_total_loss'].rank() + summary_df['recon_speed_mae'].rank()).idxmin(), 'experiment']}
"""
        
        with open(output_dir / 'report.md', 'w') as f:
            f.write(report)


def main():
    parser = argparse.ArgumentParser(description="Analyze trajectory generation experiment results")
    parser.add_argument("--results-dir", type=Path, required=True, 
                        help="Directory containing experiment results")
    parser.add_argument("--data-path", type=Path, required=True,
                        help="Path to original dataset (.npz file)")
    parser.add_argument("--metadata-path", type=Path,
                        help="Path to metadata file (.pkl)")
    parser.add_argument("--experiments", nargs='+', 
                        help="List of experiment names to analyze. If not provided, will analyze all")
    parser.add_argument("--output-dir", type=Path, default=Path("analysis_results"),
                        help="Output directory for analysis results")
    parser.add_argument("--device", type=str, default='cpu',
                        help="Device to use for model evaluation")
    parser.add_argument("--generate-report", action='store_true',
                        help="Generate comprehensive report")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = TrajectoryAnalyzer(args.results_dir, args.data_path, args.metadata_path)
    
    # Get experiment names
    if args.experiments:
        experiment_names = args.experiments
    else:
        # Find all experiment directories
        experiment_names = [d.name for d in args.results_dir.iterdir() 
                          if d.is_dir() and (d / "config.json").exists()]
    
    if not experiment_names:
        print("No experiments found!")
        return
    
    print(f"Found {len(experiment_names)} experiments: {experiment_names}")
    
    if args.generate_report:
        analyzer.generate_report(experiment_names, args.output_dir, args.device)
    else:
        # Just analyze single experiment
        if len(experiment_names) == 1:
            analysis = analyzer.analyze_experiment(experiment_names[0], args.device)
            print(json.dumps(analysis, indent=2))
        else:
            # Compare experiments
            comparisons = analyzer.compare_experiments(experiment_names, args.device)
            
            # Print summary
            for exp_name, analysis in comparisons.items():
                recon_metrics = analysis.get('reconstruction_evaluation', {})
                print(f"\n{exp_name}:")
                print(f"  Reconstruction Loss: {recon_metrics.get('total_loss', 'N/A'):.4f}")
                print(f"  Speed MAE: {recon_metrics.get('speed_mae', 'N/A'):.4f}")


if __name__ == "__main__":
    main()