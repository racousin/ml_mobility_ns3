#!/usr/bin/env python
# analyze_results.py
"""Tools for analyzing VAE/VAE-GAN training results."""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from typing import Dict, List, Optional, Tuple
import torch

# Set style
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class TrainingAnalyzer:
    """Analyze training results from VAE/VAE-GAN experiments."""
    
    def __init__(self, results_dir: Path):
        self.results_dir = Path(results_dir)
        self.experiments = self._load_experiments()
        
    def _load_experiments(self) -> Dict[str, Dict]:
        """Load all experiments from results directory."""
        experiments = {}
        
        for exp_dir in self.results_dir.iterdir():
            if exp_dir.is_dir():
                exp_data = {}
                
                # Check if this is a nested structure (look for subdirectory with same base name)
                # e.g., lstm_vae_small/lstm_vae/
                nested_dirs = [d for d in exp_dir.iterdir() if d.is_dir()]
                
                # Determine the actual data directory
                data_dir = exp_dir
                if nested_dirs:
                    # Check if there's a subdirectory that looks like it contains the actual data
                    for nested in nested_dirs:
                        if (nested / "history.json").exists():
                            data_dir = nested
                            print(f"Found nested structure: {exp_dir.name} -> {nested.name}")
                            break
                
                # Load experiment info (might be in parent dir)
                info_file = exp_dir / "experiment_info.json"
                if info_file.exists():
                    try:
                        with open(info_file) as f:
                            exp_data['info'] = json.load(f)
                    except (json.JSONDecodeError, IOError) as e:
                        print(f"Warning: Failed to load {info_file}: {e}")
                        # Don't continue here - we might still have valid data in nested dir
                
                # Load training history from data directory
                history_file = data_dir / "history.json"
                if history_file.exists():
                    try:
                        with open(history_file) as f:
                            exp_data['history'] = json.load(f)
                    except (json.JSONDecodeError, IOError) as e:
                        print(f"Warning: Failed to load {history_file}: {e}")
                        continue
                
                # Load config from data directory
                config_file = data_dir / "config.json"
                if config_file.exists():
                    try:
                        with open(config_file) as f:
                            exp_data['config'] = json.load(f)
                    except (json.JSONDecodeError, IOError) as e:
                        print(f"Warning: Failed to load {config_file}: {e}")
                        # Config is optional, so we can continue
                
                # Only add experiment if we have at least history data
                if 'history' in exp_data:
                    experiments[exp_dir.name] = exp_data
                    experiments[exp_dir.name]['data_dir'] = str(data_dir)
                    print(f"Loaded experiment: {exp_dir.name}")
                else:
                    print(f"Skipping {exp_dir.name} - no valid history data")
                    
        print(f"\nSuccessfully loaded {len(experiments)} experiments")
        return experiments
    
    def plot_training_curves(self, experiment_names: Optional[List[str]] = None, 
                           metrics: List[str] = ['loss', 'recon_loss', 'kl_loss'],
                           save_path: Optional[Path] = None):
        """Plot training curves for specified experiments."""
        if experiment_names is None:
            experiment_names = list(self.experiments.keys())
        
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 5))
        if n_metrics == 1:
            axes = [axes]
        
        for exp_name in experiment_names:
            if exp_name not in self.experiments:
                print(f"Warning: Experiment '{exp_name}' not found")
                continue
                
            history = self.experiments[exp_name].get('history', {})
            
            for idx, metric in enumerate(metrics):
                ax = axes[idx]
                
                # Plot training and validation curves
                train_key = f'train_{metric}'
                val_key = f'val_{metric}'
                
                if train_key in history:
                    epochs = np.arange(1, len(history[train_key]) + 1)
                    ax.plot(epochs, history[train_key], 
                           label=f'{exp_name} (train)', alpha=0.7, linestyle='--')
                
                if val_key in history:
                    epochs = np.arange(1, len(history[val_key]) + 1)
                    ax.plot(epochs, history[val_key], 
                           label=f'{exp_name} (val)', alpha=0.9)
                
                ax.set_xlabel('Epoch')
                ax.set_ylabel(metric.replace('_', ' ').title())
                ax.set_title(f'{metric.replace("_", " ").title()} Curves')
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved training curves to: {save_path}")
        else:
            plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
            print("Saved training curves to: training_curves.png")
    
    def plot_learning_rates(self, experiment_names: Optional[List[str]] = None,
                          save_path: Optional[Path] = None):
        """Plot learning rate schedules."""
        if experiment_names is None:
            experiment_names = list(self.experiments.keys())
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for exp_name in experiment_names:
            if exp_name not in self.experiments:
                continue
                
            history = self.experiments[exp_name].get('history', {})
            
            if 'learning_rates' in history:
                epochs = np.arange(1, len(history['learning_rates']) + 1)
                ax.semilogy(epochs, history['learning_rates'], 
                          label=f'{exp_name} (VAE)', marker='o', markersize=4)
            
            if 'disc_learning_rates' in history:
                epochs = np.arange(1, len(history['disc_learning_rates']) + 1)
                ax.semilogy(epochs, history['disc_learning_rates'], 
                          label=f'{exp_name} (Disc)', marker='s', markersize=4, 
                          linestyle='--')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved learning rates to: {save_path}")
        else:
            plt.savefig('learning_rates.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
            print("Saved learning rates to: learning_rates.png")
    
    def compare_final_metrics(self, save_path: Optional[Path] = None):
        """Create a comparison table of final metrics across experiments."""
        data = []
        
        for exp_name, exp_data in self.experiments.items():
            history = exp_data.get('history', {})
            config = exp_data.get('config', {})
            
            if not history or 'val_loss' not in history or len(history['val_loss']) == 0:
                print(f"Warning: Skipping {exp_name} - no validation loss data")
                continue
            
            row = {
                'Experiment': exp_name,
                'Architecture': config.get('architecture', 'N/A'),
                'Model Type': config.get('model_type', 'vae'),
                'Hidden Dim': config.get('model_config', {}).get('hidden_dim', 'N/A'),
                'Latent Dim': config.get('model_config', {}).get('latent_dim', 'N/A'),
                'Beta': config.get('training_config', {}).get('beta', 'N/A'),
                'Epochs': len(history.get('val_loss', [])),
                'Final Val Loss': history['val_loss'][-1] if history.get('val_loss') else None,
                'Best Val Loss': min(history['val_loss']) if history.get('val_loss') else None,
                'Final Recon Loss': history['val_recon_loss'][-1] if history.get('val_recon_loss') else None,
                'Final KL Loss': history['val_kl_loss'][-1] if history.get('val_kl_loss') else None,
            }
            
            # Add GAN metrics if available
            if 'val_gen_loss' in history:
                row['Final Gen Loss'] = history['val_gen_loss'][-1]
                row['Final Disc Loss'] = history['val_disc_loss'][-1]
            
            data.append(row)
        
        df = pd.DataFrame(data)
        
        if df.empty:
            print("Warning: No valid experiments found with metrics to compare")
            return df
            
        df = df.sort_values('Best Val Loss')
        
        # Display table
        print("\n=== Experiment Comparison ===")
        print(df.to_string(index=False))
        
        if save_path:
            # Save as CSV
            csv_path = save_path.with_suffix('.csv')
            df.to_csv(csv_path, index=False)
            print(f"\nTable saved to: {csv_path}")
            
            # Create a visual comparison
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Best validation loss comparison
            ax = axes[0, 0]
            df_sorted = df.sort_values('Best Val Loss').head(10)
            ax.barh(df_sorted['Experiment'], df_sorted['Best Val Loss'])
            ax.set_xlabel('Best Validation Loss')
            ax.set_title('Top 10 Models by Validation Loss')
            
            # Architecture comparison
            ax = axes[0, 1]
            arch_data = df.groupby('Architecture')['Best Val Loss'].mean()
            ax.bar(arch_data.index, arch_data.values)
            ax.set_xlabel('Architecture')
            ax.set_ylabel('Mean Best Val Loss')
            ax.set_title('Performance by Architecture')
            
            # Beta parameter analysis
            ax = axes[1, 0]
            beta_experiments = df[df['Experiment'].str.contains('beta')]
            if not beta_experiments.empty:
                ax.scatter(beta_experiments['Beta'], beta_experiments['Best Val Loss'])
                ax.set_xlabel('Beta')
                ax.set_ylabel('Best Val Loss')
                ax.set_xscale('log')
                ax.set_title('Beta Parameter Impact')
            
            # Latent dimension analysis
            ax = axes[1, 1]
            latent_experiments = df[df['Experiment'].str.contains('latent')]
            if not latent_experiments.empty:
                ax.scatter(latent_experiments['Latent Dim'], latent_experiments['Best Val Loss'])
                ax.set_xlabel('Latent Dimension')
                ax.set_ylabel('Best Val Loss')
                ax.set_title('Latent Dimension Impact')
            
            plt.tight_layout()
            plt.savefig(save_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Plots saved to: {save_path.with_suffix('.png')}")
        
        return df
    
    def plot_gan_metrics(self, experiment_names: Optional[List[str]] = None,
                        save_path: Optional[Path] = None):
        """Plot GAN-specific metrics."""
        if experiment_names is None:
            # Filter only GAN experiments
            experiment_names = [name for name, data in self.experiments.items() 
                              if 'gan' in name.lower() or 
                              'val_gen_loss' in data.get('history', {})]
        
        if not experiment_names:
            print("No GAN experiments found")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        for exp_name in experiment_names:
            if exp_name not in self.experiments:
                continue
                
            history = self.experiments[exp_name].get('history', {})
            
            # Generator loss
            ax = axes[0, 0]
            if 'val_gen_loss' in history:
                epochs = np.arange(1, len(history['val_gen_loss']) + 1)
                ax.plot(epochs, history['val_gen_loss'], label=exp_name)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Generator Loss')
            ax.set_title('Generator Loss')
            ax.legend()
            
            # Discriminator loss
            ax = axes[0, 1]
            if 'val_disc_loss' in history:
                epochs = np.arange(1, len(history['val_disc_loss']) + 1)
                ax.plot(epochs, history['val_disc_loss'], label=exp_name)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Discriminator Loss')
            ax.set_title('Discriminator Loss')
            ax.legend()
            
            # Discriminator accuracy on real vs fake
            ax = axes[1, 0]
            if 'val_disc_real' in history and 'val_disc_fake' in history:
                epochs = np.arange(1, len(history['val_disc_real']) + 1)
                ax.plot(epochs, history['val_disc_real'], 
                       label=f'{exp_name} (real)', linestyle='-')
                ax.plot(epochs, history['val_disc_fake'], 
                       label=f'{exp_name} (fake)', linestyle='--')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Discriminator Loss')
            ax.set_title('Discriminator Performance')
            ax.legend()
            
            # VAE loss vs GAN loss
            ax = axes[1, 1]
            if 'val_loss' in history and 'val_gen_loss' in history:
                epochs = np.arange(1, len(history['val_loss']) + 1)
                vae_component = np.array(history['val_loss']) - np.array(history['val_gen_loss'])
                ax.plot(epochs, vae_component, label=f'{exp_name} (VAE)')
                ax.plot(epochs, history['val_gen_loss'], label=f'{exp_name} (GAN)')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('VAE vs GAN Components')
            ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved GAN metrics to: {save_path}")
        else:
            plt.savefig('gan_metrics.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
            print("Saved GAN metrics to: gan_metrics.png")
    
    def analyze_early_stopping(self):
        """Analyze early stopping behavior across experiments."""
        data = []
        
        for exp_name, exp_data in self.experiments.items():
            history = exp_data.get('history', {})
            config = exp_data.get('config', {})
            
            if not history or 'val_loss' not in history:
                continue
            
            val_losses = history['val_loss']
            best_epoch = np.argmin(val_losses) + 1
            total_epochs = len(val_losses)
            
            # Check if early stopping was triggered
            early_stopped = total_epochs < config.get('training_config', {}).get('epochs', 100)
            
            data.append({
                'Experiment': exp_name,
                'Best Epoch': best_epoch,
                'Total Epochs': total_epochs,
                'Early Stopped': early_stopped,
                'Epochs After Best': total_epochs - best_epoch,
                'Best Val Loss': min(val_losses),
                'Final Val Loss': val_losses[-1],
                'Overfitting': val_losses[-1] - min(val_losses)
            })
        
        df = pd.DataFrame(data)
        
        print("\n=== Early Stopping Analysis ===")
        print(f"Experiments with early stopping: {df['Early Stopped'].sum()}/{len(df)}")
        print(f"Average best epoch: {df['Best Epoch'].mean():.1f}")
        print(f"Average total epochs: {df['Total Epochs'].mean():.1f}")
        print(f"Average epochs after best: {df['Epochs After Best'].mean():.1f}")
        print(f"Average overfitting (final - best loss): {df['Overfitting'].mean():.4f}")
        
        return df
    
    def generate_report(self, output_dir: Path):
        """Generate a comprehensive report of all experiments."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("Generating comprehensive report...")
        
        if not self.experiments:
            print("Error: No valid experiments found to analyze")
            return
        
        # 1. Training curves for all experiments
        print("Plotting training curves...")
        try:
            self.plot_training_curves(
                metrics=['loss', 'recon_loss', 'kl_loss'],
                save_path=output_dir / 'training_curves.png'
            )
        except Exception as e:
            print(f"Warning: Failed to plot training curves: {e}")
        
        # 2. Learning rate schedules
        print("Plotting learning rates...")
        try:
            self.plot_learning_rates(save_path=output_dir / 'learning_rates.png')
        except Exception as e:
            print(f"Warning: Failed to plot learning rates: {e}")
        
        # 3. Final metrics comparison
        print("Comparing final metrics...")
        try:
            metrics_df = self.compare_final_metrics(save_path=output_dir / 'metrics_comparison')
        except Exception as e:
            print(f"Warning: Failed to compare metrics: {e}")
            metrics_df = pd.DataFrame()
        
        # 4. GAN metrics if applicable
        print("Plotting GAN metrics...")
        try:
            self.plot_gan_metrics(save_path=output_dir / 'gan_metrics.png')
        except Exception as e:
            print(f"Warning: Failed to plot GAN metrics: {e}")
        
        # 5. Early stopping analysis
        print("Analyzing early stopping...")
        try:
            early_stop_df = self.analyze_early_stopping()
            early_stop_df.to_csv(output_dir / 'early_stopping_analysis.csv', index=False)
        except Exception as e:
            print(f"Warning: Failed to analyze early stopping: {e}")
            early_stop_df = pd.DataFrame()
        
        # 6. Generate summary markdown report
        if not metrics_df.empty:
            self._generate_markdown_report(output_dir, metrics_df, early_stop_df)
        else:
            print("Warning: Unable to generate markdown report due to insufficient data")
        
        print(f"\nReport generated in: {output_dir}")
    
    def _generate_markdown_report(self, output_dir: Path, metrics_df: pd.DataFrame, 
                                early_stop_df: pd.DataFrame):
        """Generate a markdown summary report."""
        report_path = output_dir / 'report.md'
        
        with open(report_path, 'w') as f:
            f.write("# VAE/VAE-GAN Training Report\n\n")
            f.write(f"Generated: {pd.Timestamp.now()}\n\n")
            
            # Best models
            f.write("## Best Models\n\n")
            best_models = metrics_df.nsmallest(5, 'Best Val Loss')
            f.write(best_models[['Experiment', 'Architecture', 'Model Type', 
                               'Best Val Loss', 'Epochs']].to_markdown(index=False))
            f.write("\n\n")
            
            # Architecture comparison
            f.write("## Architecture Comparison\n\n")
            arch_summary = metrics_df.groupby('Architecture').agg({
                'Best Val Loss': ['mean', 'std', 'min'],
                'Epochs': 'mean'
            }).round(4)
            f.write(arch_summary.to_markdown())
            f.write("\n\n")
            
            # Early stopping summary
            f.write("## Early Stopping Summary\n\n")
            f.write(f"- Experiments with early stopping: {early_stop_df['Early Stopped'].sum()}/{len(early_stop_df)}\n")
            f.write(f"- Average best epoch: {early_stop_df['Best Epoch'].mean():.1f}\n")
            f.write(f"- Average epochs after best: {early_stop_df['Epochs After Best'].mean():.1f}\n")
            f.write(f"- Average overfitting: {early_stop_df['Overfitting'].mean():.4f}\n\n")
            
            # Training efficiency
            f.write("## Training Efficiency\n\n")
            f.write("Models that converged quickly (best epoch < 20):\n\n")
            quick_converge = early_stop_df[early_stop_df['Best Epoch'] < 20]
            if not quick_converge.empty:
                f.write(quick_converge[['Experiment', 'Best Epoch', 'Best Val Loss']].to_markdown(index=False))
            else:
                f.write("No models converged within 20 epochs.\n")
            f.write("\n\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            best_model = metrics_df.iloc[0]
            f.write(f"1. Best performing model: **{best_model['Experiment']}**\n")
            f.write(f"   - Architecture: {best_model['Architecture']}\n")
            f.write(f"   - Best Val Loss: {best_model['Best Val Loss']:.4f}\n")
            f.write(f"   - Hidden/Latent Dims: {best_model['Hidden Dim']}/{best_model['Latent Dim']}\n\n")
            
            # Beta analysis
            beta_models = metrics_df[metrics_df['Experiment'].str.contains('beta')]
            if not beta_models.empty:
                best_beta_idx = beta_models['Best Val Loss'].idxmin()
                best_beta = beta_models.loc[best_beta_idx, 'Beta']
                f.write(f"2. Optimal beta value: **{best_beta}**\n\n")
            
            # Latent dim analysis
            latent_models = metrics_df[metrics_df['Experiment'].str.contains('latent')]
            if not latent_models.empty:
                best_latent_idx = latent_models['Best Val Loss'].idxmin()
                best_latent = latent_models.loc[best_latent_idx, 'Latent Dim']
                f.write(f"3. Optimal latent dimension: **{best_latent}**\n\n")
            
            # GAN vs VAE comparison
            vae_only = metrics_df[metrics_df['Model Type'] == 'vae']
            vae_gan = metrics_df[metrics_df['Model Type'] == 'vae_gan']
            if not vae_only.empty and not vae_gan.empty:
                vae_mean = vae_only['Best Val Loss'].mean()
                gan_mean = vae_gan['Best Val Loss'].mean()
                f.write(f"4. Model type comparison:\n")
                f.write(f"   - VAE average loss: {vae_mean:.4f}\n")
                f.write(f"   - VAE-GAN average loss: {gan_mean:.4f}\n")
                f.write(f"   - Recommendation: {'VAE-GAN' if gan_mean < vae_mean else 'VAE'}\n\n")
        
        print(f"Markdown report saved to: {report_path}")


def plot_sample_trajectories(checkpoint_path: Path, n_samples: int = 5,
                           transport_mode: int = 0, trip_length: int = 1000,
                           save_path: Optional[Path] = None):
    """Generate and plot sample trajectories from a trained model."""
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from ml_mobility_ns3.models.vae import ConditionalTrajectoryVAE
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint['config']
    
    # Create model
    model = ConditionalTrajectoryVAE(**config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Generate samples
    transport_modes = torch.tensor([transport_mode] * n_samples)
    trip_lengths = torch.tensor([trip_length] * n_samples)
    
    with torch.no_grad():
        trajectories = model.generate(transport_modes, trip_lengths, device='cpu')
    
    # Plot trajectories
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Latitude vs Longitude
    ax = axes[0, 0]
    for i in range(n_samples):
        traj = trajectories[i].numpy()
        ax.plot(traj[:trip_length, 1], traj[:trip_length, 0], 
               alpha=0.7, linewidth=1.5)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'Generated Trajectories (n={n_samples})')
    ax.grid(True, alpha=0.3)
    
    # Speed over time
    ax = axes[0, 1]
    for i in range(n_samples):
        traj = trajectories[i].numpy()
        time_steps = np.arange(trip_length)
        ax.plot(time_steps, traj[:trip_length, 2], alpha=0.7)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Speed')
    ax.set_title('Speed Profiles')
    ax.grid(True, alpha=0.3)
    
    # Latitude distribution
    ax = axes[1, 0]
    all_lats = trajectories[:, :trip_length, 0].numpy().flatten()
    ax.hist(all_lats, bins=50, alpha=0.7, density=True)
    ax.set_xlabel('Latitude')
    ax.set_ylabel('Density')
    ax.set_title('Latitude Distribution')
    
    # Speed distribution
    ax = axes[1, 1]
    all_speeds = trajectories[:, :trip_length, 2].numpy().flatten()
    ax.hist(all_speeds, bins=50, alpha=0.7, density=True)
    ax.set_xlabel('Speed')
    ax.set_ylabel('Density')
    ax.set_title('Speed Distribution')
    
    plt.suptitle(f'Model: {checkpoint_path.stem}, Mode: {transport_mode}, Length: {trip_length}')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved sample trajectories to: {save_path}")
    else:
        plt.savefig('sample_trajectories.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("Saved sample trajectories to: sample_trajectories.png")


def compare_models_generation(checkpoint_paths: List[Path], n_samples: int = 10,
                            transport_mode: int = 0, save_path: Optional[Path] = None):
    """Compare generation quality across multiple models."""
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from ml_mobility_ns3.models.vae import ConditionalTrajectoryVAE
    
    fig, axes = plt.subplots(len(checkpoint_paths), 3, figsize=(15, 5*len(checkpoint_paths)))
    if len(checkpoint_paths) == 1:
        axes = axes.reshape(1, -1)
    
    for idx, checkpoint_path in enumerate(checkpoint_paths):
        # Load model
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        config = checkpoint['config']
        model = ConditionalTrajectoryVAE(**config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Generate samples
        transport_modes = torch.tensor([transport_mode] * n_samples)
        trip_lengths = torch.tensor([1000] * n_samples)
        
        with torch.no_grad():
            trajectories = model.generate(transport_modes, trip_lengths, device='cpu')
        
        # Plot for this model
        # Trajectories
        ax = axes[idx, 0]
        for i in range(min(n_samples, 5)):  # Plot max 5 trajectories
            traj = trajectories[i].numpy()
            ax.plot(traj[:1000, 1], traj[:1000, 0], alpha=0.6, linewidth=1)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(f'{checkpoint_path.stem}: Trajectories')
        ax.grid(True, alpha=0.3)
        
        # Speed profiles
        ax = axes[idx, 1]
        for i in range(min(n_samples, 5)):
            traj = trajectories[i].numpy()
            ax.plot(np.arange(1000), traj[:1000, 2], alpha=0.6)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Speed')
        ax.set_title('Speed Profiles')
        ax.grid(True, alpha=0.3)
        
        # Statistics
        ax = axes[idx, 2]
        all_speeds = trajectories[:, :1000, 2].numpy().flatten()
        ax.hist(all_speeds, bins=30, alpha=0.7, density=True)
        ax.axvline(all_speeds.mean(), color='red', linestyle='--', 
                  label=f'Mean: {all_speeds.mean():.2f}')
        ax.axvline(np.median(all_speeds), color='green', linestyle='--', 
                  label=f'Median: {np.median(all_speeds):.2f}')
        ax.set_xlabel('Speed')
        ax.set_ylabel('Density')
        ax.set_title('Speed Distribution')
        ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved model comparison to: {save_path}")
    else:
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("Saved model comparison to: model_comparison.png")


def fix_corrupted_json_files(results_dir: Path):
    """Attempt to fix corrupted JSON files in the results directory."""
    fixed_count = 0
    
    for exp_dir in results_dir.iterdir():
        if exp_dir.is_dir():
            # Check experiment_info.json
            info_file = exp_dir / "experiment_info.json"
            if info_file.exists():
                try:
                    with open(info_file) as f:
                        json.load(f)
                except json.JSONDecodeError:
                    print(f"Fixing corrupted file: {info_file}")
                    try:
                        # Read the file content
                        with open(info_file, 'r') as f:
                            content = f.read()
                        
                        # Try to extract valid JSON by finding the last complete brace
                        if content.strip() and not content.strip().endswith('}'):
                            # Find the last complete JSON object
                            last_brace = content.rfind('}')
                            if last_brace > 0:
                                fixed_content = content[:last_brace + 1]
                                # Validate it's proper JSON
                                json.loads(fixed_content)
                                # Write back
                                with open(info_file, 'w') as f:
                                    f.write(fixed_content)
                                fixed_count += 1
                                print(f"  Fixed: {info_file}")
                    except Exception as e:
                        print(f"  Could not fix {info_file}: {e}")
    
    print(f"\nFixed {fixed_count} corrupted JSON files")


def main():
    parser = argparse.ArgumentParser(description="Analyze VAE/VAE-GAN training results")
    parser.add_argument("--results-dir", type=Path, required=True,
                       help="Directory containing experiment results")
    parser.add_argument("--output-dir", type=Path, default=Path("analysis_output"),
                       help="Directory to save analysis outputs")
    parser.add_argument("--experiments", nargs="+", default=None,
                       help="Specific experiments to analyze")
    parser.add_argument("--plot-curves", action="store_true",
                       help="Plot training curves")
    parser.add_argument("--plot-lr", action="store_true",
                       help="Plot learning rate schedules")
    parser.add_argument("--plot-gan", action="store_true",
                       help="Plot GAN-specific metrics")
    parser.add_argument("--compare-metrics", action="store_true",
                       help="Compare final metrics across experiments")
    parser.add_argument("--early-stopping", action="store_true",
                       help="Analyze early stopping behavior")
    parser.add_argument("--generate-report", action="store_true",
                       help="Generate comprehensive report")
    parser.add_argument("--sample-trajectories", type=Path, default=None,
                       help="Generate sample trajectories from checkpoint")
    parser.add_argument("--compare-generation", nargs="+", type=Path, default=None,
                       help="Compare generation from multiple checkpoints")
    
    parser.add_argument("--fix-json", action="store_true",
                       help="Attempt to fix corrupted JSON files before analysis")
    
    args = parser.parse_args()
    
    # Fix corrupted JSON files if requested
    if args.fix_json:
        print("Attempting to fix corrupted JSON files...")
        fix_corrupted_json_files(args.results_dir)
        print("")
    
    # Create analyzer
    analyzer = TrainingAnalyzer(args.results_dir)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run requested analyses
    if args.plot_curves:
        print("Plotting training curves...")
        analyzer.plot_training_curves(
            experiment_names=args.experiments,
            save_path=args.output_dir / "training_curves.png"
        )
    
    if args.plot_lr:
        print("Plotting learning rates...")
        analyzer.plot_learning_rates(
            experiment_names=args.experiments,
            save_path=args.output_dir / "learning_rates.png"
        )
    
    if args.plot_gan:
        print("Plotting GAN metrics...")
        analyzer.plot_gan_metrics(
            experiment_names=args.experiments,
            save_path=args.output_dir / "gan_metrics.png"
        )
    
    if args.compare_metrics:
        print("Comparing metrics...")
        analyzer.compare_final_metrics(
            save_path=args.output_dir / "metrics_comparison"
        )
    
    if args.early_stopping:
        print("Analyzing early stopping...")
        df = analyzer.analyze_early_stopping()
        df.to_csv(args.output_dir / "early_stopping_analysis.csv", index=False)
    
    if args.generate_report:
        print("Generating comprehensive report...")
        analyzer.generate_report(args.output_dir)
    
    if args.sample_trajectories:
        print(f"Generating sample trajectories from {args.sample_trajectories}...")
        plot_sample_trajectories(
            args.sample_trajectories,
            save_path=args.output_dir / "sample_trajectories.png"
        )
    
    if args.compare_generation:
        print("Comparing generation across models...")
        compare_models_generation(
            args.compare_generation,
            save_path=args.output_dir / "generation_comparison.png"
        )
    
    print(f"\nAnalysis complete! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()