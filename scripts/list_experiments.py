#!/usr/bin/env python
import json
from pathlib import Path
from tabulate import tabulate
import sys
import pandas as pd
import math
import yaml

sys.path.append(str(Path(__file__).parent.parent))


def load_experiment_metrics(exp_dir: Path) -> dict:
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


def format_metric_value(value, decimals=4):
    """Format metric value for display."""
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "N/A"
    if isinstance(value, (int, float)):
        # Use different decimal places for different metric ranges
        if abs(value) < 0.001:
            return f"{value:.6f}"
        elif abs(value) < 1:
            return f"{value:.4f}"
        elif abs(value) < 100:
            return f"{value:.3f}"
        else:
            return f"{value:.1f}"
    return str(value)


def get_loss_type(exp_dir: Path, model_info: dict) -> str:
    """Get the loss type from model info or config."""
    # Check training_config first
    if 'training_config' in model_info and 'loss' in model_info['training_config']:
        loss_config = model_info['training_config']['loss']
        if isinstance(loss_config, dict):
            return loss_config.get('type', 'simple_vae')
    
    # Fallback to config file
    config_path = exp_dir / "config.yaml"
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
                loss_config = config.get("training", {}).get("loss", {})
                return loss_config.get("type", "simple_vae")
        except:
            pass
    
    return "simple_vae"


def list_experiments():
    """List all experiments with detailed metrics."""
    manifest_path = Path("experiments") / "manifest.json"
    
    if not manifest_path.exists():
        print("No experiments found. Train a model first with: python scripts/train.py")
        return
    
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    
    if not manifest["experiments"]:
        print("No experiments found.")
        return
    
    # Prepare data for table
    table_data = []
    all_metrics_data = []  # Store for finding best models
    
    for exp in manifest["experiments"]:
        exp_dir = Path("experiments") / exp["id"]
        
        # Load model info for basic details
        model_info_path = exp_dir / "model_info.json"
        if model_info_path.exists():
            with open(model_info_path, "r") as f:
                model_info = json.load(f)
            
            # Get all metrics
            metrics = load_experiment_metrics(exp_dir)
            
            # Get loss type
            loss_type = get_loss_type(exp_dir, model_info)
            
            params = model_info.get("parameters", {})
            final_metrics = model_info.get("final_metrics", {})
            
            # Build row with all metrics
            row = [
                exp["id"][:20] + "..." if len(exp["id"]) > 20 else exp["id"],
                exp["model_type"],
                loss_type,
                exp["status"],
                f"{params.get('total', 0):,}" if params.get('total') else "N/A",
                final_metrics.get("epochs_trained", "N/A"),
                # Loss components
                format_metric_value(metrics.get('val_loss', final_metrics.get('best_val_loss'))),
                format_metric_value(metrics.get('val_recon_loss')),
                format_metric_value(metrics.get('val_kl_loss')),
                # Evaluation metrics (in interpretable units)
                format_metric_value(metrics.get('val_speed_mae')),
                format_metric_value(metrics.get('val_distance_mae')),
                format_metric_value(metrics.get('val_total_distance_mae')),
                format_metric_value(metrics.get('val_bird_distance_mae'))
            ]
            
            table_data.append(row)
            
            # Store for best model analysis
            if exp["status"] == "completed" and metrics:
                all_metrics_data.append({
                    'id': exp["id"],
                    'model': exp["model_type"],
                    'loss_type': loss_type,
                    'metrics': metrics
                })
        else:
            # Fallback for experiments without model_info.json
            row = [exp["id"], exp["model_type"], "unknown", exp["status"]]
            row.extend(["N/A"] * 9)  # 9 additional columns
            table_data.append(row)
    
    # Print main table
    headers = [
        "Experiment ID", "Model", "Loss Type", "Status", "Parameters", "Epochs",
        "Val Loss", "Recon Loss", "KL Loss",
        "Speed MAE", "Dist MAE", "Total Dist MAE", "Bird Dist MAE"
    ]
    
    print("\n" + "="*120)
    print("EXPERIMENT RESULTS")
    print("="*120)
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    print(f"\nTotal experiments: {len(manifest['experiments'])}")
    
    # Show best models by different metrics
    if all_metrics_data:
        print("\n" + "="*80)
        print("BEST MODELS BY METRIC")
        print("="*80)
        
        # Define metrics to track (with proper names and units)
        metrics_to_minimize = {
            'val_loss': ('Overall Loss', ''),
            'val_recon_loss': ('Reconstruction Loss', ''),
            'val_kl_loss': ('KL Divergence', ''),
            'val_speed_mae': ('Speed Error', 'km/h'),
            'val_distance_mae': ('Position Error', 'km'),
            'val_total_distance_mae': ('Total Distance Error', 'km'),
            'val_bird_distance_mae': ('Bird Distance Error', 'km')
        }
        
        for metric_key, (metric_name, unit) in metrics_to_minimize.items():
            # Find experiments with this metric
            valid_exps = []
            for exp_data in all_metrics_data:
                if metric_key in exp_data['metrics']:
                    value = exp_data['metrics'][metric_key]
                    if isinstance(value, (int, float)) and not math.isnan(value) and not math.isinf(value):
                        valid_exps.append((exp_data, value))
            
            if valid_exps:
                # Sort by metric value (ascending - lower is better)
                valid_exps.sort(key=lambda x: x[1])
                
                unit_str = f" {unit}" if unit else ""
                print(f"\n📈 Best by {metric_name}:")
                for i, (exp_data, value) in enumerate(valid_exps[:3], 1):
                    exp_id = exp_data['id']
                    if len(exp_id) > 35:
                        exp_id = exp_id[:35] + "..."
                    
                    model_info = f"{exp_data['model']}/{exp_data['loss_type']}"
                    print(f"  {i}. {exp_id}")
                    print(f"     {model_info}: {format_metric_value(value)}{unit_str}")
        
        # Summary of completed experiments
        completed_exps = [exp for exp in all_metrics_data]
        if completed_exps:
            print(f"\n📊 Summary:")
            print(f"   • Completed experiments: {len(completed_exps)}")
            
            # Group by model type
            model_types = {}
            for exp in completed_exps:
                model_type = exp['model']
                if model_type not in model_types:
                    model_types[model_type] = 0
                model_types[model_type] += 1
            
            print(f"   • Model types: {', '.join([f'{k}({v})' for k, v in model_types.items()])}")
            
            # Group by loss type
            loss_types = {}
            for exp in completed_exps:
                loss_type = exp['loss_type']
                if loss_type not in loss_types:
                    loss_types[loss_type] = 0
                loss_types[loss_type] += 1
            
            print(f"   • Loss functions: {', '.join([f'{k}({v})' for k, v in loss_types.items()])}")


if __name__ == "__main__":
    list_experiments()