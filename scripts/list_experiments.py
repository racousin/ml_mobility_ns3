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
        return f"{value:.{decimals}f}"
    return str(value)


def list_experiments(detailed=False):
    """List all experiments with metrics."""
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
            loss_type = "simple_vae"
            if 'training_config' in model_info and 'loss' in model_info['training_config']:
                loss_config = model_info['training_config']['loss']
                if isinstance(loss_config, dict):
                    loss_type = loss_config.get('type', 'simple_vae')
            else:
                # Fallback to config file
                config_path = exp_dir / "config.yaml"
                if config_path.exists():
                    try:
                        
                        with open(config_path, "r") as f:
                            config = yaml.safe_load(f)
                            loss_config = config.get("training", {}).get("loss", {})
                            loss_type = loss_config.get("type", "simple_vae")
                    except:
                        pass
            
            params = model_info.get("parameters", {})
            final_metrics = model_info.get("final_metrics", {})
            
            # Basic info
            row = [
                exp["id"][:20] + "..." if len(exp["id"]) > 20 else exp["id"],
                exp["model_type"],
                loss_type,
                exp["status"],
                format_metric_value(metrics.get('val_loss', final_metrics.get('best_val_loss'))),
                final_metrics.get("epochs_trained", "N/A"),
                f"{params.get('total', 0):,}" if params.get('total') else "N/A"
            ]
            
            if detailed:
                # Add detailed metrics
                row.extend([
                    format_metric_value(metrics.get('val_mse')),
                    format_metric_value(metrics.get('val_kl_loss')),
                    format_metric_value(metrics.get('val_total_distance_mae')),
                    format_metric_value(metrics.get('val_bird_distance_mae')),
                    format_metric_value(metrics.get('val_speed_mse'))
                ])
            
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
            row.extend(["N/A"] * (3 + (6 if detailed else 0)))
            table_data.append(row)
    
    # Print table
    headers = ["Experiment ID", "Model", "Loss Type", "Status", "Best Loss", "Epochs", "Parameters"]
    if detailed:
        headers.extend(["MSE", "KL Loss", "Total Dist MAE", "Bird Dist MAE", "Speed MSE"])
    
    print("\n=== Experiments ===")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    print(f"\nTotal experiments: {len(manifest['experiments'])}")
    
    # Show best models by different metrics
    if detailed and all_metrics_data:
        print("\n=== Best Models by Metric ===")
        
        metrics_to_minimize = {
            'val_loss': 'Validation Loss',
            'val_mse': 'MSE',
            'val_kl_loss': 'KL Loss',
            'val_total_distance_mae': 'Total Distance MAE',
            'val_bird_distance_mae': 'Bird Distance MAE',
            'val_speed_mse': 'Speed MSE'
        }
        
        for metric_key, metric_name in metrics_to_minimize.items():
            # Find experiments with this metric
            valid_exps = []
            for exp_data in all_metrics_data:
                if metric_key in exp_data['metrics']:
                    value = exp_data['metrics'][metric_key]
                    if isinstance(value, (int, float)) and not math.isnan(value) and not math.isinf(value):
                        valid_exps.append((exp_data, value))
            
            if valid_exps:
                # Sort by metric value
                valid_exps.sort(key=lambda x: x[1])
                
                print(f"\nBest by {metric_name}:")
                for exp_data, value in valid_exps[:3]:
                    exp_id = exp_data['id']
                    if len(exp_id) > 30:
                        exp_id = exp_id[:30] + "..."
                    print(f"  {exp_id} ({exp_data['model']}/{exp_data['loss_type']}): {format_metric_value(value)}")



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--detailed", "-d", action="store_true", help="Show detailed metrics")
    args = parser.parse_args()
    
    list_experiments(detailed=args.detailed)