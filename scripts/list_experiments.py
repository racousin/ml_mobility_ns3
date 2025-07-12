#!/usr/bin/env python
import json
from pathlib import Path
from tabulate import tabulate
import sys
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))


def load_tensorboard_metrics(exp_dir: Path) -> dict:
    """Load metrics from tensorboard logs if available."""
    tb_dir = exp_dir / "logs" / "tensorboard"
    metrics = {}
    
    try:
        # Try to load from event files using tensorboard
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        
        version_dirs = [d for d in tb_dir.iterdir() if d.is_dir()] if tb_dir.exists() else []
        if not version_dirs:
            # Check direct event files
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
                    'val_bird_distance_mae', 'val_frechet_distance'
                ]
                
                for metric in metric_names:
                    if metric in ea.scalars.Keys():
                        values = ea.scalars.Items(metric)
                        if values:
                            # Get the last value
                            metrics[metric] = values[-1].value
                
                break  # Use first valid version
    except Exception as e:
        pass  # Silently fail if tensorboard not available
    
    return metrics


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
    for exp in manifest["experiments"]:
        exp_dir = Path("experiments") / exp["id"]
        
        # Load model info for more details
        model_info_path = exp_dir / "model_info.json"
        if model_info_path.exists():
            with open(model_info_path, "r") as f:
                model_info = json.load(f)
            
            final_metrics = model_info.get("final_metrics", {})
            params = model_info.get("parameters", {})
            
            # Try to load evaluation results
            eval_results_path = exp_dir / "evaluation_results.json"
            eval_metrics = {}
            if eval_results_path.exists():
                with open(eval_results_path, "r") as f:
                    eval_results = json.load(f)
                    eval_metrics = eval_results.get("reconstruction", {})
            
            # Try to load tensorboard metrics
            tb_metrics = load_tensorboard_metrics(exp_dir)
            
            # Get loss type from config
            config_path = exp_dir / "config.yaml"
            loss_type = "simple_vae"
            if config_path.exists():
                try:
                    import yaml
                    with open(config_path, "r") as f:
                        config = yaml.safe_load(f)
                        loss_config = config.get("training", {}).get("loss", {})
                        loss_type = loss_config.get("type", "simple_vae")
                except:
                    pass
            
            row = [
                exp["id"][:20] + "..." if len(exp["id"]) > 20 else exp["id"],
                exp["model_type"],
                loss_type,
                exp["status"],
                f"{final_metrics.get('best_val_loss', 'N/A'):.4f}" if isinstance(final_metrics.get('best_val_loss'), (int, float)) else "N/A",
                final_metrics.get("epochs_trained", "N/A"),
                f"{params.get('total', 0):,}" if params.get('total') else "N/A"
            ]
            
            # Add standardized metrics if available
            if detailed:
                # Prefer evaluation results, then tensorboard metrics
                mse = eval_metrics.get('mean_speed_mae', tb_metrics.get('val_mse', 'N/A'))
                frechet = eval_metrics.get('mean_speed_mae', tb_metrics.get('val_frechet_distance', 'N/A'))
                
                row.extend([
                    f"{mse:.4f}" if isinstance(mse, (int, float)) else "N/A",
                    f"{frechet:.4f}" if isinstance(frechet, (int, float)) else "N/A"
                ])
            
            table_data.append(row)
        else:
            table_data.append([
                exp["id"],
                exp["model_type"],
                "unknown",
                exp["status"],
                "N/A",
                "N/A",
                "N/A"
            ] + (["N/A", "N/A"] if detailed else []))
    
    # Print table
    headers = ["Experiment ID", "Model", "Loss Type", "Status", "Best Loss", "Epochs", "Parameters"]
    if detailed:
        headers.extend(["MSE", "Fréchet"])
    
    print("\n=== Experiments ===")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    print(f"\nTotal experiments: {len(manifest['experiments'])}")
    
    # Show comparison of best models by metric
    if detailed and any('completed' in exp['status'] for exp in manifest['experiments']):
        print("\n=== Best Models by Metric ===")
        
        # Collect metrics for completed experiments
        completed_exps = []
        for i, exp in enumerate(manifest['experiments']):
            if exp['status'] == 'completed' and i < len(table_data):
                row = table_data[i]
                if row[7] != "N/A":  # Has MSE
                    try:
                        completed_exps.append({
                            'id': row[0],
                            'model': row[1],
                            'loss_type': row[2],
                            'mse': float(row[7]) if row[7] != "N/A" else float('inf'),
                            'frechet': float(row[8]) if row[8] != "N/A" else float('inf')
                        })
                    except:
                        pass
        
        if completed_exps:
            # Sort by different metrics
            by_mse = sorted(completed_exps, key=lambda x: x['mse'])[:3]
            by_frechet = sorted(completed_exps, key=lambda x: x['frechet'])[:3]
            
            print("\nBest by MSE:")
            for exp in by_mse:
                print(f"  {exp['id']} ({exp['model']}/{exp['loss_type']}): {exp['mse']:.4f}")
            
            print("\nBest by Fréchet Distance:")
            for exp in by_frechet:
                print(f"  {exp['id']} ({exp['model']}/{exp['loss_type']}): {exp['frechet']:.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--detailed", "-d", action="store_true", help="Show detailed metrics")
    args = parser.parse_args()
    
    list_experiments(detailed=args.detailed)