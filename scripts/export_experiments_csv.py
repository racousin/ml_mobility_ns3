#!/usr/bin/env python
import json
from pathlib import Path
import csv
import sys
import yaml
import math

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

def format_metric_value(value):
    """Format metric value."""
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "N/A"
    if isinstance(value, float):
        return str(value).replace('.', ',')
    return value

def export_experiments_to_csv(output_file="experiments_completed.csv"):
    manifest_path = Path("experiments") / "manifest.json"
    
    if not manifest_path.exists():
        print("No experiments found.")
        return
    
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    
    if not manifest["experiments"]:
        print("No experiments found.")
        return
        
    csv_columns = [
        "Model ID", "Model", "Loss Type", "Status", "Parameters", "Epochs", 
        "Val Loss", "Recon Loss", "KL Loss", "Speed MAE", "Dist MAE", 
        "Total Dist MAE", "Bird Dist MAE", "Config (model)", "Training config",
        "BIKE_generated", "BIKE_real_state", "CAR_generated", "CAR_real_state",
        "MIXED_generated", "MIXED_real_state", "PUBLIC_TRANSPORT_generated", 
        "PUBLIC_TRANSPORT_real_state", "WALKING_generated", "WALKING_real_state"
    ]
    
    # Mapping for real stats (based on sequence numbers matching in some datasets)
    real_stats_mapping = {
        "BIKE": "category_0",
        "CAR": "category_1",
        "MIXED": "category_2",
        "PUBLIC_TRANSPORT": "category_3",
        "WALKING": "category_4"
    }

    with open(output_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(csv_columns)
        
        for exp in manifest["experiments"]:
            if exp["status"] != "completed":
                continue
                
            exp_dir = Path("experiments") / exp["id"]
            
            # Default values
            model_info = {}
            metrics = {}
            loss_type = "unknown"
            eval_results = {}
            
            model_info_path = exp_dir / "model_info.json"
            if model_info_path.exists():
                with open(model_info_path, "r") as mf:
                    model_info = json.load(mf)
                metrics = load_experiment_metrics(exp_dir)
                loss_type = get_loss_type(exp_dir, model_info)
                
            eval_results_path = exp_dir / "evaluation_results.json"
            if eval_results_path.exists():
                with open(eval_results_path, "r") as ef:
                    eval_results = json.load(ef)
                    
            params = model_info.get("parameters", {})
            final_metrics = model_info.get("final_metrics", {})
            
            generated_stats = eval_results.get("generated_stats", {})
            real_stats = eval_results.get("real_stats", {})
            
            def get_real_stat(category):
                if category in real_stats:
                    return json.dumps(real_stats[category])
                fallback_key = real_stats_mapping.get(category)
                if fallback_key and fallback_key in real_stats:
                    return json.dumps(real_stats[fallback_key])
                return "{}"
                
            row = [
                exp["id"],
                exp["model_type"],
                loss_type,
                exp["status"],
                params.get('total', 'N/A'),
                final_metrics.get("epochs_trained", "N/A"),
                format_metric_value(metrics.get('val_loss', final_metrics.get('best_val_loss'))),
                format_metric_value(metrics.get('val_recon_loss')),
                format_metric_value(metrics.get('val_kl_loss')),
                format_metric_value(metrics.get('val_speed_mae')),
                format_metric_value(metrics.get('val_distance_mae')),
                format_metric_value(metrics.get('val_total_distance_mae')),
                format_metric_value(metrics.get('val_bird_distance_mae')),
                json.dumps(model_info.get("config", {})),
                json.dumps(model_info.get("training_config", {})),
                json.dumps(generated_stats.get("BIKE", {})),
                get_real_stat("BIKE"),
                json.dumps(generated_stats.get("CAR", {})),
                get_real_stat("CAR"),
                json.dumps(generated_stats.get("MIXED", {})),
                get_real_stat("MIXED"),
                json.dumps(generated_stats.get("PUBLIC_TRANSPORT", {})),
                get_real_stat("PUBLIC_TRANSPORT"),
                json.dumps(generated_stats.get("WALKING", {})),
                get_real_stat("WALKING")
            ]
            
            writer.writerow(row)
            
    print(f"Exported completed experiments to {output_file}")

if __name__ == "__main__":
    output_path = "completed_experiments_stats.csv"
    if len(sys.argv) > 1:
        output_path = sys.argv[1]
    export_experiments_to_csv(output_path)
