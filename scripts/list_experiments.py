# scripts/list_experiments.py
#!/usr/bin/env python
import json
from pathlib import Path
from tabulate import tabulate
import sys

sys.path.append(str(Path(__file__).parent.parent))


def list_experiments():
    """List all experiments."""
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
            
            table_data.append([
                exp["id"],
                exp["model_type"],
                exp["status"],
                f"{final_metrics.get('best_val_loss', 'N/A'):.4f}" if isinstance(final_metrics.get('best_val_loss'), (int, float)) else "N/A",
                final_metrics.get("epochs_trained", "N/A"),
                f"{params.get('total', 0):,}" if params.get('total') else "N/A"
            ])
        else:
            table_data.append([
                exp["id"],
                exp["model_type"],
                exp["status"],
                "N/A",
                "N/A",
                "N/A"
            ])
    
    # Print table
    headers = ["Experiment ID", "Model", "Status", "Best Loss", "Epochs", "Parameters"]
    print("\n=== Experiments ===")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    print(f"\nTotal experiments: {len(manifest['experiments'])}")


if __name__ == "__main__":
    list_experiments()