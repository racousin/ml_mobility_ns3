#!/bin/bash
# run_experiments.sh
# Script to run multiple LSTM VAE experiments with different configurations

# Default values
GPU_ID=${1:-0}
DATA_PATH=${2:-"./preprocessing/vae_dataset.npz"}
BASE_RESULTS_DIR=${3:-"results/experiments"}
DRY_RUN=${4:-false}

# Create results directory
mkdir -p ${BASE_RESULTS_DIR}

echo "Starting LSTM VAE experiments on GPU $GPU_ID"
echo "Data path: $DATA_PATH"
echo "Base results directory: $BASE_RESULTS_DIR"
echo "Dry run: $DRY_RUN"
echo ""

# Function to run a single experiment
run_experiment() {
    local exp_name=$1
    local hidden_dim=$2
    local latent_dim=$3
    local num_layers=$4
    local batch_size=$5
    local beta=$6
    local learning_rate=$7
    local dropout=$8
    local additional_args=$9
    
    local results_dir="${BASE_RESULTS_DIR}/${exp_name}"
    
    echo "=========================================="
    echo "Running experiment: $exp_name"
    echo "Hidden dim: $hidden_dim, Latent dim: $latent_dim"
    echo "Layers: $num_layers, Batch size: $batch_size"
    echo "Beta: $beta, LR: $learning_rate, Dropout: $dropout"
    echo "Results directory: $results_dir"
    echo "=========================================="
    
    local cmd="python ./train.py \
        --data-path $DATA_PATH \
        --experiment-name $exp_name \
        --epochs 100 \
        --batch-size $batch_size \
        --lr $learning_rate \
        --beta $beta \
        --hidden-dim $hidden_dim \
        --latent-dim $latent_dim \
        --num-layers $num_layers \
        --dropout $dropout \
        --device cuda \
        --gpu-id $GPU_ID \
        --lr-patience 15 \
        --lr-factor 0.5 \
        --early-stopping-patience 25 \
        --gradient-clip 1.0"
    
    # Add additional arguments
    if [ ! -z "$additional_args" ]; then
        cmd="$cmd $additional_args"
    fi
    
    echo "Command: $cmd"
    
    if [ "$DRY_RUN" = false ]; then
        # Create experiment info file
        mkdir -p $results_dir
        echo "{
    \"experiment_name\": \"$exp_name\",
    \"command\": \"$cmd\",
    \"start_time\": \"$(date)\",
    \"gpu_id\": $GPU_ID,
    \"config\": {
        \"hidden_dim\": $hidden_dim,
        \"latent_dim\": $latent_dim,
        \"num_layers\": $num_layers,
        \"batch_size\": $batch_size,
        \"beta\": $beta,
        \"learning_rate\": $learning_rate,
        \"dropout\": $dropout
    }
}" > "$results_dir/experiment_info.json"
        
        # Run the experiment
        $cmd
        
        # Update experiment info with end time
        echo "{
    \"experiment_name\": \"$exp_name\",
    \"command\": \"$cmd\",
    \"start_time\": \"$(date)\",
    \"end_time\": \"$(date)\",
    \"gpu_id\": $GPU_ID,
    \"status\": \"completed\",
    \"config\": {
        \"hidden_dim\": $hidden_dim,
        \"latent_dim\": $latent_dim,
        \"num_layers\": $num_layers,
        \"batch_size\": $batch_size,
        \"beta\": $beta,
        \"learning_rate\": $learning_rate,
        \"dropout\": $dropout
    }
}" > "$results_dir/experiment_info.json"
    else
        echo "DRY RUN - Skipping execution"
    fi
    
    echo -e "\n\n"
}

# ======================
# BASELINE EXPERIMENTS
# ======================

echo "=== BASELINE EXPERIMENTS ==="

# Small baseline model
run_experiment \
    "baseline_small" \
    128 \
    16 \
    2 \
    64 \
    0.001 \
    1e-4 \
    0.1 \
    ""

# Medium baseline model
run_experiment \
    "baseline_medium" \
    256 \
    32 \
    2 \
    32 \
    0.005 \
    1e-4 \
    0.1 \
    ""

# Large baseline model
run_experiment \
    "baseline_large" \
    512 \
    64 \
    3 \
    16 \
    0.01 \
    1e-4 \
    0.2 \
    ""

# ======================
# LATENT DIMENSION STUDY
# ======================

echo "=== LATENT DIMENSION STUDY ==="

for latent_dim in 8 16 32 64 128; do
    run_experiment \
        "latent_study_${latent_dim}" \
        256 \
        $latent_dim \
        2 \
        32 \
        0.005 \
        1e-4 \
        0.1 \
        ""
done

# ======================
# BETA ABLATION STUDY
# ======================

echo "=== BETA ABLATION STUDY ==="

for beta in 0.0001 0.001 0.005 0.01 0.05 0.1; do
    beta_exp_name=$(echo $beta | sed 's/\./_/g')  # Replace . with _ for filename
    run_experiment \
        "beta_study_${beta_exp_name}" \
        256 \
        32 \
        2 \
        32 \
        $beta \
        1e-4 \
        0.1 \
        ""
done

# ======================
# ARCHITECTURE STUDY
# ======================

echo "=== ARCHITECTURE STUDY ==="

# Different number of layers
for num_layers in 1 2 3 4; do
    run_experiment \
        "layers_study_${num_layers}" \
        256 \
        32 \
        $num_layers \
        32 \
        0.005 \
        1e-4 \
        0.1 \
        ""
done

# Different hidden dimensions
for hidden_dim in 64 128 256 512; do
    run_experiment \
        "hidden_study_${hidden_dim}" \
        $hidden_dim \
        32 \
        2 \
        32 \
        0.005 \
        1e-4 \
        0.1 \
        ""
done

# ======================
# REGULARIZATION STUDY
# ======================

echo "=== REGULARIZATION STUDY ==="

# Different dropout rates
for dropout in 0.0 0.1 0.2 0.3 0.5; do
    dropout_exp_name=$(echo $dropout | sed 's/\./_/g')  # Replace . with _ for filename
    run_experiment \
        "dropout_study_${dropout_exp_name}" \
        256 \
        32 \
        2 \
        32 \
        0.005 \
        1e-4 \
        $dropout \
        ""
done

# ======================
# LEARNING RATE STUDY
# ======================

echo "=== LEARNING RATE STUDY ==="

for lr in 5e-5 1e-4 2e-4 5e-4 1e-3; do
    lr_exp_name=$(echo $lr | sed 's/e-/_e_/g' | sed 's/\./_/g')  # Format for filename
    run_experiment \
        "lr_study_${lr_exp_name}" \
        256 \
        32 \
        2 \
        32 \
        0.005 \
        $lr \
        0.1 \
        ""
done

# ======================
# BATCH SIZE STUDY
# ======================

echo "=== BATCH SIZE STUDY ==="

for batch_size in 16 32 64 128; do
    run_experiment \
        "batch_study_${batch_size}" \
        256 \
        32 \
        2 \
        $batch_size \
        0.005 \
        1e-4 \
        0.1 \
        ""
done

# ======================
# BEST CONFIGURATIONS
# ======================

echo "=== BEST CONFIGURATIONS ==="

# High capacity model with good regularization
run_experiment \
    "best_high_capacity" \
    512 \
    64 \
    3 \
    16 \
    0.01 \
    5e-5 \
    0.2 \
    "--early-stopping-patience 30"

# Balanced model for faster training
run_experiment \
    "best_balanced" \
    256 \
    32 \
    2 \
    32 \
    0.005 \
    1e-4 \
    0.1 \
    "--early-stopping-patience 25"

# Lightweight model for quick experiments
run_experiment \
    "best_lightweight" \
    128 \
    16 \
    2 \
    64 \
    0.001 \
    2e-4 \
    0.1 \
    "--early-stopping-patience 20"

echo "All experiments completed!"
echo "Results saved in: $BASE_RESULTS_DIR"

# Generate summary report
if [ "$DRY_RUN" = false ]; then
    echo "Generating summary report..."
    python - << EOF
import json
import os
import pandas as pd
from pathlib import Path

results_dir = Path("$BASE_RESULTS_DIR")
summary = []

for exp_dir in results_dir.iterdir():
    if exp_dir.is_dir():
        info_file = exp_dir / "experiment_info.json"
        history_file = exp_dir / "history.json"
        config_file = exp_dir / "config.json"
        final_results_file = exp_dir / "final_results.json"
        
        if info_file.exists():
            with open(info_file) as f:
                info = json.load(f)
            
            # Add config information
            if config_file.exists():
                with open(config_file) as f:
                    config = json.load(f)
                    info['model_config'] = config.get('model_config', {})
                    info['dataset_info'] = config.get('dataset_info', {})
            
            # Add training results
            if history_file.exists():
                with open(history_file) as f:
                    history = json.load(f)
                    if history.get('val_loss'):
                        info['final_val_loss'] = history['val_loss'][-1]
                        info['best_val_loss'] = min(history['val_loss'])
                        info['num_epochs'] = len(history['val_loss'])
                        info['final_val_recon_loss'] = history['val_recon_loss'][-1] if history.get('val_recon_loss') else None
                        info['final_val_kl_loss'] = history['val_kl_loss'][-1] if history.get('val_kl_loss') else None
                        info['final_val_speed_mae'] = history['val_speed_mae'][-1] if history.get('val_speed_mae') else None
            
            # Add final evaluation results
            if final_results_file.exists():
                with open(final_results_file) as f:
                    final_results = json.load(f)
                    info['final_evaluation'] = final_results.get('final_validation', {})
                    info['generation_evaluation'] = final_results.get('generation_evaluation', {})
            
            summary.append(info)

# Save summary
with open(results_dir / "experiments_summary.json", 'w') as f:
    json.dump(summary, f, indent=2)

# Create a CSV summary for easy analysis
df_summary = []
for exp in summary:
    row = {
        'experiment_name': exp.get('experiment_name', ''),
        'hidden_dim': exp.get('config', {}).get('hidden_dim', ''),
        'latent_dim': exp.get('config', {}).get('latent_dim', ''),
        'num_layers': exp.get('config', {}).get('num_layers', ''),
        'batch_size': exp.get('config', {}).get('batch_size', ''),
        'beta': exp.get('config', {}).get('beta', ''),
        'learning_rate': exp.get('config', {}).get('learning_rate', ''),
        'dropout': exp.get('config', {}).get('dropout', ''),
        'final_val_loss': exp.get('final_val_loss', [None])[-1] if isinstance(exp.get('final_val_loss'), list) else exp.get('final_val_loss'),
        'best_val_loss': exp.get('best_val_loss'),
        'num_epochs': exp.get('num_epochs'),
        'final_val_recon_loss': exp.get('final_val_recon_loss', [None])[-1] if isinstance(exp.get('final_val_recon_loss'), list) else exp.get('final_val_recon_loss'),
        'final_val_kl_loss': exp.get('final_val_kl_loss', [None])[-1] if isinstance(exp.get('final_val_kl_loss'), list) else exp.get('final_val_kl_loss'),
        'final_val_speed_mae': exp.get('final_val_speed_mae', [None])[-1] if isinstance(exp.get('final_val_speed_mae'), list) else exp.get('final_val_speed_mae'),
        'status': exp.get('status', 'unknown')
    }
    df_summary.append(row)

df = pd.DataFrame(df_summary)
df.to_csv(results_dir / "experiments_summary.csv", index=False)

print(f"Summary saved to: {results_dir / 'experiments_summary.json'}")
print(f"CSV summary saved to: {results_dir / 'experiments_summary.csv'}")
print(f"\\nTop 5 experiments by validation loss:")
if not df.empty and 'best_val_loss' in df.columns:
    top_experiments = df.dropna(subset=['best_val_loss']).nsmallest(5, 'best_val_loss')
    print(top_experiments[['experiment_name', 'best_val_loss', 'final_val_speed_mae']].to_string(index=False))
EOF
fi

echo "Experiment summary completed!"