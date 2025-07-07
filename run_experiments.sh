#!/bin/bash
# run_focused_vae_experiments.sh
# Focused experiments based on successful configurations from analysis
# Targeting the promising small-to-medium VAE models with low beta values

# Default values
GPU_ID=${1:-0}
DATA_PATH=${2:-"./preprocessing/vae_dataset.npz"}
BASE_RESULTS_DIR=${3:-"results/focused_vae"}
DRY_RUN=${4:-false}

# Create results directory
mkdir -p ${BASE_RESULTS_DIR}

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
    echo "Hidden: $hidden_dim, Latent: $latent_dim, Layers: $num_layers"
    echo "Batch: $batch_size, Beta: $beta, LR: $learning_rate, Dropout: $dropout"
    echo "=========================================="
    
    local cmd="python ./train.py \
        --data-path $DATA_PATH \
        --experiment-name $exp_name \
        --epochs 150 \
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
        --early-stopping-patience 30 \
        --gradient-clip 1.0"
    
    # Add additional arguments
    if [ ! -z "$additional_args" ]; then
        cmd="$cmd $additional_args"
    fi
    
    echo "Command: $cmd"
    
    if [ "$DRY_RUN" = false ]; then
        mkdir -p $results_dir
        cat > "$results_dir/experiment_info.json" << EOF
{
    "experiment_name": "$exp_name",
    "command": "$cmd",
    "start_time": "$(date)",
    "gpu_id": $GPU_ID,
    "focus": "refined_vae",
    "config": {
        "hidden_dim": $hidden_dim,
        "latent_dim": $latent_dim,
        "num_layers": $num_layers,
        "batch_size": $batch_size,
        "beta": $beta,
        "learning_rate": $learning_rate,
        "dropout": $dropout
    }
}
EOF
        
        # Run the experiment
        $cmd
        
        # Update with completion time
        cat > "$results_dir/experiment_info.json" << EOF
{
    "experiment_name": "$exp_name",
    "command": "$cmd",
    "start_time": "$(date)",
    "end_time": "$(date)",
    "gpu_id": $GPU_ID,
    "focus": "refined_vae",
    "status": "completed",
    "config": {
        "hidden_dim": $hidden_dim,
        "latent_dim": $latent_dim,
        "num_layers": $num_layers,
        "batch_size": $batch_size,
        "beta": $beta,
        "learning_rate": $learning_rate,
        "dropout": $dropout
    }
}
EOF
    else
        echo "DRY RUN - Skipping execution"
    fi
    
    echo -e "\n"
}

echo "=== FOCUSED VAE EXPERIMENTS ==="
echo "Based on analysis: focusing on small-medium models with low beta"
echo "GPU: $GPU_ID, Data: $DATA_PATH, Results: $BASE_RESULTS_DIR"
echo ""

# ======================
# OPTIMAL SMALL MODELS (Best performer variations)
# ======================

echo "--- Optimal Small Model Variations ---"

# Best performer refinement - very similar to winning config
run_experiment \
    "optimal_small_v1" \
    128 \
    16 \
    2 \
    64 \
    0.001 \
    1e-4 \
    0.1 \
    ""

# Slight variations of the best performer
run_experiment \
    "optimal_small_v2" \
    128 \
    16 \
    2 \
    64 \
    0.0005 \
    1e-4 \
    0.1 \
    ""

run_experiment \
    "optimal_small_v3" \
    128 \
    20 \
    2 \
    64 \
    0.001 \
    1e-4 \
    0.1 \
    ""

run_experiment \
    "optimal_small_v4" \
    144 \
    16 \
    2 \
    64 \
    0.001 \
    1e-4 \
    0.1 \
    ""

# ======================
# REFINED MEDIUM MODELS
# ======================

echo "--- Refined Medium Models ---"

# Medium model with lower beta (second best performer direction)
run_experiment \
    "optimal_medium_v1" \
    256 \
    32 \
    2 \
    32 \
    0.002 \
    1e-4 \
    0.1 \
    ""

run_experiment \
    "optimal_medium_v2" \
    256 \
    28 \
    2 \
    32 \
    0.003 \
    1e-4 \
    0.1 \
    ""

run_experiment \
    "optimal_medium_v3" \
    240 \
    32 \
    2 \
    32 \
    0.002 \
    1e-4 \
    0.1 \
    ""

# ======================
# BETA FINE-TUNING (Critical parameter)
# ======================

echo "--- Beta Fine-tuning Around Optimal Values ---"

# Very fine beta exploration around best values
for beta in 0.0005 0.0008 0.0012 0.0015 0.002 0.003; do
    beta_name=$(echo $beta | sed 's/\./_/g')
    run_experiment \
        "beta_fine_${beta_name}" \
        128 \
        16 \
        2 \
        64 \
        $beta \
        1e-4 \
        0.1 \
        ""
done

# ======================
# ARCHITECTURE REFINEMENTS
# ======================

echo "--- Architecture Refinements ---"

# Slightly deeper but not too deep
run_experiment \
    "arch_deep_small" \
    128 \
    16 \
    3 \
    48 \
    0.001 \
    1e-4 \
    0.15 \
    ""

# Wider but shallow
run_experiment \
    "arch_wide_shallow" \
    192 \
    24 \
    2 \
    48 \
    0.0015 \
    1e-4 \
    0.1 \
    ""

# Single layer with more capacity
run_experiment \
    "arch_single_wide" \
    256 \
    20 \
    1 \
    64 \
    0.001 \
    1e-4 \
    0.05 \
    ""

# ======================
# LEARNING RATE OPTIMIZATION
# ======================

echo "--- Learning Rate Optimization ---"

# Different learning rates with best config
for lr in 5e-5 8e-5 1.5e-4 2e-4; do
    lr_name=$(echo $lr | sed 's/e-/_e_/g' | sed 's/\./_/g')
    run_experiment \
        "lr_opt_${lr_name}" \
        128 \
        16 \
        2 \
        64 \
        0.001 \
        $lr \
        0.1 \
        ""
done

# ======================
# BATCH SIZE OPTIMIZATION
# ======================

echo "--- Batch Size Optimization ---"

# Test different batch sizes with optimal config
for batch_size in 48 80 96 128; do
    run_experiment \
        "batch_opt_${batch_size}" \
        128 \
        16 \
        2 \
        $batch_size \
        0.001 \
        1e-4 \
        0.1 \
        ""
done

# ======================
# REGULARIZATION REFINEMENTS
# ======================

echo "--- Regularization Refinements ---"

# Different dropout rates
for dropout in 0.05 0.08 0.12 0.15; do
    dropout_name=$(echo $dropout | sed 's/\./_/g')
    run_experiment \
        "dropout_opt_${dropout_name}" \
        128 \
        16 \
        2 \
        64 \
        0.001 \
        1e-4 \
        $dropout \
        ""
done

# ======================
# CONDITION EMBEDDING OPTIMIZATION
# ======================

echo "--- Condition Embedding Optimization ---"

# Different condition dimensions
for cond_dim in 16 24 40 48; do
    run_experiment \
        "cond_opt_${cond_dim}" \
        128 \
        16 \
        2 \
        64 \
        0.001 \
        1e-4 \
        0.1 \
        "--condition-dim $cond_dim"
done

# ======================
# HYBRID CONFIGURATIONS
# ======================

echo "--- Hybrid Best Configurations ---"

# Combination of best settings from different aspects
run_experiment \
    "hybrid_optimal_1" \
    144 \
    20 \
    2 \
    64 \
    0.0008 \
    8e-5 \
    0.08 \
    "--condition-dim 24"

run_experiment \
    "hybrid_optimal_2" \
    160 \
    18 \
    2 \
    48 \
    0.0012 \
    1.2e-4 \
    0.12 \
    "--condition-dim 20"

run_experiment \
    "hybrid_optimal_3" \
    136 \
    22 \
    2 \
    56 \
    0.0006 \
    9e-5 \
    0.09 \
    "--condition-dim 28"

echo "All focused VAE experiments completed!"
echo "Results saved in: $BASE_RESULTS_DIR"

# Generate summary
if [ "$DRY_RUN" = false ]; then
    echo "Generating focused experiments summary..."
    python - << EOF
import json
import pandas as pd
from pathlib import Path

results_dir = Path("$BASE_RESULTS_DIR")
experiments = []

for exp_dir in results_dir.iterdir():
    if exp_dir.is_dir():
        info_file = exp_dir / "experiment_info.json"
        history_file = exp_dir / "history.json"
        
        if info_file.exists():
            with open(info_file) as f:
                info = json.load(f)
            
            exp_data = {
                'experiment': info.get('experiment_name', ''),
                'hidden_dim': info.get('config', {}).get('hidden_dim', ''),
                'latent_dim': info.get('config', {}).get('latent_dim', ''),
                'beta': info.get('config', {}).get('beta', ''),
                'learning_rate': info.get('config', {}).get('learning_rate', ''),
                'dropout': info.get('config', {}).get('dropout', ''),
                'batch_size': info.get('config', {}).get('batch_size', ''),
            }
            
            if history_file.exists():
                with open(history_file) as f:
                    history = json.load(f)
                    if history.get('val_loss'):
                        exp_data['best_val_loss'] = min(history['val_loss'])
                        exp_data['final_val_loss'] = history['val_loss'][-1]
                        exp_data['num_epochs'] = len(history['val_loss'])
                        if history.get('val_speed_mae'):
                            exp_data['best_speed_mae'] = min(history['val_speed_mae'])
            
            experiments.append(exp_data)

if experiments:
    df = pd.DataFrame(experiments)
    df.to_csv(results_dir / "focused_experiments_summary.csv", index=False)
    
    # Find top performers
    if 'best_val_loss' in df.columns:
        top_5 = df.nsmallest(5, 'best_val_loss')
        print("\\nTop 5 experiments by validation loss:")
        print(top_5[['experiment', 'best_val_loss', 'hidden_dim', 'latent_dim', 'beta']].to_string(index=False))

print(f"\\nSummary saved to: {results_dir / 'focused_experiments_summary.csv'}")
EOF
fi