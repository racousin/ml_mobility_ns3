#!/bin/bash
# run_experiments.sh
# Script to run multiple VAE/VAE-GAN experiments with different configurations

# Default values
GPU_ID=${1:-0}
DATA_PATH=${2:-"./preprocessing/vae_dataset.npz"}
BASE_RESULTS_DIR=${3:-"results/experiments"}
DRY_RUN=${4:-false}

# Create results directory
mkdir -p ${BASE_RESULTS_DIR}

# Note: GPU memory estimation removed for simplicity
# All experiments are designed to fit in 40GB GPU memory

# Function to run a single experiment
run_experiment() {
    local exp_name=$1
    local architecture=$2
    local hidden_dim=$3
    local latent_dim=$4
    local num_layers=$5
    local batch_size=$6
    local beta=$7
    local use_gan=$8
    local additional_args=$9
    
    local results_dir="${BASE_RESULTS_DIR}/${exp_name}"
    
    echo "=========================================="
    echo "Running experiment: $exp_name"
    echo "Architecture: $architecture"
    echo "Hidden dim: $hidden_dim, Latent dim: $latent_dim"
    echo "Layers: $num_layers, Batch size: $batch_size"
    echo "Beta: $beta, Use GAN: $use_gan"
    echo "Results directory: $results_dir"
    echo "=========================================="
    
    local cmd="python ./train.py \
        --data-path $DATA_PATH \
        --results-dir $results_dir \
        --epochs 100 \
        --batch-size $batch_size \
        --beta $beta \
        --architecture $architecture \
        --hidden-dim $hidden_dim \
        --latent-dim $latent_dim \
        --num-layers $num_layers \
        --device cuda \
        --gpu-id $GPU_ID \
        --lr 1e-4 \
        --lr-patience 10 \
        --lr-factor 0.5 \
        --early-stopping-patience 20"
    
    # Add GAN-specific arguments
    if [ "$use_gan" = true ]; then
        cmd="$cmd --use-gan --gamma 1.0 --disc-lr 1e-4 --disc-hidden-dim 128"
    fi
    
    # Add additional arguments (e.g., for attention)
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
    \"gpu_id\": $GPU_ID
}" > "$results_dir/experiment_info.json"
        
        # Run the experiment
        $cmd
        
        # Update experiment info with end time
        echo "{
    \"experiment_name\": \"$exp_name\",
    \"command\": \"$cmd\",
    \"start_time\": \"$(date)\",
    \"end_time\": \"$(date)\",
    \"gpu_id\": $GPU_ID
}" > "$results_dir/experiment_info.json"
    else
        echo "DRY RUN - Skipping execution"
    fi
    
    echo -e "\n\n"
}

# ======================
# EXPERIMENT DEFINITIONS
# ======================

echo "Starting experiments on GPU $GPU_ID"
echo "Data path: $DATA_PATH"
echo "Base results directory: $BASE_RESULTS_DIR"
echo "Dry run: $DRY_RUN"
echo ""

# --- LSTM VAE Experiments ---

# Small LSTM VAE (baseline)
run_experiment \
    "lstm_vae_small" \
    "lstm" \
    128 \
    16 \
    2 \
    64 \
    0.001 \
    false \
    ""

# Medium LSTM VAE
run_experiment \
    "lstm_vae_medium" \
    "lstm" \
    256 \
    32 \
    2 \
    32 \
    0.005 \
    false \
    ""

# Large LSTM VAE (if memory allows)
run_experiment \
    "lstm_vae_large" \
    "lstm" \
    512 \
    64 \
    3 \
    16 \
    0.01 \
    false \
    ""

# --- Attention VAE Experiments ---

# Small Attention VAE
run_experiment \
    "attention_vae_small" \
    "attention" \
    128 \
    16 \
    2 \
    32 \
    0.005 \
    false \
    "--n-heads 4 --pooling mean"

# Medium Attention VAE with different pooling
run_experiment \
    "attention_vae_medium_cls" \
    "attention" \
    256 \
    32 \
    2 \
    16 \
    0.005 \
    false \
    "--n-heads 8 --pooling cls"

# Attention VAE with causal mask
run_experiment \
    "attention_vae_causal" \
    "attention" \
    128 \
    32 \
    3 \
    24 \
    0.005 \
    false \
    "--n-heads 4 --use-causal-mask --pooling mean"

# --- VAE-GAN Experiments ---

# Small LSTM VAE-GAN
run_experiment \
    "lstm_vae_gan_small" \
    "lstm" \
    128 \
    16 \
    2 \
    32 \
    0.001 \
    true \
    "--disc-num-layers 2"

# Attention VAE-GAN (smaller batch to fit memory)
run_experiment \
    "attention_vae_gan" \
    "attention" \
    128 \
    16 \
    2 \
    16 \
    0.005 \
    true \
    "--n-heads 4 --pooling mean --disc-num-layers 2"

# --- Beta Ablation Study ---

# Different beta values for KL weight
for beta in 0.0001 0.001 0.01 0.1; do
    run_experiment \
        "lstm_vae_beta_${beta}" \
        "lstm" \
        128 \
        32 \
        2 \
        32 \
        $beta \
        false \
        ""
done

# --- Latent Dimension Study ---

# Different latent dimensions
for latent_dim in 8 16 32 64; do
    run_experiment \
        "lstm_vae_latent_${latent_dim}" \
        "lstm" \
        128 \
        $latent_dim \
        2 \
        32 \
        0.005 \
        false \
        ""
done

echo "All experiments completed!"
echo "Results saved in: $BASE_RESULTS_DIR"

# Generate summary report
if [ "$DRY_RUN" = false ]; then
    echo "Generating summary report..."
    python - << EOF
import json
import os
from pathlib import Path

results_dir = Path("$BASE_RESULTS_DIR")
summary = []

for exp_dir in results_dir.iterdir():
    if exp_dir.is_dir():
        info_file = exp_dir / "experiment_info.json"
        history_file = exp_dir / "history.json"
        
        if info_file.exists():
            with open(info_file) as f:
                info = json.load(f)
            
            if history_file.exists():
                with open(history_file) as f:
                    history = json.load(f)
                    info['final_val_loss'] = history['val_loss'][-1] if history['val_loss'] else None
                    info['best_val_loss'] = min(history['val_loss']) if history['val_loss'] else None
                    info['num_epochs'] = len(history['val_loss'])
            
            summary.append(info)

# Save summary
with open(results_dir / "experiments_summary.json", 'w') as f:
    json.dump(summary, f, indent=2)

print(f"Summary saved to: {results_dir / 'experiments_summary.json'}")
EOF
fi