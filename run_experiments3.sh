#!/bin/bash
# run_vae_gan_experiments.sh
# Script focused on VAE-GAN experiments with various configurations

# Default values
GPU_ID=${1:-2}  # Default to GPU 2
DATA_PATH=${2:-"./preprocessing/vae_dataset.npz"}
BASE_RESULTS_DIR=${3:-"results/vae_gan_experiments"}
DRY_RUN=${4:-false}

# Create results directory
mkdir -p ${BASE_RESULTS_DIR}

# Function to run a single experiment
run_experiment() {
    local exp_name=$1
    local architecture=$2
    local hidden_dim=$3
    local latent_dim=$4
    local num_layers=$5
    local batch_size=$6
    local beta=$7
    local gamma=$8
    local disc_hidden_dim=$9
    local additional_args=${10}
    
    local results_dir="${BASE_RESULTS_DIR}/${exp_name}"
    
    echo "=========================================="
    echo "Running VAE-GAN experiment: $exp_name"
    echo "Architecture: $architecture"
    echo "Hidden dim: $hidden_dim, Latent dim: $latent_dim"
    echo "Layers: $num_layers, Batch size: $batch_size"
    echo "Beta: $beta, Gamma: $gamma"
    echo "Discriminator hidden dim: $disc_hidden_dim"
    echo "Results directory: $results_dir"
    echo "=========================================="
    
    local cmd="python ./train.py \
        --data-path $DATA_PATH \
        --results-dir $results_dir \
        --epochs 150 \
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
        --early-stopping-patience 25 \
        --use-gan \
        --gamma $gamma \
        --disc-lr 1e-4 \
        --disc-hidden-dim $disc_hidden_dim"
    
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
    \"gpu_id\": $GPU_ID,
    \"model_type\": \"vae_gan\",
    \"gamma\": $gamma,
    \"disc_hidden_dim\": $disc_hidden_dim
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
    \"model_type\": \"vae_gan\",
    \"gamma\": $gamma,
    \"disc_hidden_dim\": $disc_hidden_dim
}" > "$results_dir/experiment_info.json"
    else
        echo "DRY RUN - Skipping execution"
    fi
    
    echo -e "\n\n"
}

# ======================
# VAE-GAN EXPERIMENT DEFINITIONS
# ======================

echo "Starting VAE-GAN experiments on GPU $GPU_ID"
echo "Data path: $DATA_PATH"
echo "Base results directory: $BASE_RESULTS_DIR"
echo "Dry run: $DRY_RUN"
echo ""

# --- Gamma Parameter Study ---

# Very low gamma (VAE dominates)
run_experiment \
    "lstm_gan_gamma_0.1" \
    "lstm" \
    256 \
    32 \
    2 \
    24 \
    0.005 \
    0.1 \
    128 \
    "--disc-num-layers 2"

# Balanced gamma
run_experiment \
    "lstm_gan_gamma_1.0" \
    "lstm" \
    256 \
    32 \
    2 \
    24 \
    0.005 \
    1.0 \
    128 \
    "--disc-num-layers 2"

# High gamma (GAN dominates)
run_experiment \
    "lstm_gan_gamma_5.0" \
    "lstm" \
    256 \
    32 \
    2 \
    24 \
    0.005 \
    5.0 \
    128 \
    "--disc-num-layers 2"

# --- Discriminator Architecture Study ---

# Small discriminator
run_experiment \
    "lstm_gan_small_disc" \
    "lstm" \
    256 \
    32 \
    2 \
    24 \
    0.005 \
    1.0 \
    64 \
    "--disc-num-layers 2"

# Large discriminator
run_experiment \
    "lstm_gan_large_disc" \
    "lstm" \
    256 \
    32 \
    2 \
    24 \
    0.005 \
    1.0 \
    256 \
    "--disc-num-layers 3"

# Deep discriminator
run_experiment \
    "lstm_gan_deep_disc" \
    "lstm" \
    256 \
    32 \
    2 \
    24 \
    0.005 \
    1.0 \
    128 \
    "--disc-num-layers 4"

# --- Attention VAE-GAN Variants ---

# Basic attention VAE-GAN
run_experiment \
    "attention_gan_basic" \
    "attention" \
    256 \
    32 \
    2 \
    16 \
    0.005 \
    1.0 \
    128 \
    "--n-heads 8 --pooling mean --disc-num-layers 2"

# Attention VAE-GAN with CLS pooling
run_experiment \
    "attention_gan_cls" \
    "attention" \
    256 \
    32 \
    2 \
    16 \
    0.005 \
    1.0 \
    128 \
    "--n-heads 8 --pooling cls --disc-num-layers 2"

# Causal attention VAE-GAN
run_experiment \
    "attention_gan_causal" \
    "attention" \
    256 \
    32 \
    2 \
    16 \
    0.005 \
    1.0 \
    128 \
    "--n-heads 8 --use-causal-mask --pooling mean --disc-num-layers 2"

# --- Beta-Gamma Interaction Study ---

# Low beta, low gamma
run_experiment \
    "lstm_gan_low_beta_low_gamma" \
    "lstm" \
    256 \
    32 \
    2 \
    24 \
    0.001 \
    0.5 \
    128 \
    "--disc-num-layers 2"

# Low beta, high gamma
run_experiment \
    "lstm_gan_low_beta_high_gamma" \
    "lstm" \
    256 \
    32 \
    2 \
    24 \
    0.001 \
    2.0 \
    128 \
    "--disc-num-layers 2"

# High beta, low gamma
run_experiment \
    "lstm_gan_high_beta_low_gamma" \
    "lstm" \
    256 \
    32 \
    2 \
    24 \
    0.01 \
    0.5 \
    128 \
    "--disc-num-layers 2"

# High beta, high gamma
run_experiment \
    "lstm_gan_high_beta_high_gamma" \
    "lstm" \
    256 \
    32 \
    2 \
    24 \
    0.01 \
    2.0 \
    128 \
    "--disc-num-layers 2"

# --- Learning Rate Studies for GAN ---

# Different learning rates for generator and discriminator
run_experiment \
    "lstm_gan_disc_lr_high" \
    "lstm" \
    256 \
    32 \
    2 \
    24 \
    0.005 \
    1.0 \
    128 \
    "--disc-lr 5e-4 --disc-num-layers 2"

run_experiment \
    "lstm_gan_disc_lr_low" \
    "lstm" \
    256 \
    32 \
    2 \
    24 \
    0.005 \
    1.0 \
    128 \
    "--disc-lr 5e-5 --disc-num-layers 2"

# --- Update Frequency Study ---

# Update discriminator every 2 generator updates
run_experiment \
    "lstm_gan_disc_update_2" \
    "lstm" \
    256 \
    32 \
    2 \
    24 \
    0.005 \
    1.0 \
    128 \
    "--disc-update-freq 2 --disc-num-layers 2"

# Update discriminator every 5 generator updates
run_experiment \
    "lstm_gan_disc_update_5" \
    "lstm" \
    256 \
    32 \
    2 \
    24 \
    0.005 \
    1.0 \
    128 \
    "--disc-update-freq 5 --disc-num-layers 2"

# --- Large Model VAE-GAN ---

# Large LSTM VAE-GAN
run_experiment \
    "lstm_gan_large" \
    "lstm" \
    512 \
    64 \
    3 \
    16 \
    0.005 \
    1.0 \
    256 \
    "--disc-num-layers 3"

# Large attention VAE-GAN
run_experiment \
    "attention_gan_large" \
    "attention" \
    512 \
    64 \
    3 \
    12 \
    0.005 \
    1.0 \
    256 \
    "--n-heads 16 --pooling mean --d-ff 2048 --disc-num-layers 3"

# --- Optimized Configurations ---

# Best expected LSTM VAE-GAN
run_experiment \
    "lstm_gan_optimized" \
    "lstm" \
    384 \
    48 \
    3 \
    20 \
    0.003 \
    1.5 \
    192 \
    "--lr 2e-4 --disc-lr 2e-4 --disc-num-layers 3 --condition-dim 48"

# Best expected attention VAE-GAN
run_experiment \
    "attention_gan_optimized" \
    "attention" \
    384 \
    48 \
    2 \
    16 \
    0.003 \
    1.5 \
    192 \
    "--n-heads 12 --pooling mean --d-ff 1536 --lr 2e-4 --disc-lr 2e-4 --disc-num-layers 3"

echo "All VAE-GAN experiments completed!"
echo "Results saved in: $BASE_RESULTS_DIR"

# Generate summary report
if [ "$DRY_RUN" = false ]; then
    echo "Generating VAE-GAN summary report..."
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
                    info['final_gen_loss'] = history['val_gen_loss'][-1] if 'val_gen_loss' in history else None
                    info['final_disc_loss'] = history['val_disc_loss'][-1] if 'val_disc_loss' in history else None
                    info['num_epochs'] = len(history['val_loss'])
            
            summary.append(info)

# Save summary
with open(results_dir / "vae_gan_experiments_summary.json", 'w') as f:
    json.dump(summary, f, indent=2)

print(f"VAE-GAN summary saved to: {results_dir / 'vae_gan_experiments_summary.json'}")
EOF
fi