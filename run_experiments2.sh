#!/bin/bash
# run_experiments_2.sh
# Second script with different experiments to run on another GPU in parallel

# Default values
GPU_ID=${1:-1}  # Default to GPU 1 (different from first script)
DATA_PATH=${2:-"./preprocessing/vae_dataset.npz"}
BASE_RESULTS_DIR=${3:-"results/experiments_gpu2"}
DRY_RUN=${4:-false}

# Create results directory
mkdir -p ${BASE_RESULTS_DIR}

# Function to estimate GPU memory usage
estimate_gpu_memory() {
    local batch_size=$1
    local hidden_dim=$2
    local num_layers=$3
    local architecture=$4
    local use_gan=$5
    
    # Rough estimation in GB (conservative)
    local base_mem=2.0  # Base memory for model and data
    local layer_mem=$(echo "$num_layers * 0.5" | bc -l)
    local hidden_mem=$(echo "$hidden_dim * 0.008" | bc -l)
    local batch_mem=$(echo "$batch_size * 0.05" | bc -l)
    
    if [ "$architecture" = "attention" ]; then
        base_mem=$(echo "$base_mem + 1.0" | bc -l)
    fi
    
    if [ "$use_gan" = true ]; then
        base_mem=$(echo "$base_mem * 1.8" | bc -l)
    fi
    
    local total_mem=$(echo "$base_mem + $layer_mem + $hidden_mem + $batch_mem" | bc -l)
    echo $total_mem
}

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
    
    # Estimate memory usage
    local mem_usage=$(estimate_gpu_memory $batch_size $hidden_dim $num_layers $architecture $use_gan)
    echo "Estimated GPU memory usage: ${mem_usage} GB"
    
    if (( $(echo "$mem_usage > 38" | bc -l) )); then
        echo "WARNING: Estimated memory usage exceeds 38GB. Skipping experiment: $exp_name"
        return
    fi
    
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
    \"gpu_id\": $GPU_ID,
    \"estimated_gpu_memory_gb\": $mem_usage
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
    \"estimated_gpu_memory_gb\": $mem_usage
}" > "$results_dir/experiment_info.json"
    else
        echo "DRY RUN - Skipping execution"
    fi
    
    echo -e "\n\n"
}

# ======================
# EXPERIMENT DEFINITIONS - DIFFERENT FROM SCRIPT 1
# ======================

echo "Starting experiments on GPU $GPU_ID"
echo "Data path: $DATA_PATH"
echo "Base results directory: $BASE_RESULTS_DIR"
echo "Dry run: $DRY_RUN"
echo ""

# --- Advanced LSTM Experiments ---

# LSTM with very deep architecture
run_experiment \
    "lstm_vae_deep" \
    "lstm" \
    256 \
    32 \
    4 \
    16 \
    0.005 \
    false \
    ""

# LSTM with wide latent space
run_experiment \
    "lstm_vae_wide_latent" \
    "lstm" \
    256 \
    128 \
    2 \
    16 \
    0.01 \
    false \
    ""

# --- Advanced Attention Experiments ---

# Attention with max pooling
run_experiment \
    "attention_vae_max_pool" \
    "attention" \
    256 \
    32 \
    3 \
    24 \
    0.005 \
    false \
    "--n-heads 8 --pooling max"

# Large attention model
run_experiment \
    "attention_vae_large" \
    "attention" \
    512 \
    64 \
    3 \
    16 \
    0.01 \
    false \
    "--n-heads 16 --pooling mean --d-ff 2048"

# Attention with high dropout
run_experiment \
    "attention_vae_high_dropout" \
    "attention" \
    256 \
    32 \
    2 \
    24 \
    0.005 \
    false \
    "--n-heads 8 --pooling mean --dropout 0.3"

# --- Mixed Architecture VAE-GAN ---

# Medium LSTM VAE-GAN with different gamma
run_experiment \
    "lstm_vae_gan_gamma_0.5" \
    "lstm" \
    256 \
    32 \
    3 \
    24 \
    0.005 \
    true \
    "--gamma 0.5 --disc-num-layers 3"

# LSTM VAE-GAN with high gamma
run_experiment \
    "lstm_vae_gan_gamma_2.0" \
    "lstm" \
    128 \
    32 \
    2 \
    32 \
    0.005 \
    true \
    "--gamma 2.0 --disc-num-layers 2"

# Attention VAE-GAN with causal mask
run_experiment \
    "attention_vae_gan_causal" \
    "attention" \
    128 \
    32 \
    2 \
    16 \
    0.005 \
    true \
    "--n-heads 4 --use-causal-mask --pooling mean --disc-num-layers 2"

# --- Learning Rate Studies ---

# High initial learning rate
run_experiment \
    "lstm_vae_high_lr" \
    "lstm" \
    256 \
    32 \
    2 \
    32 \
    0.005 \
    false \
    "--lr 5e-4"

# Low initial learning rate
run_experiment \
    "lstm_vae_low_lr" \
    "lstm" \
    256 \
    32 \
    2 \
    32 \
    0.005 \
    false \
    "--lr 1e-5"

# --- Batch Size Studies ---

# Very small batch size
run_experiment \
    "lstm_vae_batch_8" \
    "lstm" \
    256 \
    32 \
    2 \
    8 \
    0.005 \
    false \
    ""

# Large batch size (if memory allows)
run_experiment \
    "lstm_vae_batch_128" \
    "lstm" \
    128 \
    16 \
    2 \
    128 \
    0.005 \
    false \
    ""

# --- Condition Dimension Studies ---

# Small condition dimension
run_experiment \
    "lstm_vae_cond_16" \
    "lstm" \
    256 \
    32 \
    2 \
    32 \
    0.005 \
    false \
    "--condition-dim 16"

# Large condition dimension
run_experiment \
    "lstm_vae_cond_64" \
    "lstm" \
    256 \
    32 \
    2 \
    32 \
    0.005 \
    false \
    "--condition-dim 64"

# --- Scheduler Studies ---

# Aggressive learning rate reduction
run_experiment \
    "lstm_vae_aggressive_lr" \
    "lstm" \
    256 \
    32 \
    2 \
    32 \
    0.005 \
    false \
    "--lr-patience 5 --lr-factor 0.2"

# Conservative learning rate reduction
run_experiment \
    "lstm_vae_conservative_lr" \
    "lstm" \
    256 \
    32 \
    2 \
    32 \
    0.005 \
    false \
    "--lr-patience 20 --lr-factor 0.8"

# --- Combined Studies ---

# Best LSTM configuration from script 1 with modifications
run_experiment \
    "lstm_vae_optimized" \
    "lstm" \
    384 \
    48 \
    3 \
    24 \
    0.003 \
    false \
    "--lr 2e-4 --condition-dim 48"

# Best attention configuration with modifications
run_experiment \
    "attention_vae_optimized" \
    "attention" \
    384 \
    48 \
    3 \
    20 \
    0.003 \
    false \
    "--n-heads 12 --pooling mean --d-ff 1536"

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