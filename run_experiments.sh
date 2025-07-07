#!/bin/bash
# run_advanced_vae_experiments.sh
# Advanced VAE techniques and architectural improvements
# Building on successful small-medium VAE configurations

# Default values
GPU_ID=${1:-1}
DATA_PATH=${2:-"./preprocessing/vae_dataset.npz"}
BASE_RESULTS_DIR=${3:-"results/advanced_vae"}
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
    echo "Running advanced experiment: $exp_name"
    echo "Hidden: $hidden_dim, Latent: $latent_dim, Layers: $num_layers"
    echo "Batch: $batch_size, Beta: $beta, LR: $learning_rate, Dropout: $dropout"
    echo "Additional: $additional_args"
    echo "=========================================="
    
    local cmd="python ./train.py \
        --data-path $DATA_PATH \
        --experiment-name $exp_name \
        --epochs 120 \
        --batch-size $batch_size \
        --lr $learning_rate \
        --beta $beta \
        --hidden-dim $hidden_dim \
        --latent-dim $latent_dim \
        --num-layers $num_layers \
        --dropout $dropout \
        --device cuda \
        --gpu-id $GPU_ID \
        --lr-patience 12 \
        --lr-factor 0.7 \
        --early-stopping-patience 25 \
        --gradient-clip 1.0"
    
    # Add additional arguments
    if [ ! -z "$additional_args" ]; then
        cmd="$cmd $additional_args"
    fi
    
    echo "Command: $cmd"
    
    if [ "$DRY_RUN" = false ]; then
        mkdir -p $results_dir
        echo "{
    \"experiment_name\": \"$exp_name\",
    \"command\": \"$cmd\",
    \"start_time\": \"$(date)\",
    \"gpu_id\": $GPU_ID,
    \"focus\": \"advanced_vae\",
    \"config\": {
        \"hidden_dim\": $hidden_dim,
        \"latent_dim\": $latent_dim,
        \"num_layers\": $num_layers,
        \"batch_size\": $batch_size,
        \"beta\": $beta,
        \"learning_rate\": $learning_rate,
        \"dropout\": $dropout,
        \"additional_args\": \"$additional_args\"
    }
}" > "$results_dir/experiment_info.json"
        
        # Run the experiment
        $cmd
        
        # Update with completion time
        jq '. + {"end_time": "'$(date)'", "status": "completed"}' "$results_dir/experiment_info.json" > "$results_dir/experiment_info_tmp.json" && mv "$results_dir/experiment_info_tmp.json" "$results_dir/experiment_info.json"
    else
        echo "DRY RUN - Skipping execution"
    fi
    
    echo -e "\n"
}

echo "=== ADVANCED VAE EXPERIMENTS ==="
echo "Exploring advanced techniques while staying with pure VAE"
echo "GPU: $GPU_ID, Data: $DATA_PATH, Results: $BASE_RESULTS_DIR"
echo ""

# ======================
# WARM-UP AND ANNEALING STRATEGIES
# ======================

echo "--- Beta Annealing Strategies ---"

# Beta warm-up: start from 0 and gradually increase
run_experiment \
    "beta_warmup_v1" \
    128 \
    16 \
    2 \
    64 \
    0.001 \
    1e-4 \
    0.1 \
    "--beta-warmup-epochs 20"

run_experiment \
    "beta_warmup_v2" \
    256 \
    32 \
    2 \
    32 \
    0.003 \
    1e-4 \
    0.1 \
    "--beta-warmup-epochs 30"

# Cyclical beta scheduling
run_experiment \
    "beta_cyclical" \
    128 \
    16 \
    2 \
    64 \
    0.001 \
    1e-4 \
    0.1 \
    "--beta-cyclical --beta-cycle-length 10"

# ======================
# ADVANCED OPTIMIZATION
# ======================

echo "--- Advanced Optimization Techniques ---"

# Different optimizers
run_experiment \
    "opt_adamw" \
    128 \
    16 \
    2 \
    64 \
    0.001 \
    1e-4 \
    0.1 \
    "--optimizer adamw --weight-decay 1e-4"

run_experiment \
    "opt_radam" \
    128 \
    16 \
    2 \
    64 \
    0.001 \
    1e-4 \
    0.1 \
    "--optimizer radam"

# Cosine annealing scheduler
run_experiment \
    "scheduler_cosine" \
    128 \
    16 \
    2 \
    64 \
    0.001 \
    2e-4 \
    0.1 \
    "--scheduler cosine --cosine-min-lr 1e-6"

# One cycle learning rate
run_experiment \
    "scheduler_onecycle" \
    128 \
    16 \
    2 \
    64 \
    0.001 \
    1e-4 \
    0.1 \
    "--scheduler onecycle --max-lr 5e-4"

# ======================
# ARCHITECTURAL IMPROVEMENTS
# ======================

echo "--- Architectural Improvements ---"

# Residual connections in LSTM
run_experiment \
    "arch_residual_small" \
    128 \
    16 \
    3 \
    64 \
    0.001 \
    1e-4 \
    0.1 \
    "--use-residual-connections"

run_experiment \
    "arch_residual_medium" \
    256 \
    32 \
    3 \
    32 \
    0.003 \
    1e-4 \
    0.15 \
    "--use-residual-connections"

# Layer normalization
run_experiment \
    "arch_layernorm" \
    128 \
    16 \
    2 \
    64 \
    0.001 \
    1e-4 \
    0.1 \
    "--use-layer-norm"

# Bidirectional encoding with unidirectional decoding
run_experiment \
    "arch_bidirectional" \
    128 \
    16 \
    2 \
    64 \
    0.001 \
    1e-4 \
    0.1 \
    "--bidirectional-encoder"

# Skip connections between encoder and decoder
run_experiment \
    "arch_skip_connections" \
    256 \
    32 \
    2 \
    32 \
    0.003 \
    1e-4 \
    0.1 \
    "--use-skip-connections"

# ======================
# ADVANCED REGULARIZATION
# ======================

echo "--- Advanced Regularization ---"

# Spectral normalization
run_experiment \
    "reg_spectral_norm" \
    128 \
    16 \
    2 \
    64 \
    0.001 \
    1e-4 \
    0.1 \
    "--spectral-normalization"

# Dropout variations
run_experiment \
    "reg_variational_dropout" \
    128 \
    16 \
    2 \
    64 \
    0.001 \
    1e-4 \
    0.1 \
    "--variational-dropout"

# Weight regularization
run_experiment \
    "reg_weight_decay" \
    128 \
    16 \
    2 \
    64 \
    0.001 \
    1e-4 \
    0.1 \
    "--weight-decay 1e-3"

# Batch normalization
run_experiment \
    "reg_batch_norm" \
    256 \
    32 \
    2 \
    32 \
    0.003 \
    1e-4 \
    0.1 \
    "--use-batch-norm"

# ======================
# CONDITIONING IMPROVEMENTS
# ======================

echo "--- Advanced Conditioning Strategies ---"

# Learned positional embeddings for sequence position
run_experiment \
    "cond_positional" \
    128 \
    16 \
    2 \
    64 \
    0.001 \
    1e-4 \
    0.1 \
    "--use-positional-encoding"

# Multiple condition injection points
run_experiment \
    "cond_multi_injection" \
    256 \
    32 \
    2 \
    32 \
    0.003 \
    1e-4 \
    0.1 \
    "--multi-condition-injection"

# Hierarchical conditioning
run_experiment \
    "cond_hierarchical" \
    128 \
    16 \
    2 \
    64 \
    0.001 \
    1e-4 \
    0.1 \
    "--hierarchical-conditioning --condition-dim 48"

# Attention-based condition fusion
run_experiment \
    "cond_attention_fusion" \
    256 \
    32 \
    2 \
    32 \
    0.003 \
    1e-4 \
    0.1 \
    "--attention-condition-fusion"

# ======================
# LOSS FUNCTION VARIATIONS
# ======================

echo "--- Advanced Loss Functions ---"

# Focal loss for reconstruction
run_experiment \
    "loss_focal" \
    128 \
    16 \
    2 \
    64 \
    0.001 \
    1e-4 \
    0.1 \
    "--focal-loss --focal-alpha 0.25 --focal-gamma 2.0"

# Huber loss (robust to outliers)
run_experiment \
    "loss_huber" \
    128 \
    16 \
    2 \
    64 \
    0.001 \
    1e-4 \
    0.1 \
    "--huber-loss --huber-delta 1.0"

# Perceptual loss using feature space
run_experiment \
    "loss_perceptual" \
    256 \
    32 \
    2 \
    32 \
    0.003 \
    1e-4 \
    0.1 \
    "--perceptual-loss"

# Combined reconstruction losses
run_experiment \
    "loss_combined" \
    128 \
    16 \
    2 \
    64 \
    0.001 \
    1e-4 \
    0.1 \
    "--combined-recon-loss --l1-weight 0.3 --l2-weight 0.7"

# ======================
# SEQUENCE MODELING IMPROVEMENTS
# ======================

echo "--- Sequence Modeling Improvements ---"

# Transformer-LSTM hybrid
run_experiment \
    "seq_transformer_hybrid" \
    256 \
    32 \
    2 \
    24 \
    0.003 \
    1e-4 \
    0.1 \
    "--transformer-lstm-hybrid --n-heads 4"

# Temporal convolutions
run_experiment \
    "seq_temporal_conv" \
    128 \
    16 \
    2 \
    64 \
    0.001 \
    1e-4 \
    0.1 \
    "--temporal-convolutions --conv-channels 64"

# Multi-scale temporal modeling
run_experiment \
    "seq_multiscale" \
    256 \
    32 \
    2 \
    32 \
    0.003 \
    1e-4 \
    0.1 \
    "--multiscale-temporal --scales 1,2,4"

# ======================
# TRAINING STRATEGIES
# ======================

echo "--- Advanced Training Strategies ---"

# Progressive training (start with shorter sequences)
run_experiment \
    "train_progressive" \
    128 \
    16 \
    2 \
    64 \
    0.001 \
    1e-4 \
    0.1 \
    "--progressive-training --start-length 500 --length-increase-epochs 15"

# Curriculum learning (easy to hard samples)
run_experiment \
    "train_curriculum" \
    256 \
    32 \
    2 \
    32 \
    0.003 \
    1e-4 \
    0.1 \
    "--curriculum-learning --difficulty-metric speed_variance"

# Self-paced learning
run_experiment \
    "train_selfpaced" \
    128 \
    16 \
    2 \
    64 \
    0.001 \
    1e-4 \
    0.1 \
    "--self-paced-learning --pace-increase-rate 0.1"

# Mixed precision training
run_experiment \
    "train_mixed_precision" \
    256 \
    32 \
    2 \
    32 \
    0.003 \
    1e-4 \
    0.1 \
    "--mixed-precision"

# ======================
# ENSEMBLE APPROACHES
# ======================

echo "--- Ensemble Approaches ---"

# Multi-head VAE (multiple decoders)
run_experiment \
    "ensemble_multihead" \
    128 \
    16 \
    2 \
    64 \
    0.001 \
    1e-4 \
    0.1 \
    "--multi-head-decoder --num-heads 3"

# Bootstrap aggregating
run_experiment \
    "ensemble_bootstrap" \
    128 \
    16 \
    2 \
    64 \
    0.001 \
    1e-4 \
    0.1 \
    "--bootstrap-training --bootstrap-ratio 0.8"

# ======================
# HYBRID BEST ADVANCED
# ======================

echo "--- Hybrid Advanced Configurations ---"

# Combine multiple advanced techniques
run_experiment \
    "advanced_hybrid_1" \
    144 \
    20 \
    2 \
    64 \
    0.0008 \
    1e-4 \
    0.08 \
    "--use-layer-norm --bidirectional-encoder --attention-condition-fusion --condition-dim 24"

run_experiment \
    "advanced_hybrid_2" \
    256 \
    32 \
    3 \
    32 \
    0.003 \
    1e-4 \
    0.12 \
    "--use-residual-connections --multi-condition-injection --huber-loss --optimizer adamw"

run_experiment \
    "advanced_hybrid_3" \
    160 \
    24 \
    2 \
    48 \
    0.0012 \
    1.2e-4 \
    0.1 \
    "--spectral-normalization --beta-warmup-epochs 25 --scheduler cosine --use-positional-encoding"

echo "All advanced VAE experiments completed!"
echo "Results saved in: $BASE_RESULTS_DIR"

# Generate summary
if [ "$DRY_RUN" = false ]; then
    echo "Generating advanced experiments summary..."
    python - << EOF
import json
import pandas as pd
from pathlib import Path
import numpy as np

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
                'category': info.get('experiment_name', '').split('_')[0] + '_' + info.get('experiment_name', '').split('_')[1] if '_' in info.get('experiment_name', '') else 'other',
                'hidden_dim': info.get('config', {}).get('hidden_dim', ''),
                'latent_dim': info.get('config', {}).get('latent_dim', ''),
                'beta': info.get('config', {}).get('beta', ''),
                'techniques': info.get('config', {}).get('additional_args', ''),
            }
            
            if history_file.exists():
                with open(history_file) as f:
                    history = json.load(f)
                    if history.get('val_loss'):
                        exp_data['best_val_loss'] = min(history['val_loss'])
                        exp_data['final_val_loss'] = history['val_loss'][-1]
                        exp_data['num_epochs'] = len(history['val_loss'])
                        exp_data['convergence_speed'] = np.argmin(history['val_loss']) + 1  # Epoch of best performance
                        if history.get('val_speed_mae'):
                            exp_data['best_speed_mae'] = min(history['val_speed_mae'])
            
            experiments.append(exp_data)

if experiments:
    df = pd.DataFrame(experiments)
    df.to_csv(results_dir / "advanced_experiments_summary.csv", index=False)
    
    print("\\nExperiment categories and their performance:")
    if 'best_val_loss' in df.columns:
        category_performance = df.groupby('category')['best_val_loss'].agg(['count', 'mean', 'min']).round(4)
        print(category_performance.to_string())
        
        print("\\nTop 5 experiments by validation loss:")
        top_5 = df.nsmallest(5, 'best_val_loss')
        print(top_5[['experiment', 'best_val_loss', 'category', 'convergence_speed']].to_string(index=False))
        
        print("\\nFastest converging experiments:")
        if 'convergence_speed' in df.columns:
            fastest = df.nsmallest(5, 'convergence_speed')
            print(fastest[['experiment', 'convergence_speed', 'best_val_loss', 'category']].to_string(index=False))

print(f"\\nAdvanced experiments summary saved to: {results_dir / 'advanced_experiments_summary.csv'}")
EOF
fi