#!/bin/bash

# Script to run parallel experiments on GPUs 2, 4, 5, 6
# Each experiment runs on a separate GPU to maximize throughput

echo "Starting parallel experiments on GPUs 2, 4, 5, 6..."
echo "========================================"

# Create experiments directory if it doesn't exist
mkdir -p experiments/parallel_run_$(date +%Y%m%d_%H%M%S)

# Function to run experiment on specific GPU
run_experiment() {
    local gpu_id=$1
    local model=$2
    local training=$3
    local hidden_dim=$4
    local latent_dim=$5
    local lr=$6
    local beta_end=$7
    local free_bits=$8
    local batch_size=${9:-32}
    
    echo "Starting experiment on GPU $gpu_id: model=$model, training=$training, hidden=$hidden_dim, latent=$latent_dim, lr=$lr, beta=$beta_end"
    
    CUDA_VISIBLE_DEVICES=$gpu_id python scripts/train.py \
        model=$model \
        training=$training \
        model.hidden_dim=$hidden_dim \
        model.latent_dim=$latent_dim \
        training.learning_rate=$lr \
        training.loss.params.beta.params.end=$beta_end \
        training.loss.params.free_bits.lambda_free_bits=$free_bits \
        training.batch_size=$batch_size \
        accelerator=gpu \
        devices=[0] &
}

# Experiment Set 1: Best performers with patient training
run_experiment 2 vae_lstm patient_training 256 32 0.001 0.005 3.0 32
run_experiment 4 vae_lstm patient_training 256 32 0.0005 0.01 2.0 32
run_experiment 5 vae_lstm warmup_training 512 32 0.001 0.001 4.0 32
run_experiment 6 vae_lstm warmup_training 256 16 0.0005 0.005 3.0 32

# Wait for first batch to complete
wait

echo "First batch complete. Starting second batch..."

# Experiment Set 2: Attention models and variations
run_experiment 2 vae_attention patient_training 128 32 0.001 0.005 3.0 32
run_experiment 4 vae_attention patient_training 128 16 0.0005 0.01 2.0 32
run_experiment 5 vae_attention warmup_training 256 32 0.001 0.001 4.0 32
run_experiment 6 vae_lstm patient_training 128 16 0.002 0.002 3.5 32

# Wait for second batch
wait

echo "Second batch complete. Starting third batch..."

# Experiment Set 3: Larger models with different strategies
run_experiment 2 vae_lstm patient_training 512 64 0.0005 0.001 4.0 64
run_experiment 4 vae_lstm warmup_training 256 32 0.001 0.003 3.0 32
run_experiment 5 vae_lstm patient_training 384 48 0.0008 0.002 3.5 48
run_experiment 6 vae_attention warmup_training 192 24 0.0007 0.004 3.2 32

# Wait for third batch
wait

echo "Third batch complete. Starting fourth batch..."

# Experiment Set 4: Fine-tuned configurations based on best practices
run_experiment 2 vae_lstm patient_training 256 32 0.0003 0.008 2.5 32
run_experiment 4 vae_lstm warmup_training 320 40 0.0006 0.002 3.8 40
run_experiment 5 vae_attention patient_training 160 20 0.0009 0.006 2.8 32
run_experiment 6 vae_lstm patient_training 192 24 0.0004 0.007 3.3 32

# Wait for all experiments to complete
wait

echo "========================================"
echo "All experiments completed!"
echo "Run 'python scripts/list_experiments.py' to see results"