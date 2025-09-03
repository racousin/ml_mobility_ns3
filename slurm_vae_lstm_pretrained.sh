#!/bin/bash
#SBATCH --job-name=vae_lstm_pretrained
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=electronic
#SBATCH --gres=gpu:1

# Load modules (adjust based on your cluster)
# module load python/3.11
# module load cuda/11.8

# Install minimal requirements and activate environment
source venv/bin/activate



echo "Starting VAE-LSTM from pretrained training..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Run the training - choose one of the following:

# Option 1: Standard beta annealing with MSE loss
python scripts/train.py +experiment=vae_lstm_from_pretrained training=adaptive_training accelerator=gpu devices=[0]

# Option 2: Distance-aware loss for better real-world metrics
# python scripts/train.py +experiment=vae_lstm_distance_aware training=adaptive_training accelerator=gpu devices=[0]

echo "Training completed!"