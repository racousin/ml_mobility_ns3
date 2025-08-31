#!/bin/bash
#SBATCH --job-name=vae_lstm_pretrained
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --time=00:03:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=electronic
#SBATCH --nodelist=kavinsky
#SBATCH --gres=gpu:1

# Create logs directory if it doesn't exist
mkdir -p logs

# Load modules (adjust based on your cluster)
# module load python/3.11
# module load cuda/11.8

# Install minimal requirements and activate environment
source venv/bin/activate



echo "Starting VAE-LSTM from pretrained training..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Run the training
python scripts/train.py +experiment=vae_lstm_from_pretrained training=adaptive_training accelerator=gpu devices=[0]

echo "Training completed!"