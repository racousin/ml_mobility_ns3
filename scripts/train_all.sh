#!/bin/bash
# Script to train all models in 2 batches of 4
# Usage: bash scripts/train_all.sh

cd /network-volume/ns3/ml_mobility_ns3

echo "========================================"
echo " Batch Training - 4 models at a time"
echo " Started at: $(date)"
echo "========================================"

# --- BATCH 1: 4 smaller/simpler models ---
echo ""
echo "========================================"
echo " BATCH 1: dummy, vae_lstm, vae_dense, vae_cnn"
echo " Started at: $(date)"
echo "========================================"

python scripts/train.py model=dummy &
PID1=$!
python scripts/train.py model=vae_lstm &
PID2=$!
python scripts/train.py model=vae_dense &
PID3=$!
python scripts/train.py model=vae_cnn &
PID4=$!

# Wait for all 4 to finish
wait $PID1 $PID2 $PID3 $PID4

echo "========================================"
echo " BATCH 1 DONE at: $(date)"
echo "========================================"

# Pause to let GPU memory fully release
sleep 10

# --- BATCH 2: 4 larger/complex models ---
echo ""
echo "========================================"
echo " BATCH 2: vae_attention, vq_vae, diffusion, gpt"
echo " Started at: $(date)"
echo "========================================"

python scripts/train.py model=vae_attention &
PID5=$!
python scripts/train.py model=vq_vae &
PID6=$!
python scripts/train.py model=diffusion training=diffusion &
PID7=$!
python scripts/train.py model=gpt &
PID8=$!

# Wait for all 4 to finish
wait $PID5 $PID6 $PID7 $PID8

echo "========================================"
echo " BATCH 2 DONE at: $(date)"
echo "========================================"

echo ""
echo "========================================"
echo " ALL TRAINING COMPLETED at: $(date)"
echo "========================================"
