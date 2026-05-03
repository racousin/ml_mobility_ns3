#!/bin/bash
# Script to train all models in 2 batches of 4
# Each model runs in its own tmux window for easy log monitoring
# Usage: bash scripts/train_all.sh
#
# Windows layout:
#   0:diffusion  1:dummy  2:gpt  3:vae_attention
#   4:vae_cnn    5:vae_dense  6:vae_lstm  7:vq_vae

cd /network-volume/ns3/ml_mobility_ns3
SESSION="tml_ns3"

echo "========================================"
echo " Batch Training - 4 models at a time"
echo " Each model in its own tmux window"
echo " Started at: $(date)"
echo "========================================"

# --- BATCH 1: vae_attention, vae_lstm, gpt, vae_dense ---
echo ""
echo " BATCH 1: vae_attention, vae_lstm, gpt, vae_dense"
echo "========================================"

tmux send-keys -t $SESSION:3 "cd /network-volume/ns3/ml_mobility_ns3 && python scripts/train.py model=vae_attention" Enter
sleep 2
tmux send-keys -t $SESSION:6 "cd /network-volume/ns3/ml_mobility_ns3 && python scripts/train.py model=vae_lstm" Enter
sleep 2
tmux send-keys -t $SESSION:2 "cd /network-volume/ns3/ml_mobility_ns3 && python scripts/train.py model=gpt training=gpt" Enter
sleep 2
tmux send-keys -t $SESSION:5 "cd /network-volume/ns3/ml_mobility_ns3 && python scripts/train.py model=vae_dense" Enter

echo " All batch 1 models launched. Waiting for completion..."

# Wait for all 4 to finish by checking if python is still running in each window
while true; do
    RUNNING=0
    for win in 3 6 2 5; do
        PANE_PID=$(tmux list-panes -t $SESSION:$win -F '#{pane_pid}')
        if pgrep -P "$PANE_PID" python > /dev/null 2>&1; then
            RUNNING=$((RUNNING + 1))
        fi
    done
    if [ $RUNNING -eq 0 ]; then
        break
    fi
    echo "  [$(date +%H:%M:%S)] $RUNNING model(s) still training..."
    sleep 60
done

echo "========================================"
echo " BATCH 1 DONE at: $(date)"
echo "========================================"
sleep 10

# --- BATCH 2: dummy, vae_cnn, vq_vae, diffusion ---
echo ""
echo " BATCH 2: dummy, vae_cnn, vq_vae, diffusion"
echo "========================================"

tmux send-keys -t $SESSION:1 "cd /network-volume/ns3/ml_mobility_ns3 && python scripts/train.py model=dummy" Enter
sleep 2
tmux send-keys -t $SESSION:4 "cd /network-volume/ns3/ml_mobility_ns3 && python scripts/train.py model=vae_cnn" Enter
sleep 2
tmux send-keys -t $SESSION:7 "cd /network-volume/ns3/ml_mobility_ns3 && python scripts/train.py model=vq_vae" Enter
sleep 2
tmux send-keys -t $SESSION:0 "cd /network-volume/ns3/ml_mobility_ns3 && python scripts/train.py model=diffusion training=diffusion" Enter

echo " All batch 2 models launched. Waiting for completion..."

while true; do
    RUNNING=0
    for win in 1 4 7 0; do
        PANE_PID=$(tmux list-panes -t $SESSION:$win -F '#{pane_pid}')
        if pgrep -P "$PANE_PID" python > /dev/null 2>&1; then
            RUNNING=$((RUNNING + 1))
        fi
    done
    if [ $RUNNING -eq 0 ]; then
        break
    fi
    echo "  [$(date +%H:%M:%S)] $RUNNING model(s) still training..."
    sleep 60
done

echo "========================================"
echo " ALL TRAINING COMPLETED at: $(date)"
echo "========================================"
