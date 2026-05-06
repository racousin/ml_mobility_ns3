#!/bin/bash
tmux kill-session -t ml_ns3 2>/dev/null
tmux new-session -d -s ml_ns3 -n vae_lstm
tmux send-keys -t ml_ns3:vae_lstm "source ~/.bashrc && conda activate ml_ns3 && poetry run python scripts/train.py --config-path=../experiments/vae_lstm_2026-05-03_15-37-04 --config-name=config +training.resume_from_checkpoint=$(pwd)/experiments/vae_lstm_2026-05-03_15-37-04/checkpoints/last.ckpt" C-m

tmux new-window -t ml_ns3 -n vae_cnn
tmux send-keys -t ml_ns3:vae_cnn "source ~/.bashrc && conda activate ml_ns3 && poetry run python scripts/train.py --config-path=../experiments/vae_cnn_2026-05-03_15-37-45 --config-name=config +training.resume_from_checkpoint=$(pwd)/experiments/vae_cnn_2026-05-03_15-37-45/checkpoints/last.ckpt" C-m

tmux new-window -t ml_ns3 -n vae_dense
tmux send-keys -t ml_ns3:vae_dense "source ~/.bashrc && conda activate ml_ns3 && poetry run python scripts/train.py --config-path=../experiments/vae_dense_2026-05-03_15-38-28 --config-name=config +training.resume_from_checkpoint=$(pwd)/experiments/vae_dense_2026-05-03_15-38-28/checkpoints/last.ckpt" C-m

tmux new-window -t ml_ns3 -n vq_vae
tmux send-keys -t ml_ns3:vq_vae "source ~/.bashrc && conda activate ml_ns3 && poetry run python scripts/train.py --config-path=../experiments/vq_vae_2026-05-04_04-44-49 --config-name=config +training.resume_from_checkpoint=$(pwd)/experiments/vq_vae_2026-05-04_04-44-49/checkpoints/last.ckpt" C-m
