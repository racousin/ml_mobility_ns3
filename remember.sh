python ./train.py \
    --data-path ./preprocessing/vae_dataset.npz \
    --results-dir results/test_run_lstm_gpu \
    --epochs 3 \
    --batch-size 32 \
    --beta 0.001 \
    --device cuda \
    --gpu-id 0


python ./train.py \
    --data-path ./preprocessing/vae_dataset.npz \
    --results-dir results/test_run_attention_gpu \
    --epochs 5 \
    --batch-size 32 \
    --beta 0.005 \
    --architecture attention \
    --hidden-dim 128 \
    --latent-dim 16 \
    --num-layers 2 \
    --n-heads 4 \
    --device cuda \
    --gpu-id 2


python ./train.py \
    --data-path ./preprocessing/vae_dataset.npz \
    --results-dir results/large_lstm_gpu \
    --epochs 200 \
    --batch-size 256 \
    --lr 1e-4 \
    --beta 0.0005 \
    --hidden-dim 1024 \
    --latent-dim 256 \
    --num-layers 3 \
    --device cuda \
    --gpu-id 0

# Run experiments on GPU 2
chmod +x run_experiments.sh
./run_experiments.sh 2 ./preprocessing/vae_dataset.npz results/batch_experiments

# Dry run to see what would be executed
./run_experiments.sh 2 ./preprocessing/vae_dataset.npz results/test_run true

# Generate full report
python analyze_results.py --results-dir results/experiments --generate-report

# Plot specific experiments
python analyze_results.py --results-dir results/experiments \
    --experiments lstm_vae_small attention_vae_medium \
    --plot-curves --plot-lr

# Analyze early stopping
python analyze_results.py --results-dir results/experiments --early-stopping

# Generate sample trajectories
python analyze_results.py --results-dir results/experiments \
    --sample-trajectories results/experiments/lstm_vae_medium/best_model.pt

# Compare generation from multiple models
python analyze_results.py --results-dir results/experiments \
    --compare-generation results/experiments/*/best_model.pt



cd cpp
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=/Users/raphaelcousin/libtorch ..
cmake --build . --config Release
./run_trajectory_gen

