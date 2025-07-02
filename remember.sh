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
    --results-dir results/efficient_run_lstm_gpu \
    --epochs 150 \
    --batch-size 128 \
    --lr 1e-4 \
    --beta 0.001 \
    --hidden-dim 512 \
    --latent-dim 64 \
    --num-layers 2 \
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
    --results-dir results/complex_attention_gpu \
    --epochs 100 \
    --batch-size 64 \
    --lr 5e-5 \
    --beta 0.002 \
    --architecture attention \
    --hidden-dim 768 \
    --latent-dim 128 \
    --num-layers 4 \
    --n-heads 8 \
    --d-ff 2048 \
    --dropout 0.15 \
    --use-causal-mask \
    --pooling cls \
    --device cuda \
    --gpu-id 0

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

python ./train.py \
    --data-path ./preprocessing/vae_dataset.npz \
    --results-dir results/attention_beta_dropout_gpu \
    --epochs 75 \
    --batch-size 128 \
    --lr 1e-4 \
    --beta 0.01 \
    --architecture attention \
    --hidden-dim 512 \
    --latent-dim 64 \
    --num-layers 3 \
    --n-heads 8 \
    --dropout 0.2 \
    --device cuda \
    --gpu-id 0


cd cpp
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=/Users/raphaelcousin/libtorch ..
cmake --build . --config Release
./run_trajectory_gen

