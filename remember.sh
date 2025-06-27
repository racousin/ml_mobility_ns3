python ./train.py \
    --data-path ./preprocessing/vae_dataset.npz \
    --results-dir results/test_run \
    --epochs 3 \
    --batch-size 32 \
    --beta 0.001

python ./generate.py \
    --model-path results/test_run/best_model.pt \
    --data-path ./preprocessing/vae_dataset.npz \
    --output-path results/test_run/generated_trajectory.npy \
    --mode "car" \
    --length 250


python ./train.py \
    --data-path ./preprocessing/vae_dataset.npz \
    --results-dir results/efficient_run \
    --epochs 150 \
    --batch-size 128 \
    --lr 1e-4 \
    --beta 0.001 \
    --hidden-dim 512 \
    --latent-dim 64 \
    --num-layers 2 \
    --device cuda

python ./generate.py \
    --model-path results/efficient_run/best_model.pt \
    --data-path ./preprocessing/vae_dataset.npz \
    --output-path results/efficient_run/generated_walks.npy \
    --mode "walk" \
    --length 150 \
    --n-samples 5 \
    --device cuda