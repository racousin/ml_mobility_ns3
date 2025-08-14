# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a machine learning project for trajectory generation and mobility simulation, integrating with NS-3 network simulator. The project uses PyTorch Lightning for training VAE-based models to generate realistic mobility patterns.

## Common Development Commands

### Setup and Installation
```bash
# Install dependencies using Poetry (preferred)
poetry install

# Or using pip
pip install -r requirements.txt
pip install -e .
```

### Data Preprocessing
```bash
python scripts/preprocess.py data.data_dir=data/netmob25
```

### Model Training
```bash
# Train VAE-LSTM model on GPU
python scripts/train.py model=vae_lstm training.epochs=100 accelerator=gpu devices=[0]

# Train with specific hyperparameters
python scripts/train.py model=vae_lstm model.hidden_dim=128 training.learning_rate=1e-3

# Run hyperparameter sweep
python scripts/train.py --config-path=configs/sweep --config-name=basic_grid --multirun

# Quick test with dummy model
python scripts/train.py model=dummy training.epochs=3 accelerator=cpu
```

### Evaluation
```bash
# List all experiments
python scripts/list_experiments.py

# Evaluate a specific experiment
python scripts/evaluate.py +experiment_id=vae_dense_2025-07-14_16-14-23

```

## Architecture

### Project Structure
- **ml_mobility_ns3/**: Core Python package
  - **models/**: Model architectures (VAE-LSTM, VAE-Dense, VAE-CNN, VAE-Attention)
  - **training/**: Training loops and Lightning modules
  - **evaluation/**: Evaluation metrics and visualization
  - **data/**: Data loading and preprocessing
  - **export/**: Model export utilities for C++ integration
  - **utils/**: Common utilities
  
- **configs/**: Hydra configuration files
  - **model/**: Model configurations
  - **training/**: Training hyperparameters
  - **data/**: Dataset configurations
  - **sweep/**: Hyperparameter sweep configs
  
- **scripts/**: Entry points for training, evaluation, and export
- **cpp_project/**: C++ templates for NS-3 integration (Jinja2 templates)
- **experiments/**: Saved model checkpoints and logs
- **notebooks/**: Jupyter notebooks for analysis

### Key Design Patterns

1. **Configuration Management**: Uses Hydra for hierarchical configuration management. All configs are in `configs/` with defaults in `config.yaml`.

2. **Model Architecture**: All models inherit from `BaseTrajectoryModel` and implement:
   - `forward()`: Forward pass for training
   - `generate()`: Generate new trajectories
   - VAE-based models with different encoders/decoders

3. **Training Pipeline**: Built on PyTorch Lightning with:
   - Automatic checkpointing
   - TensorBoard logging
   - Early stopping
   - Learning rate scheduling
   - Cyclical beta annealing for VAE loss

4. **Export System**: Models can be exported to C++ for NS-3 integration using Jinja2 templates in `cpp_project/`.

## Model Types

- **dummy**: Simple baseline model for testing
- **vae_lstm**: VAE with LSTM encoder/decoder (main model)
- **vae_dense**: VAE with fully connected layers
- **vae_cnn**: VAE with convolutional layers
- **vae_attention**: VAE with attention mechanism

## Data Format

Trajectories are stored as NumPy arrays with shape `(batch_size, sequence_length, features)` where features typically include x, y coordinates and potentially other mobility features. Data is normalized using scalers saved in `data/processed/`.

## NS-3 Integration

The project exports trained models to C++ code that can be integrated into NS-3 mobility models. The export process generates:
- Header files with model weights
- Implementation files with inference logic
- Example NS-3 scripts for testing