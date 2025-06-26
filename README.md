# Trajectory Generator

A simple VAE-based trajectory generation package for the NetMob25 dataset.

## Installation

```bash
# Install poetry if not already installed
pip install poetry

# Install dependencies
poetry install

# Activate the virtual environment
poetry shell
```

## Quick Start with Fake Data

Since the full NetMob25 dataset requires registration, we provide a script to generate realistic fake data:

```bash
# Generate fake dataset
python generate_fake_data.py

# Test with fake data
python test_fake_data.py

# Train model on fake data
python train.py --data-dir data/fake_netmob25 --epochs 20 --n-samples 500
```

## Using Real NetMob25 Data

```python
from pathlib import Path
from ml_mobility_ns3 import (
    NetMob25Loader,
    TrajectoryPreprocessor,
    TrajectoryVAE,
    VAETrainer,
    TrajectoryGenerator
)

# Load data
loader = NetMob25Loader(Path("path/to/netmob25/data"))
trajectories = loader.sample_trajectories(n_samples=1000)

# Preprocess
preprocessor = TrajectoryPreprocessor(sequence_length=50)
preprocessor.fit(trajectories[:800])
train_data = preprocessor.transform(trajectories[:800])
val_data = preprocessor.transform(trajectories[800:])

# Create and train model
model = TrajectoryVAE(
    input_dim=4,
    sequence_length=50,
    hidden_dim=128,
    latent_dim=32
)

trainer = VAETrainer(model, device='cuda')  # or 'cpu'
trainer.fit(
    train_data,
    val_data,
    epochs=50,
    batch_size=32,
    save_path=Path("checkpoints/best_model.pt")
)

# Generate new trajectories
generator = TrajectoryGenerator(model, preprocessor)
new_trajectories = generator.generate(n_samples=10)
```

## Project Structure

```
ml_mobility_ns3/
├── data/           # Data loading and preprocessing
├── models/         # VAE model definition
├── training/       # Training utilities
├── inference/      # Generation and inference
└── utils/          # Visualization utilities
```

## Features

- Simple data loader for NetMob25 dataset
- Trajectory preprocessing with normalization
- VAE model for trajectory generation
- Training with GPU support
- Trajectory generation and interpolation
- Visualization utilities

## Fake Data Generation

The `generate_fake_data.py` script creates a realistic dataset mimicking NetMob25:
- 10 users with demographic profiles
- 1000 trips with realistic patterns (commutes, shopping, leisure)
- GPS traces for each trip
- Paris region coordinates
- Realistic transport modes and speeds

## Usage Examples

See the notebooks in `notebooks/` for detailed examples:
- `01_data_exploration.ipynb`: Data loading and visualization
- `02_model_experiments.ipynb`: Model training and generation