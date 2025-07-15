# ml_mobility_ns3

Trajectory generation models for mobility simulation.

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Usage

### Data Preprocessing

```bash
python scripts/preprocess.py data.data_dir=data/netmob25
```

### Training

```bash
python scripts/train.py model=vae_lstm training.epochs=100
```

### List Experiments
```bash
python scripts/list_experiments.py
```

### Evaluation

```bash
python scripts/evaluate.py +experiment_id=vae_dense_2025-07-14_16-14-23
```


### Configuration

All configurations are managed through Hydra. Default configs are in `configs/`.

### Model Selection

```bash
python scripts/train.py model=dummy training.epochs=3 accelerator=cpu # Use dummy model
python scripts/train.py model=vae_lstm  accelerator=gpu devices=[3] device=cuda # Use VAE-LSTM model

python scripts/train.py --config-path=configs/sweep --config-name=basic_grid --multirun
```

### Hyperparameter Tuning

```bash
python scripts/train.py model.hidden_dim=128 training.learning_rate=1e-3
```

## Project Structure

- `ml_mobility_ns3/`: Core package
  - `data/`: Data preprocessing and datasets
  - `models/`: Model architectures
  - `training/`: PyTorch Lightning modules
  - `evaluation/`: Evaluation metrics and scripts
  - `export/`: Model export utilities
- `configs/`: Hydra configuration files
- `scripts/`: Entry point scripts

### Export to C++

```bash
python scripts/export.py export.output_dir=path/to/cpp/output
```

