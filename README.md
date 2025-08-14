# ml_mobility_ns3

Trajectory generation models for mobility simulation.

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Part 1: Preprocess, Train, Evaluate Trajectory Generation

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

### Generated Trajectories Example

![Generated Trajectories](./2500gen.png)

*Example of generated trajectories showing the model's capability to produce realistic mobility patterns.*

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

## Part 2: Use Trajectory Generation Model in NS-3

### NS-3 Integration

#### Prerequisites

1. **Install NS-3:**
```bash
wget https://www.nsnam.org/releases/ns-allinone-3.45.tar.bz2
tar -jxvf ns-allinone-3.45.tar.bz2
cd ns-allinone-3.45/ns-3.45
./ns3 configure --enable-examples --enable-tests
./ns3 build
```

2. **Download and Install PyTorch:**
```bash
# Download PyTorch C++ (LibTorch) - CPU version
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.1.0+cpu.zip
export TORCH_ROOT=$(pwd)/libtorch
export CMAKE_PREFIX_PATH=$TORCH_ROOT:$CMAKE_PREFIX_PATH
```

#### Export and Integration

```bash
# 1. Export your trained model
poetry run python scripts/export.py +experiment_id=vae_lstm_2025-07-16_12-30-52
```

This creates `cpp_export/ns3.45_vae_lstm_2025-07-16_12-30-52/` containing:
- `netmob25-mobility-model.h` - NS-3 mobility model header
- `netmob25-mobility-model.cc` - NS-3 mobility model implementation  
- `netmob25-mobility-example.cc` - Complete simulation example
- `model.p` - ML model file for PyTorch inference

```bash
# 2. Copy files to NS-3
cd cpp_export/ns3.45_vae_lstm_2025-07-16_12-30-52
cp netmob25-mobility-model.* ns-3.45/src/mobility/model/
cp netmob25-mobility-example.cc ns-3.45/scratch/
cp model.p ns-3.45/

# 3. Build NS-3 with PyTorch support
cd ns-3.45
CMAKE_PREFIX_PATH=$TORCH_ROOT ./ns3 configure --enable-examples
CMAKE_PREFIX_PATH=$TORCH_ROOT ./ns3 build

# 4. Run the example with ML trajectory generation
./ns3 run "scratch/netmob25-mobility-example --useMLGeneration=true --modelPath=model.p --nNodes=4 --simTime=30"
```

### Expected Output

The simulation will show node positions and generate an animation file:

```
=== Netmob25 Mobility Example ===
Nodes: 4
Simulation time: 30 seconds  
ML Generation: Enabled
Model path: model.p
Animation file: netmob25-animation.xml
Experiment: vae_lstm_2025-07-16_12-30-52
==============================

Time: 5s - Node positions:
  Node 0: (245.3, 178.9)
  Node 1: (312.7, 89.2)  
  Node 2: (156.8, 234.1)
  Node 3: (89.4, 167.5)

Simulation completed!
Generated animation: netmob25-animation.xml
```

You can visualize the mobility patterns using NetAnim with the generated XML file.