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

### NS-3 Installation

First, check your OS version and install dependencies:

```bash
cat /etc/os-release
sudo apt install git g++ clang python3 python3-pip cmake ninja-build tcpdump wireshark
python3 -m pip install --user cppyy==3.1.2
```

Download and build NS-3:

```bash
tar -jxvf ns-3.45.tar.bz2
cd ns-3.45
./ns3 configure --enable-examples --enable-tests
./ns3 build
./test.py
./ns3 run hello-simulator
```

### Export Model for NS-3

Export your trained model for integration with NS-3:

```bash
# Export a specific experiment
poetry run python scripts/export.py +experiment_id=your_experiment_id

# Example using the dummy model
poetry run python scripts/export.py +experiment_id=dummy_2025-07-13_19-17-59
```

This will:
- Convert the PyTorch model to TorchScript format
- Generate C++ project files with LibTorch integration
- Create a complete build system with CMake

The exported files will be in the `cpp_export/` directory.

### Build and Test C++ Trajectory Generator

After exporting, build and test the C++ trajectory generator:

```bash
cd cpp_export

# Build the project (requires LibTorch/PyTorch)
./build.sh

# Test the trajectory generator
./build/run_trajectory_gen
```

#### Manual Build (if build.sh fails)

If the automatic build script fails, you can build manually:

```bash
cd cpp_export
mkdir -p build
cd build

# Find PyTorch cmake path
CMAKE_PATH=$(poetry run python3 -c "import torch; print(torch.utils.cmake_prefix_path)")

# Configure and build
cmake .. -DCMAKE_PREFIX_PATH="$CMAKE_PATH"
make -j4

# Run the generator
./run_trajectory_gen
```

### System Requirements for C++ Integration

- **CMake** 3.10 or higher
- **C++17** compatible compiler
- **PyTorch/LibTorch** (automatically detected from Python installation)

On macOS:
```bash
# Install XCode command line tools
xcode-select --install

# Install CMake (via Homebrew)
brew install cmake
```

### Installation of the netmob25 Mobility Model in NS-3.45

Follow these steps to integrate the netmob25 mobility model:

1. **Copy model files:**
   ```bash
   cp netmob25-mobility-model.h ns-3.45/src/mobility/model/
   cp netmob25-mobility-model.cc ns-3.45/src/mobility/model/
   ```

2. **Update CMakeLists.txt:**
   Add `netmob25-mobility-model.h` and `netmob25-mobility-model.cc` to the CMakeLists.txt file in `ns-3.45/src/mobility/`

3. **Copy example file:**
   ```bash
   cp netmob25-mobility-example.cc ns-3.45/scratch/
   ```

4. **Recompile NS-3:**
   ```bash
   cd ns-3.45
   ./ns3 configure --enable-examples --enable-tests
   ./ns3 build
   ```

5. **Test the mobility model:**
   ```bash
   ./ns3 run scratch/netmob25-mobility-example.cc
   ```

## NS-3 Integration with ML Trajectory Generator

### Quick Test of C++ Trajectory Generator

Before integrating with NS-3, test the trajectory generator:

```bash
cd cpp_export

# Build both the standalone test and NS-3 simulation test
./build.sh

# Test the basic trajectory generator
./build/run_trajectory_gen

# Test the NS-3 mobility simulation
./build/ns3_trajectory_test
```

### Full NS-3 Integration

#### 1. Install Trajectory Generator into NS-3

```bash
cd cpp_export

# Install trajectory generator and example into NS-3
./install_to_ns3.sh ../ns-3.45
```

This script will:
- Copy `trajectory_generator.{h,cc}` to NS-3's mobility module
- Copy `model.pt` to NS-3's scratch directory  
- Copy the complete NS-3 example to scratch
- Update NS-3's CMakeLists.txt to include trajectory generator
- Create configuration helper for PyTorch paths

#### 2. Configure NS-3 with PyTorch Support

```bash
cd ns-3.45

# Get PyTorch configuration
poetry run python3 scratch/trajectory_pytorch_config.py

# Configure NS-3 with PyTorch (use the command shown by the script above)
./ns3 configure --enable-examples --build-profile=optimized \
  CPPFLAGS="-I/path/to/pytorch/include" \
  LINKFLAGS="-L/path/to/pytorch/lib -ltorch -ltorch_library -lc10"
```

#### 3. Build and Run NS-3 Example

```bash
# Build NS-3
./ns3 build

# Run the ML trajectory mobility example
./ns3 run scratch/ns3-trajectory-mobility-example

# Optional: Run with parameters
./ns3 run "scratch/ns3-trajectory-mobility-example --nNodes=10 --time=200"
```

#### 4. NS-3 Example Features

The NS-3 example (`ns3-trajectory-mobility-example.cc`) demonstrates:

- **Custom Mobility Model**: `MLTrajectoryMobilityModel` that uses ML-generated trajectories
- **WiFi Network**: Ad-hoc network with realistic mobility patterns
- **UDP Applications**: Echo client/server to test connectivity
- **Animation Output**: NetAnim-compatible XML for visualization
- **Tracing**: ASCII and PCAP traces for analysis

#### 5. Customizing for Your Models

To use your own trained models:

1. Export your model:
```bash
poetry run python scripts/export.py +experiment_id=your_model_id
```

2. Copy the new `model.pt` to NS-3:
```bash
cp cpp_export/model.pt ns-3.45/scratch/
```

3. Modify trajectory scaling in the example if needed:
```cpp
// In ns3-trajectory-mobility-example.cc
m_currentPosition = Vector (
    m_trajectory[m_currentStep][0].item<float>() * 1000,  // Adjust scale factor
    m_trajectory[m_currentStep][1].item<float>() * 1000,  // Adjust scale factor
    0.0
);
```

#### 6. Integration with Your Own NS-3 Applications

To integrate the trajectory generator with your own NS-3 code:

1. Include the trajectory generator:
```cpp
#include "trajectory_generator.h"
```

2. Initialize and generate trajectories:
```cpp
TrajectoryGenerator generator("model.pt");
auto trajectories = generator.generate(numNodes, sequenceLength);
```

3. Apply to nodes using `MLTrajectoryMobilityModel` or extract waypoints for other mobility models

### Troubleshooting NS-3 Integration

**Build Issues:**
- Ensure PyTorch is properly installed and paths are correct
- Check that `trajectory_generator.{h,cc}` are in `src/mobility/model/`
- Verify CMakeLists.txt was updated correctly

**Runtime Issues:**
- Make sure `model.pt` is in the correct directory (usually `scratch/`)
- Check that the model was exported correctly
- Ensure trajectory scaling is appropriate for your simulation area

**Performance:**
- Use CPU-optimized models for faster simulation
- Consider reducing trajectory sequence length for large-scale simulations
- Profile memory usage with many nodes