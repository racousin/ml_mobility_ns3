#!/usr/bin/env python
"""Generate C++ project from trained VAE model."""

import argparse
import json
import pickle
import shutil
from pathlib import Path
import torch
import numpy as np
import logging
from jinja2 import Template

# Add project root to path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from ml_mobility_ns3.models.vae import ConditionalTrajectoryVAE
from ml_mobility_ns3.utils.model_utils import convert_to_torchscript
from ml_mobility_ns3.utils.model_utils import load_model_from_checkpoint

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TorchScriptModel(torch.nn.Module):
    def __init__(self, vae_model, fixed_n_samples=1):
        super().__init__()
        self.vae = vae_model
        self.fixed_n_samples = fixed_n_samples
        
    def forward(self, transport_mode: torch.Tensor, trip_length: torch.Tensor):
        """Generate trajectories - TorchScript compatible version with fixed n_samples."""
        device = transport_mode.device
        conditions = self.vae.get_conditions(transport_mode, trip_length)
        z = torch.randn(self.fixed_n_samples, self.vae.latent_dim, device=device)
        trajectories = self.vae.decode(z, conditions)
        return trajectories


def convert_model_to_torchscript(model_path: Path, output_path: Path, device: str = 'cpu'):
    """Convert the trained VAE model to TorchScript format."""
    logger.info(f"Loading model from {model_path}")
    
    # Load the trained model
    model, config = load_model_from_checkpoint(model_path, device)
    model.eval()
    
    # Create TorchScript wrapper
    script_model = TorchScriptModel(model)
    script_model.eval()
    
    # Create example inputs for tracing
    transport_mode = torch.tensor([0], dtype=torch.long, device=device)  # car mode
    trip_length = torch.tensor([150], dtype=torch.long, device=device)
    example_inputs = (transport_mode, trip_length)
    
    logger.info("Converting model to TorchScript...")
    convert_to_torchscript(script_model, example_inputs, output_path, method='trace')
    
    return config


def extract_metadata(preprocessing_dir: Path, config: dict) -> dict:
    """Extract metadata from preprocessing directory."""
    logger.info(f"Extracting metadata from {preprocessing_dir}")
    
    # Load metadata.pkl
    with open(preprocessing_dir / "metadata.pkl", 'rb') as f:
        metadata = pickle.load(f)
    
    # Create simplified metadata for C++
    cpp_metadata = {
        "transport_modes": metadata['transport_modes'],
        "model_config": config,
        "sequence_length": config['sequence_length'],
        "input_dim": config['input_dim'],
        "latent_dim": config['latent_dim']
    }
    
    return cpp_metadata


def extract_scalers(preprocessing_dir: Path) -> dict:
    """Extract and convert scalers to C++ compatible format."""
    logger.info(f"Extracting scalers from {preprocessing_dir}")
    
    # Load scalers.pkl
    with open(preprocessing_dir / "scalers.pkl", 'rb') as f:
        scalers = pickle.load(f)
    
    # Convert sklearn scalers to simple mean/scale format
    trajectory_scaler = scalers['trajectory']
    
    cpp_scalers = {
        "trajectory": {
            "mean": trajectory_scaler.mean_.tolist(),
            "scale": trajectory_scaler.scale_.tolist()
        }
    }
    
    return cpp_scalers


def render_template(template_path: Path, output_path: Path, **kwargs):
    """Render a Jinja2 template with given variables."""
    logger.info(f"Rendering template {template_path.name} -> {output_path.name}")
    
    with open(template_path, 'r') as f:
        template_content = f.read()
    
    template = Template(template_content)
    rendered = template.render(**kwargs)
    
    with open(output_path, 'w') as f:
        f.write(rendered)


def generate_cpp_project(experiment_path: Path, cpp_project_templates: Path):
    """Generate complete C++ project from experiment."""
    
    # Validate input paths
    if not experiment_path.exists():
        raise FileNotFoundError(f"Experiment path not found: {experiment_path}")
    
    model_path = experiment_path / "best_model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Find preprocessing directory (assuming it's in the parent directory)
    preprocessing_dir = experiment_path.parent.parent / "preprocessing"
    if not preprocessing_dir.exists():
        raise FileNotFoundError(f"Preprocessing directory not found: {preprocessing_dir}")
    
    # Create output directory
    experiment_name = experiment_path.name
    output_dir = Path("cpp") / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Generating C++ project for experiment: {experiment_name}")
    logger.info(f"Output directory: {output_dir}")
    
    # Step 1: Convert model to TorchScript
    traced_model_path = output_dir / "traced_vae_model.pt"
    config = convert_model_to_torchscript(model_path, traced_model_path)
    
    # Step 2: Extract and save metadata
    metadata = extract_metadata(preprocessing_dir, config)
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Step 3: Extract and save scalers
    scalers = extract_scalers(preprocessing_dir)
    scalers_path = output_dir / "scalers.json"
    with open(scalers_path, 'w') as f:
        json.dump(scalers, f, indent=2)
    
    # Step 4: Copy json.hpp header
    json_header_src = cpp_project_templates / "json.hpp"
    json_header_dst = output_dir / "json.hpp"
    if json_header_src.exists():
        shutil.copy2(json_header_src, json_header_dst)
    else:
        logger.warning(f"json.hpp not found at {json_header_src}, you'll need to download it manually")
    
    # Step 5: Generate C++ files from templates
    template_vars = {
        "experiment_name": experiment_name,
        "project_name": f"{experiment_name.replace('_', '').title()}Cpp",
        "transport_modes": metadata["transport_modes"],
        "sequence_length": metadata["sequence_length"],
        "input_dim": metadata["input_dim"],
        "latent_dim": metadata["latent_dim"]
    }
    
    # Render all templates
    templates = [
        ("CMakeLists.txt.jinja", "CMakeLists.txt"),
        ("main.cc.jinja", "main.cc"),
        ("trajectory_generator.h.jinja", "trajectory_generator.h"),
        ("trajectory_generator.cc.jinja", "trajectory_generator.cc"),
        ("README.md.jinja", "README.md")
    ]
    
    for template_name, output_name in templates:
        template_path = cpp_project_templates / template_name
        output_path = output_dir / output_name
        
        if template_path.exists():
            render_template(template_path, output_path, **template_vars)
        else:
            logger.warning(f"Template not found: {template_path}")
    
    # Step 6: Create build script
    build_script_content = f"""#!/bin/bash
# Build script for {experiment_name} C++ project

set -e

echo "Building {experiment_name} C++ trajectory generator..."

# Create build directory
mkdir -p build
cd build

# Configure with CMake
cmake ..

# Build
make -j$(nproc)

echo "Build complete! Executable: ./run_trajectory_gen"
echo "Run with: ./build/run_trajectory_gen"
"""
    
    build_script_path = output_dir / "build.sh"
    with open(build_script_path, 'w') as f:
        f.write(build_script_content)
    build_script_path.chmod(0o755)  # Make executable
    
    logger.info(f"C++ project generated successfully in: {output_dir}")
    logger.info("Files created:")
    for file_path in sorted(output_dir.rglob("*")):
        if file_path.is_file():
            logger.info(f"  {file_path.relative_to(output_dir)}")
    
    logger.info(f"\nTo build and run:")
    logger.info(f"  cd {output_dir}")
    logger.info(f"  ./build.sh")
    logger.info(f"  ./build/run_trajectory_gen")


def main():
    parser = argparse.ArgumentParser(description="Generate C++ project from trained VAE model")
    parser.add_argument("experiment_path", type=Path, 
                       help="Path to experiment directory (e.g., results/efficient_run)")
    parser.add_argument("--cpp-templates", type=Path, default="cpp_project",
                       help="Path to C++ project templates directory")
    
    args = parser.parse_args()
    
    try:
        generate_cpp_project(args.experiment_path, args.cpp_templates)
    except Exception as e:
        logger.error(f"Failed to generate C++ project: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())