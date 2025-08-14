import torch
import torch.nn as nn
from pathlib import Path
import json
import logging
from jinja2 import Environment, FileSystemLoader
import shutil
from typing import Dict, Any
import numpy as np

logger = logging.getLogger(__name__)


class CppExporter:
    def __init__(self, config):
        self.config = config
        self.output_dir = Path(config.export.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"CppExporter initialized with output_dir: {self.output_dir}")
        
    def export_model(self, checkpoint, metadata: Dict[str, Any], experiment_name: str = "trajectory_model"):
        logger.info(f"Exporting model to C++ in {self.output_dir}")
        
        # Create experiment-specific directory in cpp_ns3_export
        experiment_dir = self.output_dir / experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Recreate model architecture
        model = self._recreate_model(checkpoint)
        
        # Convert to TorchScript and save in experiment directory
        if self.config.export.compile_torchscript:
            self._export_torchscript(model, experiment_dir)
        
        # Save metadata in experiment directory
        self._save_metadata(metadata, experiment_name, experiment_dir)
        
        # Generate NS-3 files in experiment directory
        self._generate_ns3_files(experiment_name, experiment_dir)
        
        logger.info(f"Export complete! NS-3 files created in {experiment_dir}")
        
    def _recreate_model(self, checkpoint):
        """Recreate model from checkpoint"""
        if isinstance(checkpoint, dict):
            # Extract hyperparameters
            hparams = checkpoint.get('hyper_parameters', {})
            model_config = hparams.get('model', {})
            
            # Determine model type
            model_type = model_config.get('type', 'dummy')
            
            # Import the appropriate model class
            if model_type == 'dummy':
                from ml_mobility_ns3.models.dummy import DummyModel
                # Extract parameters from config
                model = DummyModel(
                    input_dim=model_config.get('input_dim', 3),
                    sequence_length=model_config.get('sequence_length', 2000),
                    num_transport_modes=model_config.get('num_transport_modes', 5),
                    latent_dim=model_config.get('latent_dim', 16)
                )
            elif model_type == 'vae_lstm':
                from ml_mobility_ns3.models.vae_lstm import VAELSTM
                model = VAELSTM(**model_config)
            elif model_type == 'vae_dense':
                from ml_mobility_ns3.models.vae_dense import VAEDense
                model = VAEDense(**model_config)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Load state dict
            if 'state_dict' in checkpoint:
                # Remove 'model.' prefix from keys if present
                state_dict = {}
                for k, v in checkpoint['state_dict'].items():
                    if k.startswith('model.'):
                        state_dict[k[6:]] = v
                    else:
                        state_dict[k] = v
                model.load_state_dict(state_dict, strict=False)
            
            return model
        else:
            # Checkpoint is already a model
            return checkpoint
        
    def _export_torchscript(self, model, experiment_dir=None):
        model.eval()
        
        # Create a wrapper for models that return dictionaries
        class ModelWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                
            def forward(self, x, transport_mode=None, length=None, mask=None):
                # Call the original model
                output = self.model(x, transport_mode, length, mask)
                # Return only the reconstruction
                if isinstance(output, dict):
                    return output['recon']
                return output
        
        wrapped_model = ModelWrapper(model)
        wrapped_model.eval()
        
        # Create example inputs
        batch_size = 1
        seq_len = min(100, model.sequence_length if hasattr(model, 'sequence_length') else 100)
        input_dim = model.input_dim if hasattr(model, 'input_dim') else 3
        
        example_x = torch.randn(batch_size, seq_len, input_dim)
        example_mode = torch.zeros(batch_size, dtype=torch.long)
        example_length = torch.tensor([seq_len] * batch_size, dtype=torch.long)
        
        # Try to trace the wrapped model
        try:
            with torch.no_grad():
                traced_model = torch.jit.trace(wrapped_model, 
                                              (example_x, example_mode, example_length))
                
            # Save traced model
            save_dir = experiment_dir if experiment_dir else self.output_dir
            traced_path = save_dir / 'model.pt'
            traced_model.save(str(traced_path))
            logger.info(f"Saved TorchScript model to {traced_path}")
            
        except Exception as e:
            logger.warning(f"Could not trace model: {e}")
            # Save the model state dict as fallback
            save_dir = experiment_dir if experiment_dir else self.output_dir
            torch.save(model.state_dict(), save_dir / 'model_state.pt')
            torch.save({'model_config': model.config if hasattr(model, 'config') else {}}, 
                      save_dir / 'model_config.pt')
            logger.info("Saved model state dict and config instead of TorchScript")
        
    def _create_example_input(self, model):
        """Create example input for model tracing"""
        # Get input dimensions from model config if available
        if hasattr(model, 'input_dim'):
            input_dim = model.input_dim
        else:
            input_dim = 3  # Default
        
        if hasattr(model, 'sequence_length'):
            seq_len = min(model.sequence_length, 100)  # Use smaller for testing
        else:
            seq_len = 100
            
        # Default input shape (batch_size=1, seq_len, features)
        return torch.randn(1, seq_len, input_dim)
    
    def _save_metadata(self, metadata: Dict[str, Any], experiment_name: str, experiment_dir=None):
        """Save metadata as JSON"""
        # Convert numpy arrays to lists for JSON serialization
        json_metadata = {}
        for k, v in metadata.items():
            if isinstance(v, np.ndarray):
                json_metadata[k] = v.tolist()
            else:
                json_metadata[k] = v
        
        json_metadata['experiment_name'] = experiment_name
        
        save_dir = experiment_dir if experiment_dir else self.output_dir
        metadata_path = save_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(json_metadata, f, indent=2)
        logger.info(f"Saved metadata to {metadata_path}")
        
    def _generate_ns3_files(self, experiment_name: str, experiment_dir: Path):
        """Generate NS-3 mobility model files from templates"""
        template_dir = Path('cpp_project')
        
        if not template_dir.exists():
            logger.error(f"Template directory {template_dir} not found")
            return
            
        # Setup Jinja2 environment
        env = Environment(loader=FileSystemLoader(str(template_dir)))
        
        # Template context
        context = {
            'project_name': 'netmob25_mobility',
            'experiment_name': experiment_name,
            'model_path': 'model.p',
        }
        
        # Generate NS-3 specific files from templates
        ns3_templates = [
            'netmob25-mobility-model.h.jinja',
            'netmob25-mobility-model.cc.jinja', 
            'netmob25-mobility-example.cc.jinja',
            'trajectory_generator.h.jinja',
            'trajectory_generator.cc.jinja',
            'CMakeLists.txt.jinja'
        ]
        
        for template_name in ns3_templates:
            template_file = template_dir / template_name
            if template_file.exists():
                template = env.get_template(template_name)
                output_content = template.render(**context)
                
                # Remove .jinja extension and save in experiment directory
                output_file = experiment_dir / template_file.stem
                with open(output_file, 'w') as f:
                    f.write(output_content)
                logger.info(f"Generated NS-3 file: {output_file}")
            else:
                logger.warning(f"Template {template_name} not found")
                
        # Copy non-template files (like json.hpp)
        static_files = ['json.hpp']
        for file_name in static_files:
            src_file = template_dir / file_name
            if src_file.exists():
                dst_file = experiment_dir / file_name
                shutil.copy(src_file, dst_file)
                logger.info(f"Copied static file: {dst_file}")
            else:
                logger.warning(f"Static file {file_name} not found")
        
        # Copy model.pt as model.p in experiment directory
        model_file = experiment_dir / 'model.pt'
        if model_file.exists():
            model_p_file = experiment_dir / 'model.p'
            shutil.copy(model_file, model_p_file)
            logger.info(f"Created {model_p_file} for NS-3 compatibility")
            
    
    def integrate_with_ns3(self, ns3_path: str):
        """Integrate the exported model with NS-3"""
        logger.info(f"Integrating with NS-3 at {ns3_path}")
        
        ns3_path = Path(ns3_path)
        if not ns3_path.exists():
            raise FileNotFoundError(f"NS-3 directory not found: {ns3_path}")
        
        if not (ns3_path / "ns3").exists():
            raise ValueError(f"Invalid NS-3 directory (ns3 script not found): {ns3_path}")
        
        # Use the export_to_ns3.sh script from cpp_ns3_export directory
        export_script = Path("cpp_ns3_export") / "export_to_ns3.sh"
        if not export_script.exists():
            logger.warning("export_to_ns3.sh script not found, creating basic integration")
            self._manual_ns3_integration(ns3_path)
        else:
            # Run the export script
            import subprocess
            try:
                result = subprocess.run([str(export_script), str(ns3_path)], 
                                      capture_output=True, text=True, check=True)
                logger.info("NS-3 integration successful")
                logger.info(result.stdout)
            except subprocess.CalledProcessError as e:
                logger.error(f"NS-3 integration failed: {e.stderr}")
                raise
        
        # Copy model files to NS-3
        model_file = self.output_dir / "model.pt"
        metadata_file = self.output_dir / "metadata.json"
        
        if model_file.exists():
            shutil.copy(model_file, ns3_path)
            logger.info(f"Copied model file to {ns3_path}")
        
        if metadata_file.exists():
            shutil.copy(metadata_file, ns3_path)
            logger.info(f"Copied metadata file to {ns3_path}")
        
        # Create NS-3 specific build and test script
        self._create_ns3_test_script(ns3_path)
        
        logger.info("NS-3 integration complete!")
        logger.info(f"To test: cd {ns3_path} && ./test_netmob25.sh")
    
    def _manual_ns3_integration(self, ns3_path: Path):
        """Manual NS-3 integration when export script is not available"""
        logger.info("Performing manual NS-3 integration")
        
        # Copy mobility model files if they exist in cpp_ns3_export
        cpp_ns3_export_dir = Path("cpp_ns3_export")
        
        mobility_files = [
            "netmob25-mobility-model.h",
            "netmob25-mobility-model.cc",
            "netmob25-mobility-example.cc"
        ]
        
        for filename in mobility_files:
            src_file = cpp_ns3_export_dir / filename
            if src_file.exists():
                if "example" in filename:
                    dst_path = ns3_path / "scratch" / filename
                else:
                    dst_path = ns3_path / "src" / "mobility" / "model" / filename
                
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(src_file, dst_path)
                logger.info(f"Copied {filename} to {dst_path}")
        
        # Update CMakeLists.txt if it hasn't been updated
        self._update_cmake_if_needed(ns3_path)
    
    def _update_cmake_if_needed(self, ns3_path: Path):
        """Update mobility CMakeLists.txt if netmob25 files aren't included"""
        cmake_file = ns3_path / "src" / "mobility" / "CMakeLists.txt"
        
        if not cmake_file.exists():
            logger.warning(f"CMakeLists.txt not found at {cmake_file}")
            return
        
        with open(cmake_file, 'r') as f:
            content = f.read()
        
        if "netmob25-mobility-model.cc" in content:
            logger.info("CMakeLists.txt already includes netmob25 files")
            return
        
        logger.info("Updating CMakeLists.txt to include netmob25 files")
        
        # Add to MOBILITY_SOURCE_FILES if PyTorch section exists
        if "if(TARGET torch)" in content:
            # The modern approach is already there, files should be added automatically
            logger.info("PyTorch integration detected, files should be added automatically")
        else:
            logger.warning("Manual CMakeLists.txt update may be needed")
    
    def _create_ns3_test_script(self, ns3_path: Path):
        """Create a test script for NS-3 integration"""
        test_script = ns3_path / "test_netmob25_export.sh"
        
        test_content = f'''#!/bin/bash
# Test script for exported Netmob25 mobility model in NS-3

echo "=== Testing Exported Netmob25 Mobility Model ==="

# Configure NS-3 with PyTorch support
echo "Configuring NS-3..."
CMAKE_PREFIX_PATH="/Users/raphaelcousin/Library/Caches/pypoetry/virtualenvs/ml-mobility-ns3-nuqJhA4m-py3.13/lib/python3.13/site-packages/torch/share/cmake" ./ns3 configure --enable-examples --enable-tests

# Build NS-3
echo "Building NS-3..."
CMAKE_PREFIX_PATH="/Users/raphaelcousin/Library/Caches/pypoetry/virtualenvs/ml-mobility-ns3-nuqJhA4m-py3.13/lib/python3.13/site-packages/torch/share/cmake" ./ns3 build

# Test basic mobility model
echo "Testing basic mobility model..."
./ns3 run "scratch/netmob25-mobility-example --nNodes=3 --simTime=10 --printInterval=2"

# Test with ML model if available
if [ -f "model.pt" ]; then
    echo "Testing with ML model..."
    ./ns3 run "scratch/netmob25-mobility-example --useMLGeneration=true --modelPath=model.pt --nNodes=2 --simTime=5"
else
    echo "No model.pt found, skipping ML test"
fi

echo "Test complete!"
'''
        
        with open(test_script, 'w') as f:
            f.write(test_content)
        test_script.chmod(0o755)
        
        logger.info(f"Created NS-3 test script: {test_script}")