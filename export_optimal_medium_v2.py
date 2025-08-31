#!/usr/bin/env python3
"""
Export script for the optimal_medium_v2 model using the legacy notebook architecture.
This script exports the trained model to C++ NS-3 format.
"""

import torch
import torch.nn as nn
import numpy as np
import json
import pickle
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Legacy ConditionalTrajectoryVAE class from notebooks/Untitled.py
class ConditionalTrajectoryVAE(nn.Module):
    """LSTM-based Conditional VAE for trajectory generation with transport mode and length conditioning."""

    def __init__(
        self,
        input_dim: int = 3,  # lat, lon, speed
        sequence_length: int = 2000,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        num_layers: int = 2,
        num_transport_modes: int = 5,  # number of transport modes
        condition_dim: int = 32,  # dimension for condition embeddings
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.num_transport_modes = num_transport_modes
        self.num_layers = num_layers
        self.dropout = dropout

        # Condition embeddings
        self.transport_mode_embedding = nn.Embedding(num_transport_modes, condition_dim)
        self.length_projection = nn.Linear(1, condition_dim)

        # Total condition dimension (transport mode + length)
        total_condition_dim = condition_dim * 2

        # Encoder - LSTM with bidirectional processing
        self.encoder_lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, 
            batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0
        )
        encoder_output_dim = hidden_dim * 2  # bidirectional

        # Latent space projections (include conditions)
        self.fc_mu = nn.Linear(encoder_output_dim + total_condition_dim, latent_dim)
        self.fc_logvar = nn.Linear(encoder_output_dim + total_condition_dim, latent_dim)

        # Decoder
        self.fc_latent = nn.Linear(latent_dim + total_condition_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(
            hidden_dim, hidden_dim, num_layers, 
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.fc_out = nn.Linear(hidden_dim, input_dim)

        # Dropout layers
        self.dropout_layer = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.uniform_(param, -0.1, 0.1)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def get_conditions(self, transport_mode: torch.Tensor, trip_length: torch.Tensor) -> torch.Tensor:
        """Create condition embeddings from transport mode and trip length."""
        # Transport mode embedding
        mode_embed = self.transport_mode_embedding(transport_mode)  # (batch, condition_dim)

        # Length embedding (normalize length and project)
        length_normalized = trip_length.unsqueeze(-1).float() / self.sequence_length  # normalize to [0,1]
        length_embed = self.length_projection(length_normalized)  # (batch, condition_dim)

        # Concatenate conditions
        conditions = torch.cat([mode_embed, length_embed], dim=-1)  # (batch, condition_dim * 2)
        return conditions

    def encode(self, x: torch.Tensor, conditions: torch.Tensor):
        """Encode trajectory to latent distribution parameters."""
        # LSTM encoding
        _, (h, _) = self.encoder_lstm(x)
        # Concatenate forward and backward hidden states from last layer
        h = torch.cat([h[-2], h[-1]], dim=1)  # (batch, hidden_dim * 2)
        h = self.dropout_layer(h)

        # Concatenate with conditions
        h_conditioned = torch.cat([h, conditions], dim=-1)

        mu = self.fc_mu(h_conditioned)
        logvar = self.fc_logvar(h_conditioned)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        """Decode from latent space to trajectory."""
        batch_size = z.size(0)

        # Concatenate latent with conditions
        z_conditioned = torch.cat([z, conditions], dim=-1)

        # Project to hidden and create sequence
        h = self.fc_latent(z_conditioned)
        h = torch.tanh(h)  # Add non-linearity
        h = h.unsqueeze(1).repeat(1, self.sequence_length, 1)

        # Decode sequence
        out, _ = self.decoder_lstm(h)
        out = self.dropout_layer(out)
        out = self.fc_out(out)

        return out

    def forward(self, x: torch.Tensor, transport_mode: torch.Tensor, trip_length: torch.Tensor, mask=None):
        """Forward pass."""
        conditions = self.get_conditions(transport_mode, trip_length)
        mu, logvar = self.encode(x, conditions)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, conditions)
        return recon, mu, logvar

    def generate(self, transport_mode: torch.Tensor, trip_length: torch.Tensor, n_samples=None, device='cpu') -> torch.Tensor:
        """Generate new trajectories given conditions."""
        if n_samples is None:
            n_samples = transport_mode.size(0)

        conditions = self.get_conditions(transport_mode, trip_length)
        z = torch.randn(n_samples, self.latent_dim).to(device)

        with torch.no_grad():
            trajectories = self.decode(z, conditions)
        return trajectories


def load_model_and_scalers():
    """Load the trained model and scalers."""
    # Paths
    model_dir = Path("results/optimal_medium_v2")
    config_path = model_dir / "config.json"
    weights_path = model_dir / "best_model.pt"
    scalers_path = Path("data/processed/scalers.pkl")
    
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    model_config = config['model_config']
    
    # Create model with config from the saved model
    model = ConditionalTrajectoryVAE(
        input_dim=model_config['input_dim'],
        sequence_length=model_config['sequence_length'],
        hidden_dim=model_config['hidden_dim'],
        latent_dim=model_config['latent_dim'],
        num_layers=model_config['num_layers'],
        num_transport_modes=model_config['num_transport_modes'],
        condition_dim=model_config['condition_dim'],
        dropout=model_config['dropout']
    )
    
    # Load weights
    logger.info(f"Loading model weights from {weights_path}")
    checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
    
    # Extract model state dict from the checkpoint
    if 'model_state_dict' in checkpoint:
        model_state_dict = checkpoint['model_state_dict']
    else:
        model_state_dict = checkpoint
    
    model.load_state_dict(model_state_dict)
    model.eval()
    
    # Load scalers
    logger.info(f"Loading scalers from {scalers_path}")
    with open(scalers_path, 'rb') as f:
        scalers = pickle.load(f)
    
    logger.info("Model and scalers loaded successfully!")
    return model, scalers, config


def export_to_cpp(model, scalers, config, output_dir="cpp_ns3_export/optimal_medium_v2"):
    """Export model to C++ format for NS-3."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Exporting to {output_path}")
    
    # 1. Save model weights as TorchScript
    logger.info("Creating TorchScript model...")
    
    # Create a simplified model wrapper for export
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            
        def forward(self, x, transport_mode, trip_length):
            # Call the original model and return only reconstruction
            recon, _, _ = self.model(x, transport_mode, trip_length)
            return recon
    
    wrapped_model = ModelWrapper(model)
    wrapped_model.eval()
    
    # Create example inputs for tracing
    batch_size = 1
    seq_len = 100  # Smaller for testing
    input_dim = model.input_dim
    
    example_x = torch.randn(batch_size, seq_len, input_dim)
    example_mode = torch.zeros(batch_size, dtype=torch.long)
    example_length = torch.tensor([seq_len] * batch_size, dtype=torch.long)
    
    # Trace the model
    try:
        with torch.no_grad():
            traced_model = torch.jit.trace(wrapped_model, (example_x, example_mode, example_length))
        
        model_path = output_path / 'model.pt'
        traced_model.save(str(model_path))
        logger.info(f"Saved TorchScript model to {model_path}")
    except Exception as e:
        logger.error(f"Failed to trace model: {e}")
        # Fallback: save state dict
        torch.save(model.state_dict(), output_path / 'model_state.pt')
        logger.info("Saved model state dict as fallback")
    
    # 2. Save metadata
    logger.info("Saving metadata...")
    metadata = {
        'experiment_name': 'optimal_medium_v2',
        'model_type': 'ConditionalTrajectoryVAE',
        'model_config': config['model_config'],
        'transport_modes': config['dataset_info']['transport_modes'],
        'n_transport_modes': config['dataset_info']['n_transport_modes'],
        'sequence_length': config['model_config']['sequence_length'],
        'input_dim': config['model_config']['input_dim'],
        'latent_dim': config['model_config']['latent_dim'],
        'hidden_dim': config['model_config']['hidden_dim'],
        'scaling_method': 'StandardScaler',  # Based on the old preprocessing
        'note': 'Legacy model exported from optimal_medium_v2 with individual transport modes'
    }
    
    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved metadata.json")
    
    # 3. Save scalers in JSON format for C++
    logger.info("Converting scalers to JSON...")
    
    trajectory_scaler = scalers['trajectory']
    
    # Check scaler type and extract parameters accordingly
    if hasattr(trajectory_scaler, 'mean_'):
        # StandardScaler
        scalers_json = {
            'trajectory': {
                'mean': trajectory_scaler.mean_.tolist(),
                'scale': trajectory_scaler.scale_.tolist(),
                'type': 'StandardScaler'
            }
        }
    elif hasattr(trajectory_scaler, 'min_'):
        # MinMaxScaler
        scalers_json = {
            'trajectory': {
                'min': trajectory_scaler.min_.tolist(),
                'scale': trajectory_scaler.scale_.tolist(),
                'data_min': trajectory_scaler.data_min_.tolist(),
                'data_max': trajectory_scaler.data_max_.tolist(),
                'type': 'MinMaxScaler'
            }
        }
    else:
        # Unknown scaler type
        logger.warning("Unknown scaler type, using default parameters")
        scalers_json = {
            'trajectory': {
                'min': [0.0, 0.0, 0.0],
                'scale': [1.0, 1.0, 1.0],
                'type': 'Identity'
            }
        }
    
    with open(output_path / 'scalers.json', 'w') as f:
        json.dump(scalers_json, f, indent=2)
    logger.info("Saved scalers.json")
    
    # 4. Generate NS-3 C++ files using templates
    logger.info("Generating NS-3 C++ files...")
    generate_ns3_files(metadata, output_path)
    
    logger.info(f"✅ Export complete! Files saved to {output_path}")
    return output_path


def generate_ns3_files(metadata, output_path):
    """Generate NS-3 C++ files from templates."""
    template_dir = Path('cpp_project')
    
    if not template_dir.exists():
        logger.error(f"Template directory {template_dir} not found")
        return
        
    # Setup Jinja2 environment
    env = Environment(loader=FileSystemLoader(str(template_dir)))
    
    # Template context
    context = {
        'project_name': 'netmob25_mobility',
        'experiment_name': metadata['experiment_name'],
        'model_path': 'model.pt',
        'sequence_length': metadata['sequence_length'],
        'input_dim': metadata['input_dim'],
        'latent_dim': metadata['latent_dim'],
        'hidden_dim': metadata['hidden_dim'],
        'transport_modes': metadata['transport_modes'],
        'n_transport_modes': metadata['n_transport_modes']
    }
    
    # Generate NS-3 files from templates
    ns3_templates = [
        'netmob25-mobility-model.h.jinja',
        'netmob25-mobility-model.cc.jinja', 
        'netmob25-mobility-example.cc.jinja',
        'CMakeLists.txt.jinja'
    ]
    
    for template_name in ns3_templates:
        template_file = template_dir / template_name
        if template_file.exists():
            template = env.get_template(template_name)
            output_content = template.render(**context)
            
            # Remove .jinja extension and save
            output_file = output_path / template_file.stem
            with open(output_file, 'w') as f:
                f.write(output_content)
            logger.info(f"Generated NS-3 file: {output_file.name}")
        else:
            logger.warning(f"Template {template_name} not found")


def main():
    """Main export function."""
    logger.info("Starting export of optimal_medium_v2 model...")
    
    try:
        # Load model and scalers
        model, scalers, config = load_model_and_scalers()
        
        # Export to C++
        output_path = export_to_cpp(model, scalers, config)
        
        # Print summary
        logger.info(f"""
🎉 Export completed successfully!

📂 Output directory: {output_path}
📁 Generated files:
  - model.pt (TorchScript model)
  - metadata.json (model configuration)
  - scalers.json (coordinate transformation)
  - netmob25-mobility-model.h (NS-3 header)
  - netmob25-mobility-model.cc (NS-3 implementation)
  - netmob25-mobility-example.cc (NS-3 example)
  - CMakeLists.txt (NS-3 build configuration)

🚀 Ready for NS-3 integration!
        """)
        
    except Exception as e:
        logger.error(f"Export failed: {e}")
        raise


if __name__ == "__main__":
    main()