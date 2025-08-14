#!/usr/bin/env python
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from pathlib import Path
import sys
import pickle
import json
import logging

sys.path.append(str(Path(__file__).parent.parent))
from ml_mobility_ns3.export.cpp_export import CppExporter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # Handle experiment_id parameter
    if hasattr(cfg, 'experiment_id'):
        experiment_dir = Path('experiments') / cfg.experiment_id
        if not experiment_dir.exists():
            raise ValueError(f"Experiment {cfg.experiment_id} not found")
        
        # Load checkpoint from experiment
        checkpoint_path = experiment_dir / 'checkpoints' / 'best_model.ckpt'
        if not checkpoint_path.exists():
            checkpoint_path = experiment_dir / 'checkpoints' / 'last.ckpt'
        
        # Load experiment config
        with open(experiment_dir / 'config.yaml', 'r') as f:
            experiment_cfg = OmegaConf.load(f)
            # Create a new config with structured merging disabled
            cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
            OmegaConf.set_struct(cfg, False)
            # Override the export.output_dir to use cpp_ns3_export
            if 'export' not in experiment_cfg:
                experiment_cfg['export'] = {}
            experiment_cfg['export']['output_dir'] = 'cpp_ns3_export'
            cfg = OmegaConf.merge(cfg, experiment_cfg)
    else:
        # Use default checkpoint path
        checkpoint_path = Path(cfg.get('checkpoint_path', 'best_model.ckpt'))
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    logger.info(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Extract model state from checkpoint
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model_state = checkpoint['state_dict']
        hyper_parameters = checkpoint.get('hyper_parameters', {})
    else:
        model_state = checkpoint
        hyper_parameters = {}
    
    # Load metadata if available
    metadata_path = Path('data/processed/metadata.pkl')
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
    
    # Export - use experiment_id if available, otherwise use model name
    if hasattr(cfg, 'experiment_id'):
        export_name = cfg.experiment_id
    else:
        export_name = checkpoint_path.stem
    
    exporter = CppExporter(cfg)
    exporter.export_model(checkpoint, metadata, export_name)
    
    # Check for NS-3 integration option
    if hasattr(cfg, 'ns3_path') and cfg.ns3_path:
        logger.info(f"Integrating with NS-3 at {cfg.ns3_path}")
        exporter.integrate_with_ns3(cfg.ns3_path)
    

if __name__ == "__main__":
    main()