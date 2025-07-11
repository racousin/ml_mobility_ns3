#!/usr/bin/env python
import hydra
from omegaconf import DictConfig
import torch
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from ml_mobility_ns3.export.cpp_export import CppExporter


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # Load model
    checkpoint_path = Path(cfg.training.checkpoint_dir) / 'best_model.ckpt'
    model = torch.load(checkpoint_path)
    
    # Load metadata
    metadata_path = Path(cfg.data.output_dir) / 'metadata.pkl'
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    # Export
    exporter = CppExporter(cfg)
    exporter.export_model(model, metadata)
    

if __name__ == "__main__":
    main()