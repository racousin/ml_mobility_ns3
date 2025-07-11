#!/usr/bin/env python
import hydra
from omegaconf import DictConfig
import logging
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from ml_mobility_ns3.data.preprocessor import TrajectoryPreprocessor

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    logger.info("Starting data preprocessing")
    
    preprocessor = TrajectoryPreprocessor(cfg.data)
    dataset = preprocessor.process()
    
    logger.info(f"Preprocessing complete. Output saved to {cfg.data.output_dir}")
    

if __name__ == "__main__":
    main()