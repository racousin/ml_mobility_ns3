# ml_mobility_ns3/utils/model_utils.py
import torch
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


def deep_merge(base: Dict, update: Dict) -> Dict:
    """Deep merge two dictionaries."""
    for key, value in update.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_checkpoint(checkpoint_path: Path, 
                   cfg: DictConfig,
                   lightning_module_class,
                   device: str = 'cpu',
                   skip_loss_init: bool = True) -> Any:
    """Load a model from checkpoint with proper error handling.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        cfg: Configuration object
        lightning_module_class: Lightning module class to instantiate
        device: Device to load the model on
        skip_loss_init: Skip loss function initialization (for evaluation mode)
    """
    logger.info(f"Loading model from {checkpoint_path}")
    
    try:
        # For PyTorch 2.6+, we need to handle OmegaConf objects
        torch.serialization.add_safe_globals([
            OmegaConf,
            DictConfig,
            type(cfg),
            type(cfg.model) if hasattr(cfg, 'model') else None
        ])
        
        # Load with proper error handling
        model = lightning_module_class.load_from_checkpoint(
            checkpoint_path,
            config=cfg,
            skip_loss_init=skip_loss_init,
            map_location=device,
            strict=False  # Allow missing/unexpected keys
        )
        model.eval()
        logger.info(f"Successfully loaded model: {cfg.model.name}")
        return model
        
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        
        # Try alternative loading method
        try:
            logger.info("Trying alternative loading method...")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            
            # Create new model instance with skip_loss_init
            model = lightning_module_class(cfg, skip_loss_init=skip_loss_init)
            
            # Load state dict
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            model.eval()
            logger.info("Successfully loaded model with alternative method")
            return model
            
        except Exception as e2:
            logger.error(f"Alternative loading also failed: {e2}")
            raise RuntimeError(f"Could not load checkpoint from {checkpoint_path}")


def merge_configs(base_cfg: DictConfig, exp_cfg: Optional[DictConfig]) -> DictConfig:
    """Merge experiment config with base config."""
    if exp_cfg is None:
        return base_cfg
    
    # Convert to containers for merging
    base_cfg_dict = OmegaConf.to_container(base_cfg, resolve=True)
    exp_cfg_dict = OmegaConf.to_container(exp_cfg, resolve=True)
    
    # Deep merge the configs
    merged_dict = deep_merge(base_cfg_dict, exp_cfg_dict)
    
    # Create new config
    return OmegaConf.create(merged_dict)