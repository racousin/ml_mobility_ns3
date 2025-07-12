#!/usr/bin/env python
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from pathlib import Path
import json
import sys
import logging
import re

sys.path.append(str(Path(__file__).parent.parent))
from ml_mobility_ns3.evaluation.evaluator import TrajectoryEvaluator
from ml_mobility_ns3.data.dataset import TrajectoryDataset
from ml_mobility_ns3.training.lightning_module import TrajectoryLightningModule
from torch.utils.data import DataLoader


def find_experiment_dir(experiment_id: str = None) -> Path:
    """Find experiment directory by ID or get the latest."""
    experiments_dir = Path("experiments")
    
    if experiment_id:
        # If full path provided
        if "/" in experiment_id or "\\" in experiment_id:
            exp_dir = Path(experiment_id)
            if exp_dir.exists():
                return exp_dir
            
        # If just experiment ID provided
        exp_dir = experiments_dir / experiment_id
        if exp_dir.exists():
            return exp_dir
            
        # Try to find by partial match
        matching = [d for d in experiments_dir.iterdir() 
                   if d.is_dir() and experiment_id in d.name]
        if matching:
            return matching[0]
    
    # Get latest experiment
    all_experiments = [d for d in experiments_dir.iterdir() if d.is_dir()]
    if all_experiments:
        return sorted(all_experiments, key=lambda x: x.stat().st_mtime)[-1]
    
    return None


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    logger = logging.getLogger(__name__)
    
    # Find experiment directory
    experiment_id = cfg.get('experiment_id', None) or cfg.get('exp_id', None)
    
    if experiment_id:
        exp_dir = find_experiment_dir(experiment_id)
    else:
        # Find latest experiment
        exp_dir = find_experiment_dir()
    
    if not exp_dir or not exp_dir.exists():
        logger.error("No experiment directory found")
        logger.info("Usage:")
        logger.info("  python scripts/evaluate.py experiment_id=dummy_2025-07-12_13-14-14")
        logger.info("  python scripts/evaluate.py exp_id=dummy_2025-07-12_13-14-14")
        logger.info("  python scripts/evaluate.py  # Uses latest experiment")
        
        # List available experiments
        experiments_dir = Path("experiments")
        if experiments_dir.exists():
            experiments = [d.name for d in experiments_dir.iterdir() if d.is_dir()]
            if experiments:
                logger.info("\nAvailable experiments:")
                for exp in sorted(experiments)[-10:]:  # Show last 10
                    logger.info(f"  {exp}")
        return
    
    checkpoint_dir = exp_dir / "checkpoints"
    logger.info(f"Using experiment: {exp_dir.name}")
    
    # Load experiment config if available
    exp_config_path = exp_dir / "config.yaml"
    if exp_config_path.exists():
        exp_cfg = OmegaConf.load(exp_config_path)
        
        # Smart config merging - handle missing keys gracefully
        # First, create a non-struct config for merging
        base_cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        exp_cfg_dict = OmegaConf.to_container(exp_cfg, resolve=True)
        
        # Deep merge the configs
        def deep_merge(base, update):
            for key, value in update.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    deep_merge(base[key], value)
                else:
                    base[key] = value
            return base
        
        merged_dict = deep_merge(base_cfg_dict, exp_cfg_dict)
        cfg = OmegaConf.create(merged_dict)
        
        logger.info(f"Loaded experiment config from {exp_config_path}")
    
    # Look for checkpoints
    checkpoint_path = None
    
    # First try best_model.ckpt
    best_model_path = checkpoint_dir / 'best_model.ckpt'
    if best_model_path.exists():
        checkpoint_path = best_model_path
        logger.info(f"Using best model checkpoint: {checkpoint_path}")
    else:
        # Get all .ckpt files
        all_checkpoints = list(checkpoint_dir.glob('*.ckpt'))
        
        # Filter checkpoints
        checkpoints = []
        for cp in all_checkpoints:
            if cp.stem == 'last':
                continue
            # Look for newer format first (00_0.0000.ckpt)
            if re.match(r'^\d{2}[_-]\d+\.\d+$', cp.stem):
                checkpoints.append(cp)
            # Also check for formats with epoch= prefix
            elif 'epoch=' in cp.stem and 'val_loss' in cp.stem:
                checkpoints.append(cp)
        
        if checkpoints:
            # Try to extract validation loss from filename
            checkpoint_losses = []
            for cp in checkpoints:
                # Try different formats
                patterns = [
                    r'val_loss[=_](\d+\.\d+)',
                    r'^\d+[_-](\d+\.\d+)$',
                    r'epoch=\d+[_-]val_loss[=_](\d+\.\d+)'
                ]
                
                for pattern in patterns:
                    match = re.search(pattern, cp.stem)
                    if match:
                        loss = float(match.group(1))
                        checkpoint_losses.append((cp, loss))
                        break
                else:
                    # Use modification time as fallback
                    checkpoint_losses.append((cp, cp.stat().st_mtime))
            
            if checkpoint_losses:
                checkpoint_losses.sort(key=lambda x: x[1])
                checkpoint_path = checkpoint_losses[0][0]
                logger.info(f"Found best checkpoint: {checkpoint_path}")
    
    # Fallback to last checkpoint
    if checkpoint_path is None and (checkpoint_dir / 'last.ckpt').exists():
        checkpoint_path = checkpoint_dir / 'last.ckpt'
        logger.info(f"Using last checkpoint: {checkpoint_path}")
    
    if checkpoint_path is None:
        logger.error(f"No valid checkpoints found in {checkpoint_dir}")
        logger.info("Available files: " + ", ".join([f.name for f in checkpoint_dir.iterdir()]))
        return
    
    # Load model from checkpoint
    logger.info(f"Loading model from {checkpoint_path}")
    try:
        # For PyTorch 2.6+, we need to handle OmegaConf objects
        import torch.serialization
        torch.serialization.add_safe_globals([
            OmegaConf,
            DictConfig,
            type(cfg),
            type(cfg.model)
        ])
        
        # Load with proper error handling
        model = TrajectoryLightningModule.load_from_checkpoint(
            checkpoint_path,
            config=cfg,
            map_location='cpu',
            strict=False  # Allow missing/unexpected keys
        )
        model.eval()
        logger.info(f"Successfully loaded model: {cfg.model.name}")
        
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        
        # Try alternative loading method
        try:
            logger.info("Trying alternative loading method...")
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            
            # Create new model instance
            model = TrajectoryLightningModule(cfg)
            
            # Load state dict
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            model.eval()
            logger.info("Successfully loaded model with alternative method")
            
        except Exception as e2:
            logger.error(f"Alternative loading also failed: {e2}")
            return
    
    # Create evaluator
    evaluator = TrajectoryEvaluator(model.model, cfg)
    
    # Load test data
    dataset_path = Path(cfg.data.output_dir) / 'dataset.npz'
    if not dataset_path.exists():
        logger.error(f"Dataset not found at {dataset_path}")
        logger.info("Please run preprocessing first using: python scripts/preprocess.py")
        return
        
    dataset = TrajectoryDataset(dataset_path)
    
    # Use a subset for evaluation
    test_size = min(1000, len(dataset))
    test_indices = torch.randperm(len(dataset))[:test_size]
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    logger.info(f"Evaluating on {len(test_dataset)} samples...")
    
    # Evaluate
    recon_metrics = evaluator.evaluate_reconstruction(test_loader)
    gen_metrics = evaluator.evaluate_generation()
    
    # Save results
    results = {
        'experiment_id': exp_dir.name,
        'checkpoint': str(checkpoint_path),
        'model_type': cfg.model.name,
        'reconstruction': recon_metrics,
        'generation': gen_metrics
    }
    
    # Save in experiment directory
    output_path = exp_dir / 'evaluation_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {output_path}")
    
    # Also save in root for convenience
    root_output_path = Path('evaluation_results.json')
    with open(root_output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n=== Evaluation Results ===")
    print(f"Experiment: {exp_dir.name}")
    print(f"Model: {cfg.model.name}")
    print(f"Checkpoint: {checkpoint_path.name}")
    
    print("\nReconstruction Metrics:")
    for key, value in recon_metrics.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}")
    
    print("\nGeneration Metrics:")
    for key, value in gen_metrics.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    

if __name__ == "__main__":
    main()