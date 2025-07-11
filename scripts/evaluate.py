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


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    logger = logging.getLogger(__name__)
    
    # Find checkpoint
    checkpoint_dir = Path(cfg.training.checkpoint_dir)
    
    if not checkpoint_dir.exists():
        logger.error(f"Checkpoint directory {checkpoint_dir} does not exist")
        logger.info("Please train a model first using: python scripts/train.py")
        return
    
    # Look for checkpoints
    checkpoint_path = None
    
    # Get all .ckpt files
    all_checkpoints = list(checkpoint_dir.glob('*.ckpt'))
    
    # Filter checkpoints
    checkpoints = []
    for cp in all_checkpoints:
        if cp.stem == 'last':
            continue
        # Look for newer format first (00-0.0000.ckpt)
        if re.match(r'^\d{2}-\d+\.\d+$', cp.stem):
            checkpoints.append(cp)
        # Also check for old format
        elif 'val_loss' in cp.stem:
            checkpoints.append(cp)
    
    if checkpoints:
        # Try to extract validation loss from filename
        checkpoint_losses = []
        for cp in checkpoints:
            # Try new format first
            match = re.search(r'^(\d+)-(\d+\.\d+)$', cp.stem)
            if match:
                loss = float(match.group(2))
                checkpoint_losses.append((cp, loss))
            else:
                # Try old format
                match = re.search(r'val_loss[=_](\d+\.\d+)', cp.stem)
                if match:
                    loss = float(match.group(1))
                    checkpoint_losses.append((cp, loss))
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
            logger.info("You may need to specify the model type, e.g.: python scripts/evaluate.py model=dummy")
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
        'checkpoint': str(checkpoint_path),
        'model_type': cfg.model.name,
        'reconstruction': recon_metrics,
        'generation': gen_metrics
    }
    
    output_path = Path('evaluation_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {output_path}")
    
    # Print summary
    print("\n=== Evaluation Results ===")
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