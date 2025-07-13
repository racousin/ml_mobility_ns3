#!/usr/bin/env python
"""Simplified evaluation script for trajectory generation models."""
import hydra
from omegaconf import DictConfig
import torch
from pathlib import Path
import json
import sys
import logging

sys.path.append(str(Path(__file__).parent.parent))
from ml_mobility_ns3.evaluation.evaluator import TrajectoryEvaluator
from ml_mobility_ns3.data.dataset import TrajectoryDataset
from ml_mobility_ns3.training.lightning_module import TrajectoryLightningModule
from ml_mobility_ns3.utils.experiment_utils import ExperimentManager
from ml_mobility_ns3.utils.model_utils import load_checkpoint, merge_configs
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Main evaluation entry point."""
    exp_manager = ExperimentManager()
    
    # Find experiment
    experiment_id = cfg.get('experiment_id') or cfg.get('exp_id')
    exp_dir = exp_manager.find_experiment_dir(experiment_id)
    
    if not exp_dir or not exp_dir.exists():
        logger.error("No experiment directory found")
        logger.info("Usage: python scripts/evaluate.py experiment_id=experiment_name")
        return
    
    logger.info(f"Using experiment: {exp_dir.name}")
    
    # Load experiment config
    _, exp_config = exp_manager.load_experiment_info(exp_dir)
    if exp_config:
        cfg = merge_configs(cfg, exp_config)
    
    # Find best checkpoint
    checkpoint_path = exp_manager.find_best_checkpoint(exp_dir / "checkpoints")
    if checkpoint_path is None:
        logger.error(f"No valid checkpoints found")
        return
    
    logger.info(f"Using checkpoint: {checkpoint_path}")
    
    # Load model
    model = load_checkpoint(
        checkpoint_path,
        cfg,
        TrajectoryLightningModule,
        device='cpu'
    )
    
    # Load entire dataset
    dataset_path = Path(cfg.data.output_dir) / 'dataset.npz'
    if not dataset_path.exists():
        logger.error(f"Dataset not found at {dataset_path}")
        return
    
    dataset = TrajectoryDataset(dataset_path)
    logger.info(f"Dataset size: {len(dataset)} sequences")
    
    # Create dataloader for entire dataset
    dataloader = DataLoader(
        dataset, 
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=9
    )
    
    # Evaluate
    evaluator = TrajectoryEvaluator(model.model, cfg)
    results = evaluator.evaluate_generation(dataloader)
    
    # Print results
    if 'real_stats' in results and results['real_stats']:
        evaluator.stat_metrics.print_category_statistics(
            results['real_stats'], 
            "Real Data Statistics"
        )
    
    if 'generated_stats' in results:
        evaluator.stat_metrics.print_category_statistics(
            results['generated_stats'], 
            "Generated Data Statistics"
        )
    
    # Save results
    output_path = exp_dir / 'evaluation_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_path}")
    
    # Print comparison if both statistics available
    if 'real_stats' in results and 'generated_stats' in results:
        print("\n=== Relative Differences ===")
        for cat_name in sorted(results['generated_stats'].keys()):
            if cat_name in results['real_stats']:
                real = results['real_stats'][cat_name]
                gen = results['generated_stats'][cat_name]
                
                print(f"\n{cat_name}:")
                metrics = ['duration_mean', 'total_distance_mean', 'bird_distance_mean', 'speed_mean']
                for metric in metrics:
                    real_val = real[metric]
                    gen_val = gen[metric]
                    diff = (gen_val - real_val) / (real_val + 1e-8) * 100
                    metric_name = metric.replace('_mean', '').replace('_', ' ').title()
                    print(f"  {metric_name}: {diff:+.1f}%")


if __name__ == "__main__":
    main()