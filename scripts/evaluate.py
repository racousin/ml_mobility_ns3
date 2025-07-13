#!/usr/bin/env python
"""Clean evaluation script for trajectory models."""
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


class EvaluationPipeline:
    """Encapsulates the evaluation pipeline logic."""
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.exp_manager = ExperimentManager()
    
    def find_and_load_experiment(self) -> tuple:
        """Find experiment and load its configuration."""
        # Get experiment ID from config
        experiment_id = self.cfg.get('experiment_id') or self.cfg.get('exp_id')
        
        # Find experiment directory
        exp_dir = self.exp_manager.find_experiment_dir(experiment_id)
        
        if not exp_dir or not exp_dir.exists():
            self._handle_no_experiment()
            return None, None
        
        logger.info(f"Using experiment: {exp_dir.name}")
        
        # Load experiment config and merge with base config
        _, exp_config = self.exp_manager.load_experiment_info(exp_dir)
        if exp_config:
            self.cfg = merge_configs(self.cfg, exp_config)
            logger.info(f"Loaded experiment config from {exp_dir}")
        
        return exp_dir
    
    def _handle_no_experiment(self):
        """Handle case when no experiment is found."""
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
    
    def load_model(self, exp_dir: Path) -> TrajectoryLightningModule:
        """Load model from experiment directory."""
        checkpoint_dir = exp_dir / "checkpoints"
        
        # Find best checkpoint
        checkpoint_path = self.exp_manager.find_best_checkpoint(checkpoint_dir)
        
        if checkpoint_path is None:
            logger.error(f"No valid checkpoints found in {checkpoint_dir}")
            logger.info("Available files: " + ", ".join([f.name for f in checkpoint_dir.iterdir()]))
            return None
        
        logger.info(f"Using checkpoint: {checkpoint_path}")
        
        # Load model
        model = load_checkpoint(
            checkpoint_path,
            self.cfg,
            TrajectoryLightningModule,
            device='cpu'
        )
        
        return model
    
    def load_test_data(self) -> DataLoader:
        """Load test dataset."""
        dataset_path = Path(self.cfg.data.output_dir) / 'dataset.npz'
        if not dataset_path.exists():
            logger.error(f"Dataset not found at {dataset_path}")
            logger.info("Please run preprocessing first using: python scripts/preprocess.py")
            return None
        
        dataset = TrajectoryDataset(dataset_path)
        
        # Use a subset for evaluation
        test_size = min(
            self.cfg.get('evaluation', {}).get('test_size', 1000), 
            len(dataset)
        )
        test_indices = torch.randperm(len(dataset))[:test_size]
        test_dataset = torch.utils.data.Subset(dataset, test_indices)
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.cfg.training.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        logger.info(f"Evaluating on {len(test_dataset)} samples...")
        
        return test_loader
    
    def evaluate_model(self, model: TrajectoryLightningModule, 
                      test_loader: DataLoader) -> dict:
        """Run evaluation."""
        evaluator = TrajectoryEvaluator(model.model, self.cfg)
        
        # Evaluate
        recon_metrics = evaluator.evaluate_reconstruction(test_loader)
        gen_metrics = evaluator.evaluate_generation()
        
        return {
            'reconstruction': recon_metrics,
            'generation': gen_metrics
        }
    
    def save_results(self, exp_dir: Path, results: dict, checkpoint_name: str):
        """Save evaluation results."""
        # Add metadata
        results['experiment_id'] = exp_dir.name
        results['checkpoint'] = checkpoint_name
        results['model_type'] = self.cfg.model.name
        
        # Save in experiment directory
        output_path = exp_dir / 'evaluation_results.json'
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
        
        # Also save in root for convenience
        root_output_path = Path('evaluation_results.json')
        with open(root_output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def print_summary(self, results: dict):
        """Print evaluation summary."""
        print("\n=== Evaluation Results ===")
        print(f"Experiment: {results['experiment_id']}")
        print(f"Model: {results['model_type']}")
        print(f"Checkpoint: {Path(results['checkpoint']).name}")
        
        print("\nReconstruction Metrics:")
        for key, value in results['reconstruction'].items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.4f}")
        
        print("\nGeneration Metrics:")
        for key, value in results['generation'].items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
    
    def run(self):
        """Run the complete evaluation pipeline."""
        # Find and load experiment
        exp_dir = self.find_and_load_experiment()
        if exp_dir is None:
            return
        
        # Load model
        model = self.load_model(exp_dir)
        if model is None:
            return
        
        # Load test data
        test_loader = self.load_test_data()
        if test_loader is None:
            return
        
        # Evaluate
        results = self.evaluate_model(model, test_loader)
        
        # Find checkpoint name for results
        checkpoint_path = self.exp_manager.find_best_checkpoint(exp_dir / "checkpoints")
        
        # Save results
        results = self.save_results(exp_dir, results, str(checkpoint_path))
        
        # Print summary
        self.print_summary(results)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Main evaluation entry point."""
    pipeline = EvaluationPipeline(cfg)
    pipeline.run()


if __name__ == "__main__":
    main()