#!/usr/bin/env python
"""Clean training script for trajectory models."""
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from pathlib import Path
import sys
import logging
import json
from datetime import datetime
import shutil

sys.path.append(str(Path(__file__).parent.parent))
from ml_mobility_ns3.data.dataset import TrajectoryDataset
from ml_mobility_ns3.training.lightning_module import TrajectoryLightningModule
from ml_mobility_ns3.training.callbacks import BestMetricsTracker, EarlyStoppingTracker
from ml_mobility_ns3.utils.experiment_utils import ExperimentManager
from torch.utils.data import DataLoader, random_split

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Encapsulates the training pipeline logic."""
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.exp_manager = ExperimentManager()
        
    def setup_experiment(self) -> Path:
        """Setup experiment directory and tracking."""
        # Create experiment ID
        hydra_config = getattr(self.cfg, 'hydra', None)
        experiment_id = self.exp_manager.create_experiment_id(
            self.cfg.model.name, 
            OmegaConf.to_container(hydra_config) if hydra_config else None
        )
        
        # Setup directory structure
        exp_dir = Path("experiments") / experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        (exp_dir / "checkpoints").mkdir(exist_ok=True)
        (exp_dir / "logs").mkdir(exist_ok=True)
        
        # Save config
        with open(exp_dir / "config.yaml", "w") as f:
            OmegaConf.save(self.cfg, f)
        
        # Create initial model info
        model_info = {
            "experiment_id": experiment_id,
            "model_type": self.cfg.model.name,
            "model_class": self.cfg.model._target_.split('.')[-1],
            "created_at": datetime.now().isoformat(),
            "status": "training",
            "architecture": OmegaConf.to_container(self.cfg.model),
            "training_config": OmegaConf.to_container(self.cfg.training),
            "hydra_config": OmegaConf.to_container(hydra_config) if hydra_config else None
        }
        
        with open(exp_dir / "model_info.json", "w") as f:
            json.dump(model_info, f, indent=2)
        
        # Update manifest
        self.exp_manager.update_manifest(experiment_id, self.cfg.model.name)
        
        logger.info(f"Created experiment: {experiment_id}")
        logger.info(f"Experiment directory: {exp_dir}")
        
        return exp_dir, experiment_id
    
    def setup_device(self):
        """Setup computing device."""
        if self.cfg.accelerator == 'auto':
            if torch.cuda.is_available():
                self.cfg.accelerator = 'gpu'
                self.cfg.device = 'cuda'
                # Keep devices as is for GPU (could be 'auto', int, or list)
            elif torch.backends.mps.is_available():
                self.cfg.accelerator = 'mps'
                self.cfg.device = 'mps'
                # Keep devices as is for MPS
            else:
                self.cfg.accelerator = 'cpu'
                self.cfg.device = 'cpu'
                # Set devices to 1 for CPU (required by PyTorch Lightning)
                self.cfg.devices = 1
        elif self.cfg.accelerator == 'cpu':
            # Ensure device is set correctly for CPU
            self.cfg.device = 'cpu'
            # Set devices to integer for CPU accelerator
            if not isinstance(self.cfg.devices, int):
                self.cfg.devices = 1
        
        logger.info(f"Using accelerator: {self.cfg.accelerator}, device: {self.cfg.device}, devices: {self.cfg.devices}")
    
    def load_data(self) -> tuple:
        """Load and split dataset."""
        dataset_path = Path(self.cfg.data.output_dir) / 'dataset.npz'
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {dataset_path}. "
                "Please run preprocessing first using: python scripts/preprocess.py"
            )
        
        dataset = TrajectoryDataset(dataset_path)
        logger.info(f"Loaded dataset with {len(dataset)} samples")
        
        # Split dataset
        val_size = int(len(dataset) * self.cfg.training.val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.cfg.training.batch_size,
            shuffle=True,
            num_workers=9,
            pin_memory=False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.cfg.training.batch_size,
            shuffle=False,
            num_workers=9,
            pin_memory=False
        )
        
        return train_loader, val_loader
    

    def create_callbacks(self, exp_dir: Path) -> list:
        """Create training callbacks."""
        callbacks = []
        
        # Model checkpoint
        checkpoint_callback = ModelCheckpoint(
            dirpath=exp_dir / "checkpoints",
            filename='{epoch:02d}_{val_loss:.4f}',
            monitor=self.cfg.training.best_metric_monitor ,
            mode='min',
            save_top_k=1,
            save_last=True,
            verbose=True,
            enable_version_counter=False
        )
        callbacks.append(checkpoint_callback)
        
        # Early stopping (optional - check if enabled)
        if self.cfg.training.get('early_stopping_enabled', True):
            early_stopping = EarlyStopping(
                monitor=self.cfg.training.early_stopping_monitor,
                patience=self.cfg.training.early_stopping_patience,
                mode='min',
                verbose=True
            )
            callbacks.append(early_stopping)
            logger.info("Early stopping enabled")
        else:
            logger.info("Early stopping disabled")
        
        # Best metrics tracker
        metrics_tracker = BestMetricsTracker(
            exp_dir, 
            monitor=self.cfg.training.best_metric_monitor 
        )
        callbacks.append(metrics_tracker)
        
        # ADDED: Early stopping and LR tracking
        tracking_callback = EarlyStoppingTracker()
        callbacks.append(tracking_callback)
        
        return callbacks, checkpoint_callback
    
    def finalize_experiment(self, exp_dir: Path, experiment_id: str, 
                          trainer: pl.Trainer, checkpoint_callback, status: str):
        """Finalize experiment after training."""
        # Save best model checkpoint (only if checkpoint_callback exists)
        if checkpoint_callback and checkpoint_callback.best_model_path:
            best_checkpoint = Path(checkpoint_callback.best_model_path)
            best_model_link = exp_dir / "checkpoints" / "best_model.ckpt"
            
            if best_checkpoint.exists():
                shutil.copy2(best_checkpoint, best_model_link)
                logger.info(f"Copied best model to: {best_model_link}")
        
        # Load best metrics
        best_metrics_file = exp_dir / "best_metrics.json"
        if best_metrics_file.exists():
            with open(best_metrics_file, "r") as f:
                best_metrics = json.load(f)
        else:
            best_metrics = {}
        
        # Update model info
        model_info_path = exp_dir / "model_info.json"
        with open(model_info_path, "r") as f:
            model_info = json.load(f)
        
        # Add parameter count if not already there and model exists
        if "parameters" not in model_info and hasattr(self, 'model'):
            param_count = sum(p.numel() for p in self.model.model.parameters())
            model_info["parameters"] = {
                "total": param_count,
                "trainable": sum(p.numel() for p in self.model.model.parameters() if p.requires_grad)
            }
        
        # Update with final info
        final_metrics = {
            "best_val_loss": best_metrics.get('val_loss'),
            "best_epoch": best_metrics.get('epoch'),
            "best_metrics": best_metrics
        }
        
        # Add trainer info if available
        if trainer:
            final_metrics["epochs_trained"] = trainer.current_epoch + 1
        
        model_info.update({
            "status": status,
            "completed_at": datetime.now().isoformat(),
            "final_metrics": final_metrics,
            "checkpoint_files": {
                "best": "checkpoints/best_model.ckpt",
                "last": "checkpoints/last.ckpt"
            }
        })
        
        with open(model_info_path, "w") as f:
            json.dump(model_info, f, indent=2)
        
        # Update manifest
        self.exp_manager.update_manifest(experiment_id, self.cfg.model.name, status)
        
        # Save training summary
        summary = {
            "experiment_id": experiment_id,
            "model_type": self.cfg.model.name,
            "status": status,
            "final_metrics": final_metrics,
            "training_time": datetime.now().isoformat(),
            "checkpoints": {
                "best": str(exp_dir / "checkpoints" / "best_model.ckpt"),
                "last": str(exp_dir / "checkpoints" / "last.ckpt")
            }
        }
        
        with open(exp_dir / "training_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"\nExperiment completed: {experiment_id}")
        logger.info(f"Results saved in: {exp_dir}")
        if best_metrics.get('val_loss') is not None:
            logger.info(f"Best validation loss: {best_metrics['val_loss']:.4f}")
    
    def run(self):
        """Run the complete training pipeline."""
        # Setup
        exp_dir, experiment_id = self.setup_experiment()
        self.setup_device()
        pl.seed_everything(self.cfg.seed)
        
        try:
            # Load data
            train_loader, val_loader = self.load_data()
            
            # Create model
            self.model = TrajectoryLightningModule(self.cfg)
            
            # Create callbacks and logger
            callbacks, checkpoint_callback = self.create_callbacks(exp_dir)
            
            tb_logger = TensorBoardLogger(
                save_dir=exp_dir / "logs",
                name="tensorboard",
                version=""
            )
            
            # Create trainer
            trainer = pl.Trainer(
                max_epochs=self.cfg.training.epochs,
                accelerator=self.cfg.accelerator,
                devices=self.cfg.devices,
                callbacks=callbacks,
                logger=tb_logger,
                gradient_clip_val=self.cfg.training.gradient_clip,
                log_every_n_steps=10,
                val_check_interval=1.0,
                enable_progress_bar=True,
                enable_model_summary=True,
                default_root_dir=exp_dir
            )
            
            # Train
            trainer.fit(self.model, train_loader, val_loader)
            
            # Finalize
            self.finalize_experiment(
                exp_dir, experiment_id, trainer, checkpoint_callback, "completed"
            )
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            self.finalize_experiment(
                exp_dir, experiment_id, None, None, "failed"
            )
            raise


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Main training entry point."""
    pipeline = TrainingPipeline(cfg)
    pipeline.run()


if __name__ == "__main__":
    main()