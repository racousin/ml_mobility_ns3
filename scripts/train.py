#!/usr/bin/env python
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
from ml_mobility_ns3.training.callbacks import BestMetricsTracker

sys.path.append(str(Path(__file__).parent.parent))
from ml_mobility_ns3.data.dataset import TrajectoryDataset
from ml_mobility_ns3.training.lightning_module import TrajectoryLightningModule
from torch.utils.data import DataLoader, random_split


def create_experiment_id(model_name: str, cfg: DictConfig) -> str:
    """Create unique experiment ID with hydra job info."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Add hydra job number if in multirun
    if hasattr(cfg, 'hydra') and 'job' in cfg.hydra:
        job_num = cfg.hydra.job.num
        job_id = cfg.hydra.job.id
        return f"{model_name}_{timestamp}_job{job_num}_{job_id}"
    
    return f"{model_name}_{timestamp}"


def setup_experiment_dir(experiment_id: str, cfg: DictConfig) -> Path:
    """Setup experiment directory structure."""
    exp_dir = Path("experiments") / experiment_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (exp_dir / "checkpoints").mkdir(exist_ok=True)
    (exp_dir / "logs").mkdir(exist_ok=True)
    
    # Save config
    with open(exp_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)
    
    # Get loss configuration
    loss_config = cfg.training.get('loss', {'type': 'simple_vae', 'params': {'beta': 1.0}})
    
    # Create model info
    model_info = {
        "experiment_id": experiment_id,
        "model_type": cfg.model.name,
        "model_class": cfg.model._target_.split('.')[-1],
        "created_at": datetime.now().isoformat(),
        "status": "training",
        "architecture": OmegaConf.to_container(cfg.model),
        "training_config": OmegaConf.to_container(cfg.training),
        "loss_config": OmegaConf.to_container(loss_config),  # Add this
        "hydra_config": OmegaConf.to_container(cfg.hydra) if hasattr(cfg, 'hydra') else None
    }
    
    with open(exp_dir / "model_info.json", "w") as f:
        json.dump(model_info, f, indent=2)
    
    return exp_dir


def update_manifest(experiment_id: str, model_type: str, status: str = "training"):
    """Update experiment manifest."""
    manifest_path = Path("experiments") / "manifest.json"
    
    # Load existing manifest or create new
    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
    else:
        manifest = {"experiments": []}
    
    # Add or update experiment
    experiment = {
        "id": experiment_id,
        "model_type": model_type,
        "status": status,
        "created_at": datetime.now().isoformat(),
        "last_updated": datetime.now().isoformat()
    }
    
    # Check if experiment already exists
    existing = False
    for i, exp in enumerate(manifest["experiments"]):
        if exp["id"] == experiment_id:
            manifest["experiments"][i].update(experiment)
            existing = True
            break
    
    if not existing:
        manifest["experiments"].append(experiment)
    
    # Save manifest
    manifest_path.parent.mkdir(exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    logger = logging.getLogger(__name__)
    metrics_tracker = BestMetricsTracker(exp_dir)
    # Create experiment ID and directory
    experiment_id = create_experiment_id(cfg.model.name)
    exp_dir = setup_experiment_dir(experiment_id, cfg)
    
    logger.info(f"Created experiment: {experiment_id}")
    logger.info(f"Experiment directory: {exp_dir}")
    
    # Update manifest
    update_manifest(experiment_id, cfg.model.name)
    
    pl.seed_everything(cfg.seed)
    if cfg.accelerator == 'auto':
        if torch.cuda.is_available():
            cfg.accelerator = 'gpu'
            cfg.device = 'cuda'
        elif torch.backends.mps.is_available():
            cfg.accelerator = 'mps'
            cfg.device = 'mps'
        else:
            cfg.accelerator = 'cpu'
            cfg.device = 'cpu'

    logger.info(f"Using accelerator: {cfg.accelerator}, device: {cfg.device}")
    
    # Load dataset
    dataset = TrajectoryDataset(Path(cfg.data.output_dir) / 'dataset.npz')
    logger.info(f"Loaded dataset with {len(dataset)} samples")
    
    # Split dataset
    val_size = int(len(dataset) * cfg.training.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    # Create model
    model = TrajectoryLightningModule(cfg)
    
    # Save model parameter count
    param_count = sum(p.numel() for p in model.model.parameters())
    model_info_path = exp_dir / "model_info.json"
    with open(model_info_path, "r") as f:
        model_info = json.load(f)
    model_info["parameters"] = {
        "total": param_count,
        "trainable": sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    }
    with open(model_info_path, "w") as f:
        json.dump(model_info, f, indent=2)
    
    # Callbacks with experiment directory
    checkpoint_callback = ModelCheckpoint(
        dirpath=exp_dir / "checkpoints",
        filename='{epoch:02d}_{val_loss:.4f}',
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        save_last=True,
        verbose=True,
        enable_version_counter=False
    )
    
    # FIXED: Use cfg.training.early_stopping_patience 
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=cfg.training.early_stopping_patience,
        mode='min',
        verbose=True
    )
    
    # Logger with experiment directory
    tb_logger = TensorBoardLogger(
        save_dir=exp_dir / "logs",
        name="tensorboard",
        version=""  # Don't create version subdirectory
    )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        callbacks=[checkpoint_callback, early_stopping, metrics_tracker],
        logger=tb_logger,
        gradient_clip_val=cfg.training.gradient_clip,
        log_every_n_steps=10,
        val_check_interval=1.0,
        enable_progress_bar=True,
        enable_model_summary=True,
        default_root_dir=exp_dir
    )
    
    try:
        # Train
        trainer.fit(model, train_loader, val_loader)
        logger.info(f"Checkpoint callback state:")
        logger.info(f"  best_model_path: {checkpoint_callback.best_model_path}")
        logger.info(f"  best_model_score: {checkpoint_callback.best_model_score}")
        logger.info(f"  best_k_models: {checkpoint_callback.best_k_models}")
        logger.info(f"  monitor: {checkpoint_callback.monitor}")
        
        # Create best_model.ckpt symlink/copy
        if checkpoint_callback.best_model_path:
            best_checkpoint = Path(checkpoint_callback.best_model_path)
            best_model_link = exp_dir / "checkpoints" / "best_model.ckpt"
            
            # Copy the best checkpoint (more portable than symlink)
            if best_checkpoint.exists():
                shutil.copy2(best_checkpoint, best_model_link)
                logger.info(f"Copied best model to: {best_model_link}")
        
        # Update experiment status
        status = "completed"
        
        # Get final metrics from metrics tracker
        best_metrics_file = exp_dir / "best_metrics.json"
        if best_metrics_file.exists():
            with open(best_metrics_file, "r") as f:
                best_metrics = json.load(f)
        else:
            best_metrics = metrics_tracker.best_metrics
        
        final_metrics = {
            "best_val_loss": best_metrics.get('val_loss'),
            "epochs_trained": trainer.current_epoch + 1,
            "best_epoch": best_metrics.get('epoch'),
            "best_metrics": best_metrics  # Include all best metrics
        }
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        status = "failed"
        final_metrics = {"error": str(e)}
    
    # Update model info with final status
    with open(model_info_path, "r") as f:
        model_info = json.load(f)
    
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
    
    # Update manifest with final status
    update_manifest(experiment_id, cfg.model.name, status)
    
    # Save training summary
    summary = {
        "experiment_id": experiment_id,
        "model_type": cfg.model.name,
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
    if 'best_val_loss' in final_metrics and final_metrics['best_val_loss'] is not None:
        logger.info(f"Best validation loss: {final_metrics['best_val_loss']:.4f}")

if __name__ == "__main__":
    main()