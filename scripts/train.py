#!/usr/bin/env python
import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from pathlib import Path
import sys
import logging

sys.path.append(str(Path(__file__).parent.parent))
from ml_mobility_ns3.data.dataset import TrajectoryDataset
from ml_mobility_ns3.training.lightning_module import TrajectoryLightningModule
from torch.utils.data import DataLoader, random_split


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # Get logger inside the function
    logger = logging.getLogger(__name__)
    
    pl.seed_everything(cfg.seed)
    
    # For Mac, disable MPS for now as it can be unstable
    if torch.backends.mps.is_available():
        logger.info("MPS available but using CPU for stability")
        cfg.device = 'cpu'
    
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
        num_workers=0,  # Set to 0 to avoid multiprocessing issues on Mac
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
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.training.checkpoint_dir,
        filename='{epoch:02d}-{val_loss:.4f}',  # Fixed format without redundant prefixes
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        save_last=True,
        verbose=True
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=cfg.training.early_stopping_patience,
        mode='min',
        verbose=True
    )
    
    # Logger
    tb_logger = TensorBoardLogger('logs', name=cfg.model.name)
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        accelerator='cpu',  # Force CPU for now
        devices=1,
        callbacks=[checkpoint_callback, early_stopping],
        logger=tb_logger,
        gradient_clip_val=cfg.training.gradient_clip,
        log_every_n_steps=10,
        val_check_interval=1.0,
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    # Train
    trainer.fit(model, train_loader, val_loader)
    
    # Test with best model
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        logger.info(f"Best model saved at: {best_model_path}")
    

if __name__ == "__main__":
    main()