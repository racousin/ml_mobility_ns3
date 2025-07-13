# ml_mobility_ns3/utils/experiment_utils.py
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Tuple
import re
import logging

logger = logging.getLogger(__name__)


class ExperimentManager:
    """Centralized experiment management utilities."""
    
    @staticmethod
    def find_experiment_dir(experiment_id: Optional[str] = None, 
                          experiments_dir: Path = Path("experiments")) -> Optional[Path]:
        """Find experiment directory by ID or get the latest."""
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
    
    @staticmethod
    def find_best_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
        """Find the best checkpoint in a directory."""
        # First try best_model.ckpt
        best_model_path = checkpoint_dir / 'best_model.ckpt'
        if best_model_path.exists():
            return best_model_path
        
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
                return checkpoint_losses[0][0]
        
        # Fallback to last checkpoint
        last_path = checkpoint_dir / 'last.ckpt'
        if last_path.exists():
            return last_path
        
        return None
    
    @staticmethod
    def create_experiment_id(model_name: str, hydra_config: Optional[Dict] = None) -> str:
        """Create unique experiment ID with optional hydra job info."""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # Add hydra job number if in multirun
        if hydra_config and 'job' in hydra_config:
            job_num = hydra_config['job'].num
            job_id = hydra_config['job'].id
            return f"{model_name}_{timestamp}_job{job_num}_{job_id}"
        
        return f"{model_name}_{timestamp}"
    
    @staticmethod
    def update_manifest(experiment_id: str, model_type: str, status: str = "training",
                       manifest_path: Path = Path("experiments/manifest.json")):
        """Update experiment manifest."""
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
    
    @staticmethod
    def load_experiment_info(exp_dir: Path) -> Tuple[Optional[Dict], Optional[Dict]]:
        """Load model info and config from experiment directory."""
        model_info = None
        config = None
        
        model_info_path = exp_dir / "model_info.json"
        if model_info_path.exists():
            with open(model_info_path, "r") as f:
                model_info = json.load(f)
        
        config_path = exp_dir / "config.yaml"
        if config_path.exists():
            from omegaconf import OmegaConf
            config = OmegaConf.load(config_path)
        
        return model_info, config