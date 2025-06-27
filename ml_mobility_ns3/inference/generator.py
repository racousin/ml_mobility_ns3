import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Union

from ..models.vae import TrajectoryVAE
from ..data.preprocessor import TrajectoryPreprocessor


class TrajectoryGenerator:
    """Generate new trajectories using trained VAE."""
    
    def __init__(
        self,
        model: TrajectoryVAE,
        preprocessor: TrajectoryPreprocessor,
        device: Optional[str] = None,
    ):
        self.model = model
        self.preprocessor = preprocessor
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path,
        preprocessor: TrajectoryPreprocessor,
        model_kwargs: dict = {},
        device: Optional[str] = None,
    ):
        """Load generator from checkpoint."""
        # Create model
        model = TrajectoryVAE(**model_kwargs)
        
        # Load checkpoint
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return cls(model, preprocessor, device)
    
    def generate(self, n_samples: int = 1) -> List[np.ndarray]:
        """Generate new trajectories."""
        # Generate from model
        with torch.no_grad():
            trajectories = self.model.generate(n_samples, self.device)
            trajectories = trajectories.cpu().numpy()
        
        # Inverse transform
        trajectories = self.preprocessor.inverse_transform(trajectories)
        
        return trajectories
    
    def interpolate(
        self,
        traj1: Union[np.ndarray, pd.DataFrame],
        traj2: Union[np.ndarray, pd.DataFrame],
        n_steps: int = 10,
    ) -> List[np.ndarray]:
        """Interpolate between two trajectories in latent space."""
        # Preprocess trajectories
        if isinstance(traj1, pd.DataFrame):
            traj1 = self.preprocessor.preprocess_trajectory(traj1)
        if isinstance(traj2, pd.DataFrame):
            traj2 = self.preprocessor.preprocess_trajectory(traj2)
        
        # Transform and prepare for model
        traj1_norm = self.preprocessor.transform([pd.DataFrame(traj1)])[0]
        traj2_norm = self.preprocessor.transform([pd.DataFrame(traj2)])[0]
        
        traj1_tensor = torch.FloatTensor(traj1_norm).unsqueeze(0).to(self.device)
        traj2_tensor = torch.FloatTensor(traj2_norm).unsqueeze(0).to(self.device)
        
        # Encode to latent space
        with torch.no_grad():
            mu1, _ = self.model.encode(traj1_tensor)
            mu2, _ = self.model.encode(traj2_tensor)
            
            # Interpolate in latent space
            interpolated = []
            for alpha in np.linspace(0, 1, n_steps):
                z = (1 - alpha) * mu1 + alpha * mu2
                traj = self.model.decode(z)
                interpolated.append(traj.cpu().numpy()[0])
        
        # Inverse transform
        interpolated = self.preprocessor.inverse_transform(np.array(interpolated))
        
        return interpolated
    
    def reconstruct(self, trajectory: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Reconstruct a trajectory through the VAE."""
        # Preprocess
        if isinstance(trajectory, pd.DataFrame):
            trajectory = self.preprocessor.preprocess_trajectory(trajectory)
        
        # Transform
        traj_norm = self.preprocessor.transform([pd.DataFrame(trajectory)])[0]
        traj_tensor = torch.FloatTensor(traj_norm).unsqueeze(0).to(self.device)
        
        # Reconstruct
        with torch.no_grad():
            recon, _, _ = self.model(traj_tensor)
            recon = recon.cpu().numpy()[0]
        
        # Inverse transform
        recon = self.preprocessor.inverse_transform([recon])[0]
        
        return recon
    
    def to_dataframe(self, trajectory: np.ndarray) -> pd.DataFrame:
        """Convert generated trajectory to DataFrame format."""
        df = pd.DataFrame(trajectory, columns=['LATITUDE', 'LONGITUDE', 'TIME_MINUTES', 'SPEED'])
        
        # Add timestamp (relative to start)
        base_time = pd.Timestamp.now()
        df['UTC DATETIME'] = base_time + pd.to_timedelta(df['TIME_MINUTES'], unit='min')
        
        return df