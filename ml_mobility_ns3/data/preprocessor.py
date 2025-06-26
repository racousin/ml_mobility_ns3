import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from sklearn.preprocessing import StandardScaler


class TrajectoryPreprocessor:
    """Simple preprocessor for trajectory data."""
    
    def __init__(self, sequence_length: int = 50):
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()
        self.fitted = False
        
    def preprocess_trajectory(self, df: pd.DataFrame) -> np.ndarray:
        """Convert GPS trajectory to numpy array."""
        # Extract features: lat, lon, time_diff, speed
        features = []
        
        df = df.sort_values('UTC_TIMESTAMP').reset_index(drop=True)
        
        for i in range(len(df)):
            row = df.iloc[i]
            feat = [
                row['LATITUDE'],
                row['LONGITUDE'],
            ]
            
            # Add time difference from start (in minutes)
            if i == 0:
                feat.append(0.0)
            else:
                time_diff = (df.iloc[i]['UTC_TIMESTAMP'] - df.iloc[0]['UTC_TIMESTAMP']).total_seconds() / 60
                feat.append(time_diff)
            
            # Add speed
            speed = row.get('SPEED', 0.0)
            feat.append(speed if pd.notna(speed) else 0.0)
            
            features.append(feat)
            
        return np.array(features)
    
    def pad_or_truncate(self, trajectory: np.ndarray) -> np.ndarray:
        """Pad or truncate trajectory to fixed length."""
        if len(trajectory) > self.sequence_length:
            # Take evenly spaced points
            indices = np.linspace(0, len(trajectory) - 1, self.sequence_length, dtype=int)
            return trajectory[indices]
        elif len(trajectory) < self.sequence_length:
            # Pad with last position
            padding = np.repeat(trajectory[-1:], self.sequence_length - len(trajectory), axis=0)
            return np.vstack([trajectory, padding])
        return trajectory
    
    def fit(self, trajectories: List[pd.DataFrame]):
        """Fit the scaler on trajectory data."""
        all_features = []
        for traj_df in trajectories:
            features = self.preprocess_trajectory(traj_df)
            all_features.append(features)
        
        all_features = np.vstack(all_features)
        self.scaler.fit(all_features)
        self.fitted = True
        
    def transform(self, trajectories: List[pd.DataFrame]) -> np.ndarray:
        """Transform trajectories to model input format."""
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        processed = []
        for traj_df in trajectories:
            # Convert to features
            features = self.preprocess_trajectory(traj_df)
            # Normalize
            features = self.scaler.transform(features)
            # Pad/truncate
            features = self.pad_or_truncate(features)
            processed.append(features)
            
        return np.array(processed)
    
    def inverse_transform(self, trajectories: np.ndarray) -> List[np.ndarray]:
        """Convert normalized trajectories back to original scale."""
        result = []
        for traj in trajectories:
            # Remove padding (where all values are the same)
            mask = np.any(np.diff(traj, axis=0) != 0, axis=1)
            if np.any(mask):
                last_unique = np.where(mask)[0][-1] + 2
                traj = traj[:last_unique]
            
            # Inverse transform
            traj = self.scaler.inverse_transform(traj)
            result.append(traj)
            
        return result