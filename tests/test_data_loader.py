import pytest
from pathlib import Path
import pandas as pd
import numpy as np

from ml_mobility_ns3 import NetMob25Loader


def test_loader_initialization():
    """Test loader can be initialized."""
    loader = NetMob25Loader(Path("dummy/path"))
    assert loader.data_dir == Path("dummy/path")
    assert loader.individuals_df is None
    assert loader.trips_df is None


def test_preprocess_trajectory():
    """Test trajectory preprocessing."""
    from ml_mobility_ns3 import TrajectoryPreprocessor
    
    # Create dummy trajectory
    dummy_traj = pd.DataFrame({
        'UTC_TIMESTAMP': pd.date_range('2023-01-01', periods=10, freq='1min'),
        'LATITUDE': np.linspace(48.8, 48.9, 10),
        'LONGITUDE': np.linspace(2.3, 2.4, 10),
        'SPEED': np.random.rand(10) * 10
    })
    
    preprocessor = TrajectoryPreprocessor(sequence_length=20)
    features = preprocessor.preprocess_trajectory(dummy_traj)
    
    assert features.shape == (10, 4)  # 10 points, 4 features
    assert features[0, 2] == 0.0  # First time diff should be 0