"""Quick example of using the trajectory generator package."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ml_mobility_ns3 import (
    NetMob25Loader,
    TrajectoryPreprocessor,
    TrajectoryVAE,
    VAETrainer,
    TrajectoryGenerator,
)
from ml_mobility_ns3.utils.visualization import plot_multiple_trajectories


def create_dummy_trajectories(n_trajectories=100, n_points=50):
    """Create dummy trajectories for testing."""
    trajectories = []
    
    for _ in range(n_trajectories):
        # Random walk trajectory
        t = np.linspace(0, 10, n_points)
        lat = 48.8566 + np.cumsum(np.random.randn(n_points) * 0.001)
        lon = 2.3522 + np.cumsum(np.random.randn(n_points) * 0.001)
        speed = np.abs(np.random.randn(n_points) * 5 + 10)
        
        df = pd.DataFrame({
            'UTC DATETIME': pd.date_range('2023-01-01', periods=n_points, freq='1min'),
            'LATITUDE': lat,
            'LONGITUDE': lon,
            'SPEED': speed,
            'VALID': 'SPS'
        })
        trajectories.append(df)
    
    return trajectories


def main():
    print("Creating dummy data...")
    trajectories = create_dummy_trajectories(n_trajectories=200)
    
    # Split data
    train_trajectories = trajectories[:160]
    val_trajectories = trajectories[160:]
    
    print("Preprocessing data...")
    preprocessor = TrajectoryPreprocessor(sequence_length=50)
    preprocessor.fit(train_trajectories)
    
    train_data = preprocessor.transform(train_trajectories)
    val_data = preprocessor.transform(val_trajectories)
    print(f"Train shape: {train_data.shape}, Val shape: {val_data.shape}")
    
    print("Creating model...")
    model = TrajectoryVAE(
        input_dim=4,
        sequence_length=50,
        hidden_dim=64,
        latent_dim=16,
        num_layers=1,
    )
    
    print("Training...")
    trainer = VAETrainer(model, learning_rate=1e-3, beta=0.5)
    trainer.fit(
        train_data,
        val_data,
        epochs=20,
        batch_size=32,
    )
    
    print("\nGenerating new trajectories...")
    generator = TrajectoryGenerator(model, preprocessor)
    new_trajectories = generator.generate(n_samples=5)
    print(new_trajectories)


    # Plot
    print("Plotting results...")
    fig = plot_multiple_trajectories(
        [generator.to_dataframe(traj) for traj in new_trajectories],
        title="Generated Trajectories"
    )
    plt.savefig("generated_trajectories.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    # Plot training history
    from ml_mobility_ns3.utils.visualization import plot_training_history
    fig = plot_training_history(trainer.history)
    plt.savefig("training_history.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Done!")


if __name__ == "__main__":
    main()