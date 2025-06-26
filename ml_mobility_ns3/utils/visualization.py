import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import folium
import torch
from typing import List, Optional, Tuple, Union


def plot_trajectory(
    trajectory: Union[pd.DataFrame, np.ndarray],
    ax: Optional[plt.Axes] = None,
    title: str = "Trajectory",
    color: str = 'blue',
    alpha: float = 0.8,
) -> plt.Axes:
    """Plot a single trajectory."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    if isinstance(trajectory, pd.DataFrame):
        x = trajectory['LONGITUDE'].values
        y = trajectory['LATITUDE'].values
    else:
        x = trajectory[:, 1]  # longitude
        y = trajectory[:, 0]  # latitude
    
    ax.plot(x, y, 'o-', color=color, alpha=alpha, markersize=4)
    ax.plot(x[0], y[0], 'go', markersize=8, label='Start')
    ax.plot(x[-1], y[-1], 'ro', markersize=8, label='End')
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_multiple_trajectories(
    trajectories: List[Union[pd.DataFrame, np.ndarray]],
    title: str = "Multiple Trajectories",
    figsize: Tuple[int, int] = (10, 8),
) -> plt.Figure:
    """Plot multiple trajectories on the same plot."""
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(trajectories)))
    
    for i, traj in enumerate(trajectories):
        if isinstance(traj, pd.DataFrame):
            x = traj['LONGITUDE'].values
            y = traj['LATITUDE'].values
        else:
            x = traj[:, 1]
            y = traj[:, 0]
        
        ax.plot(x, y, 'o-', color=colors[i], alpha=0.6, markersize=3)
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_training_history(history: dict, figsize: Tuple[int, int] = (12, 4)) -> plt.Figure:
    """Plot training history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Loss plot
    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train')
    if 'val_loss' in history and history['val_loss']:
        ax1.plot(epochs, history['val_loss'], 'r-', label='Validation')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Log scale loss
    ax2.semilogy(epochs, history['train_loss'], 'b-', label='Train')
    if 'val_loss' in history and history['val_loss']:
        ax2.semilogy(epochs, history['val_loss'], 'r-', label='Validation')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss (log scale)')
    ax2.set_title('Training Loss (Log Scale)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_folium_map(
    trajectories: List[Union[pd.DataFrame, np.ndarray]],
    center: Optional[Tuple[float, float]] = None,
    zoom_start: int = 12,
) -> folium.Map:
    """Create interactive map with trajectories."""
    # Calculate center if not provided
    if center is None:
        all_lats = []
        all_lons = []
        for traj in trajectories:
            if isinstance(traj, pd.DataFrame):
                all_lats.extend(traj['LATITUDE'].values)
                all_lons.extend(traj['LONGITUDE'].values)
            else:
                all_lats.extend(traj[:, 0])
                all_lons.extend(traj[:, 1])
        center = (np.mean(all_lats), np.mean(all_lons))
    
    # Create map
    m = folium.Map(location=center, zoom_start=zoom_start)
    
    # Add trajectories
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'darkred', 'darkblue', 'darkgreen']
    
    for i, traj in enumerate(trajectories):
        if isinstance(traj, pd.DataFrame):
            points = list(zip(traj['LATITUDE'], traj['LONGITUDE']))
        else:
            points = [(lat, lon) for lat, lon in traj[:, :2]]
        
        # Add polyline
        folium.PolyLine(
            points,
            color=colors[i % len(colors)],
            weight=3,
            opacity=0.8,
            popup=f'Trajectory {i+1}'
        ).add_to(m)
        
        # Add start/end markers
        folium.CircleMarker(
            points[0],
            radius=6,
            color='green',
            fill=True,
            popup=f'Start {i+1}'
        ).add_to(m)
        
        folium.CircleMarker(
            points[-1],
            radius=6,
            color='red',
            fill=True,
            popup=f'End {i+1}'
        ).add_to(m)
    
    return m


def plot_latent_space(
    model,
    data_loader,
    device: str = 'cpu',
    n_samples: int = 1000,
) -> plt.Figure:
    """Plot 2D projection of latent space."""
    model.eval()
    latent_codes = []
    
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i * batch[0].size(0) >= n_samples:
                break
            x = batch[0].to(device)
            mu, _ = model.encode(x)
            latent_codes.append(mu.cpu().numpy())
    
    latent_codes = np.vstack(latent_codes)[:n_samples]
    
    # If latent dim > 2, use PCA
    if latent_codes.shape[1] > 2:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        latent_2d = pca.fit_transform(latent_codes)
        explained_var = pca.explained_variance_ratio_
    else:
        latent_2d = latent_codes
        explained_var = [1.0, 0.0]
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(latent_2d[:, 0], latent_2d[:, 1], alpha=0.5, s=20)
    ax.set_xlabel(f'Dim 1 ({explained_var[0]:.1%} var)')
    ax.set_ylabel(f'Dim 2 ({explained_var[1]:.1%} var)')
    ax.set_title('Latent Space Visualization')
    ax.grid(True, alpha=0.3)
    
    return fig