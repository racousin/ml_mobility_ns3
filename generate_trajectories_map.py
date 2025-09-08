#!/usr/bin/env python3
"""
Simple script to generate trajectories from a trained model and create an interactive map.
Based on the analysis from notebooks/Untitled.py but simplified for the specific experiment.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import json
import pickle
import folium
from typing import List, Dict
import sys

# Add project to path
sys.path.append('.')

# Import the model from the project
from ml_mobility_ns3.models.vae_lstm import ConditionalTrajectoryVAE

def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'cpu'  # Use CPU on Mac to avoid MPS issues
    else:
        return 'cpu'

def load_model_from_experiment(experiment_path: Path, device: str = None):
    """Load model from Lightning experiment checkpoint."""
    if device is None:
        device = get_device()
    
    # Load model info
    with open(experiment_path / 'model_info.json', 'r') as f:
        model_info = json.load(f)
    
    # Extract model architecture config
    arch_config = model_info['architecture']
    
    # Create model instance
    model = ConditionalTrajectoryVAE(
        input_dim=arch_config['input_dim'],
        hidden_dim=arch_config['hidden_dim'],
        latent_dim=arch_config['latent_dim'],
        num_layers=arch_config['num_layers'],
        condition_dim=arch_config['condition_dim'],
        dropout=arch_config['dropout']
    )
    
    # Load checkpoint
    checkpoint_path = experiment_path / 'checkpoints' / 'best_model.ckpt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Extract model state dict from Lightning checkpoint
    model_state_dict = checkpoint['state_dict']
    
    # Remove 'model.' prefix from keys (Lightning wrapper adds this)
    cleaned_state_dict = {}
    for key, value in model_state_dict.items():
        if key.startswith('model.'):
            cleaned_key = key[6:]  # Remove 'model.' prefix
            cleaned_state_dict[cleaned_key] = value
    
    # Load weights
    model.load_state_dict(cleaned_state_dict)
    model = model.to(device)
    model.eval()
    
    return model, model_info

def load_scalers(preprocessing_dir: Path):
    """Load the scalers dictionary."""
    scaler_path = preprocessing_dir / 'scalers.pkl'
    with open(scaler_path, 'rb') as f:
        return pickle.load(f)

def generate_sample_trajectories(model, device: str, n_samples: int = 50):
    """Generate a diverse sample of trajectories for visualization."""
    transport_modes = ['BIKE', 'CAR', 'MIXED', 'PUBLIC_TRANSPORT', 'WALKING']
    
    all_trajectories = []
    
    for mode_name in transport_modes:
        mode_idx = transport_modes.index(mode_name)
        
        # Generate various trajectory lengths (5-60 minutes)
        durations_min = np.random.uniform(5, 60, n_samples // len(transport_modes))
        trip_lengths = (durations_min * 60 / 2).astype(int)  # Convert to steps (2 sec per step)
        trip_lengths = np.clip(trip_lengths, 10, 2000)  # Clip to model limits
        
        # Create tensors
        n_mode_samples = len(trip_lengths)
        mode_tensor = torch.full((n_mode_samples,), mode_idx, dtype=torch.long).to(device)
        length_tensor = torch.tensor(trip_lengths, dtype=torch.long).to(device)
        
        # Prepare conditions dictionary for generate method
        conditions = {
            'transport_mode': mode_tensor,
            'length': length_tensor
        }
        
        # Generate trajectories
        with torch.no_grad():
            trajectories = model.generate(conditions, n_mode_samples)
        
        # Store with metadata
        for i, (traj, length) in enumerate(zip(trajectories.cpu().numpy(), trip_lengths)):
            all_trajectories.append({
                'trajectory': traj,
                'transport_mode': mode_name,
                'length': length,
                'duration_min': length * 2 / 60
            })
    
    return all_trajectories

def inverse_transform_trajectories(trajectories, scalers):
    """Convert normalized trajectories back to real coordinates."""
    trajectory_scaler = scalers['trajectory']
    
    transformed_trajectories = []
    for traj_info in trajectories:
        traj = traj_info['trajectory']
        length = traj_info['length']
        
        # Get valid portion and reshape for scaler
        valid_traj = traj[:length]
        
        # Transform back to real coordinates
        real_traj = trajectory_scaler.inverse_transform(valid_traj)
        
        # Update trajectory info
        traj_info_copy = traj_info.copy()
        traj_info_copy['trajectory'] = real_traj
        transformed_trajectories.append(traj_info_copy)
    
    return transformed_trajectories

def create_interactive_map(trajectories: List[Dict], output_file: str = "all_generated_trajectories3.html"):
    """Create an interactive Folium map with all generated trajectories."""
    
    # Define colors for each transport mode
    mode_colors = {
        'CAR': '#FF0000',        # Red
        'WALKING': '#00FF00',    # Green  
        'BIKE': '#0000FF',       # Blue
        'PUBLIC_TRANSPORT': '#FF00FF',  # Magenta
        'MIXED': '#FFA500'       # Orange
    }
    
    # Collect all coordinates for centering
    all_lats = []
    all_lons = []
    
    valid_trajectories = []
    for traj_info in trajectories:
        traj = traj_info['trajectory']
        
        # Check if trajectory has valid coordinates
        if len(traj) > 1:
            lats = traj[:, 0]
            lons = traj[:, 1]
            
            # Filter out unrealistic coordinates (basic sanity check)
            if np.all(np.isfinite(lats)) and np.all(np.isfinite(lons)):
                all_lats.extend(lats)
                all_lons.extend(lons)
                valid_trajectories.append(traj_info)
    
    if not valid_trajectories:
        print("No valid trajectories found!")
        return None
    
    print(f"Creating map with {len(valid_trajectories)} valid trajectories")
    
    # Calculate map center
    center_lat = np.median(all_lats)
    center_lon = np.median(all_lons)
    
    # Create map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=11)
    
    # Group trajectories by mode for statistics
    mode_counts = {}
    
    # Add trajectories to map
    for i, traj_info in enumerate(valid_trajectories):
        traj = traj_info['trajectory']
        mode = traj_info['transport_mode']
        duration = traj_info['duration_min']
        
        # Count trajectories per mode
        if mode not in mode_counts:
            mode_counts[mode] = 0
        mode_counts[mode] += 1
        
        # Create coordinate pairs for folium
        points = [(lat, lon) for lat, lon in traj[:, :2]]
        
        # Get color for this mode
        color = mode_colors.get(mode, '#808080')
        
        # Calculate trajectory statistics for popup
        speeds_ms = traj[:, 2]
        speeds_kmh = speeds_ms * 3.6
        avg_speed = np.mean(speeds_kmh)
        
        # Distance calculation
        if len(traj) > 1:
            lat_diff = np.diff(traj[:, 0])
            lon_diff = np.diff(traj[:, 1])
            distances = np.sqrt(lat_diff**2 + lon_diff**2) * 111  # Approximate km
            total_distance = np.sum(distances)
        else:
            total_distance = 0
        
        # Create popup info
        popup_text = f"""
        <b>{mode} Trajectory {mode_counts[mode]}</b><br>
        Duration: {duration:.1f} minutes<br>
        Avg Speed: {avg_speed:.1f} km/h<br>
        Total Distance: {total_distance:.1f} km<br>
        Points: {len(points)}
        """
        
        # Add trajectory to map
        folium.PolyLine(
            points,
            color=color,
            weight=3,
            opacity=0.7,
            popup=folium.Popup(popup_text, max_width=200)
        ).add_to(m)
    
    # Create legend
    legend_items = []
    total_trajectories = sum(mode_counts.values())
    
    for mode, count in mode_counts.items():
        color = mode_colors.get(mode, '#808080')
        legend_items.append(f'<span style="color: {color}; font-size: 16px;">●</span> {mode}: {count} trajectories')
    
    legend_html = f'''
    <div style="position: fixed; 
                top: 20px; right: 20px; width: 320px; height: auto;
                background-color: white; border: 2px solid grey; z-index: 9999; 
                font-size: 14px; padding: 15px; border-radius: 5px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
    <h4 style="margin: 0 0 10px 0; color: #333;">Generated Trajectories</h4>
    <p style="margin: 5px 0; font-weight: bold;">Total: {total_trajectories} trajectories</p>
    <div style="margin: 10px 0;">
    {'<br>'.join(legend_items)}
    </div>
    <p style="margin: 10px 0 0 0; font-size: 12px; color: #666; font-style: italic;">
    Click on any trajectory for details
    </p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Save map
    m.save(output_file)
    print(f"Interactive map saved to: {output_file}")
    
    # Print summary statistics
    print(f"\nGenerated Trajectory Summary:")
    print("-" * 40)
    for mode, count in mode_counts.items():
        print(f"  {mode}: {count} trajectories")
    print(f"  Total: {total_trajectories} trajectories")
    
    return m

def main():
    """Main execution function."""
    # Set paths
    experiment_path = Path("experiments/vae_lstm_2025-09-07_20-55-25")
    preprocessing_dir = Path("data/processed")
    output_file = "all_generated_trajectories3.html"
    
    # Setup device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model from experiment...")
    model, model_info = load_model_from_experiment(experiment_path, device)
    
    # Load scalers
    print("Loading data scalers...")
    scalers = load_scalers(preprocessing_dir)
    
    # Generate sample trajectories
    print("Generating sample trajectories...")
    n_samples = 2500  # Total trajectories to generate
    normalized_trajectories = generate_sample_trajectories(model, device, n_samples)
    print(f"Generated {len(normalized_trajectories)} normalized trajectories")
    
    # Transform back to real coordinates
    print("Converting to real coordinates...")
    real_trajectories = inverse_transform_trajectories(normalized_trajectories, scalers)
    
    # Create interactive map
    print("Creating interactive map...")
    map_obj = create_interactive_map(real_trajectories, output_file)
    
    if map_obj:
        print(f"\n✅ Success! Interactive map created: {output_file}")
        print("Open the HTML file in your browser to view the trajectories.")
    else:
        print("❌ Failed to create map - no valid trajectories found.")

if __name__ == "__main__":
    main()