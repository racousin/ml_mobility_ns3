#!/usr/bin/env python3
import torch
import numpy as np
import folium
import json
import pickle
from pathlib import Path
from typing import List, Dict
import sys
import argparse

# Add project to path
sys.path.append('.')

from ml_mobility_ns3.training.lightning_module import TrajectoryLightningModule
from ml_mobility_ns3.utils.model_utils import load_checkpoint
from ml_mobility_ns3.utils.experiment_utils import ExperimentManager

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def load_scalers(preprocessing_dir: Path):
    scaler_path = preprocessing_dir / 'scalers.pkl'
    with open(scaler_path, 'rb') as f:
        return pickle.load(f)

def generate_sample_trajectories(model, device: str, n_samples: int = 50):
    transport_modes = ['BIKE', 'CAR', 'MIXED', 'PUBLIC_TRANSPORT', 'WALKING']
    all_trajectories = []
    samples_per_mode = n_samples // len(transport_modes)
    
    for mode_idx, mode_name in enumerate(transport_modes):
        trip_lengths = np.random.randint(300, 1000, samples_per_mode)
        mode_tensor = torch.full((samples_per_mode,), mode_idx, dtype=torch.long).to(device)
        length_tensor = torch.tensor(trip_lengths, dtype=torch.long).to(device)
        
        conditions = {'transport_mode': mode_tensor, 'length': length_tensor}
        
        with torch.no_grad():
            # We access .model because TrajectoryLightningModule wraps the actual model
            trajectories = model.model.generate(conditions, samples_per_mode, target_length=2000)
        
        for i, (traj, length) in enumerate(zip(trajectories.cpu().numpy(), trip_lengths)):
            all_trajectories.append({
                'trajectory': traj,
                'transport_mode': mode_name,
                'length': length,
                'duration_min': length * 2 / 60
            })
    return all_trajectories

def inverse_transform_trajectories(trajectories, scalers):
    trajectory_scaler = scalers['trajectory']
    transformed_trajectories = []
    for traj_info in trajectories:
        traj = traj_info['trajectory']
        length = traj_info['length']
        valid_traj = traj[:length]
        real_traj = trajectory_scaler.inverse_transform(valid_traj)
        
        traj_info_copy = traj_info.copy()
        traj_info_copy['trajectory'] = real_traj
        transformed_trajectories.append(traj_info_copy)
    return transformed_trajectories

def create_interactive_map(trajectories: List[Dict], output_file: str):
    mode_colors = {
        'CAR': '#FF0000', 'WALKING': '#00FF00', 'BIKE': '#0000FF',
        'PUBLIC_TRANSPORT': '#FF00FF', 'MIXED': '#FFA500'
    }
    
    all_lats, all_lons = [], []
    valid_trajectories = []
    for traj_info in trajectories:
        traj = traj_info['trajectory']
        if len(traj) > 1:
            lats, lons = traj[:, 0], traj[:, 1]
            if np.all(np.isfinite(lats)) and np.all(np.isfinite(lons)):
                all_lats.extend(lats)
                all_lons.extend(lons)
                valid_trajectories.append(traj_info)
    
    if not valid_trajectories:
        print("No valid trajectories found!")
        return
    
    center_lat, center_lon = np.median(all_lats), np.median(all_lons)
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
    
    mode_counts = {}
    for traj_info in valid_trajectories:
        traj, mode, duration = traj_info['trajectory'], traj_info['transport_mode'], traj_info['duration_min']
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
        
        points = [(lat, lon) for lat, lon in traj[:, :2]]
        color = mode_colors.get(mode, '#808080')
        avg_speed = np.mean(traj[:, 2]) * 3.6
        
        lat_diff, lon_diff = np.diff(traj[:, 0]), np.diff(traj[:, 1])
        total_dist = np.sum(np.sqrt(lat_diff**2 + lon_diff**2) * 111)
        
        popup_text = f"<b>{mode}</b><br>Duration: {duration:.1f}m<br>Speed: {avg_speed:.1f}km/h<br>Dist: {total_dist:.1f}km"
        folium.PolyLine(points, color=color, weight=2, opacity=0.6, popup=folium.Popup(popup_text, max_width=200)).add_to(m)
    
    m.save(output_file)
    print(f"Map saved to {output_file}. Summary: {mode_counts}")

def main():
    parser = argparse.ArgumentParser()
    parser.get_default('exp_id')
    parser.add_argument('--exp_id', type=str, required=True)
    parser.add_argument('--n_samples', type=int, default=500)
    args = parser.parse_args()
    
    exp_manager = ExperimentManager()
    exp_dir = exp_manager.find_experiment_dir(args.exp_id)
    if not exp_dir:
        print(f"Experiment {args.exp_id} not found"); return
        
    # Load model info and config
    model_info, cfg = exp_manager.load_experiment_info(exp_dir)
    if cfg is None:
        print(f"Could not load config for {args.exp_id}")
        return
    
    checkpoint_path = exp_manager.find_best_checkpoint(exp_dir / "checkpoints")
    
    device = get_device()
    model = load_checkpoint(checkpoint_path, cfg, TrajectoryLightningModule, device=device)
    scalers = load_scalers(Path("data/processed"))
    
    print(f"Generating {args.n_samples} trajectories...")
    norm_trajs = generate_sample_trajectories(model, device, args.n_samples)
    real_trajs = inverse_transform_trajectories(norm_trajs, scalers)
    
    output_file = f"map_{args.exp_id}.html"
    create_interactive_map(real_trajs, output_file)

if __name__ == "__main__":
    main()
