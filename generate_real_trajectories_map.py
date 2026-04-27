#!/usr/bin/env python3
"""
Generate an interactive Folium map of REAL trajectories from interpolated_trips.pkl.
Mirrors generate_trajectories_map.py but uses real data (no model / no scalers needed).
"""

import numpy as np
import pickle
import folium
from pathlib import Path
from typing import List, Dict


def load_real_trajectories(preprocessing_dir: Path, n_per_mode: int = 500, seed: int = 42):
    """Load real trajectories from interpolated_trips.pkl, sampled per transport mode."""
    rng = np.random.default_rng(seed)
    with open(preprocessing_dir / 'interpolated_trips.pkl', 'rb') as f:
        interpolated_trips = pickle.load(f)

    transport_modes = ['BIKE', 'CAR', 'MIXED', 'PUBLIC_TRANSPORT', 'WALKING']
    by_mode = {m: [] for m in transport_modes}
    for trip in interpolated_trips:
        cat = trip.get('category')
        if cat in by_mode:
            by_mode[cat].append(trip)

    sampled = []
    for mode, trips in by_mode.items():
        if len(trips) == 0:
            print(f"  {mode}: 0 trajectories available")
            continue
        if n_per_mode and len(trips) > n_per_mode:
            idx = rng.choice(len(trips), n_per_mode, replace=False)
            trips = [trips[i] for i in idx]
        print(f"  {mode}: {len(trips)} trajectories sampled")

        for trip in trips:
            gps_points = trip['gps_points']  # [timestamp, lat, lon, speed]
            traj = gps_points[:, 1:4].astype(np.float32)  # (lat, lon, speed)
            length = len(traj)
            sampled.append({
                'trajectory': traj,
                'transport_mode': mode,
                'length': length,
                'duration_min': length * 2 / 60,
            })
    return sampled


def create_interactive_map(trajectories: List[Dict], output_file: str = "all_real_trajectories.html"):
    """Create an interactive Folium map of real trajectories, colored by transport mode."""
    mode_colors = {
        'CAR': '#FF0000',
        'WALKING': '#00FF00',
        'BIKE': '#0000FF',
        'PUBLIC_TRANSPORT': '#FF00FF',
        'MIXED': '#FFA500',
    }

    all_lats, all_lons = [], []
    valid_trajectories = []
    for traj_info in trajectories:
        traj = traj_info['trajectory']
        if len(traj) <= 1:
            continue
        lats, lons = traj[:, 0], traj[:, 1]
        if np.all(np.isfinite(lats)) and np.all(np.isfinite(lons)):
            all_lats.extend(lats)
            all_lons.extend(lons)
            valid_trajectories.append(traj_info)

    if not valid_trajectories:
        print("No valid trajectories found!")
        return None

    print(f"Creating map with {len(valid_trajectories)} valid trajectories")
    center_lat = float(np.median(all_lats))
    center_lon = float(np.median(all_lons))
    m = folium.Map(location=[center_lat, center_lon], zoom_start=11)

    mode_counts = {}
    for traj_info in valid_trajectories:
        traj = traj_info['trajectory']
        mode = traj_info['transport_mode']
        duration = traj_info['duration_min']
        mode_counts[mode] = mode_counts.get(mode, 0) + 1

        points = [(lat, lon) for lat, lon in traj[:, :2]]
        color = mode_colors.get(mode, '#808080')

        speeds_kmh = traj[:, 2] * 3.6
        avg_speed = float(np.mean(speeds_kmh))

        lat_diff = np.diff(traj[:, 0])
        lon_diff = np.diff(traj[:, 1])
        total_distance = float(np.sum(np.sqrt(lat_diff**2 + lon_diff**2) * 111))

        popup_text = f"""
        <b>{mode} Trajectory {mode_counts[mode]}</b><br>
        Duration: {duration:.1f} minutes<br>
        Avg Speed: {avg_speed:.1f} km/h<br>
        Total Distance: {total_distance:.1f} km<br>
        Points: {len(points)}
        """

        folium.PolyLine(
            points,
            color=color,
            weight=3,
            opacity=0.7,
            popup=folium.Popup(popup_text, max_width=200),
        ).add_to(m)

    total = sum(mode_counts.values())
    legend_items = [
        f'<span style="color: {mode_colors.get(mode, "#808080")}; font-size: 16px;">●</span> {mode}: {count} trajectories'
        for mode, count in mode_counts.items()
    ]
    legend_html = f'''
    <div style="position: fixed;
                top: 20px; right: 20px; width: 320px; height: auto;
                background-color: white; border: 2px solid grey; z-index: 9999;
                font-size: 14px; padding: 15px; border-radius: 5px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
    <h4 style="margin: 0 0 10px 0; color: #333;">Real Trajectories</h4>
    <p style="margin: 5px 0; font-weight: bold;">Total: {total} trajectories</p>
    <div style="margin: 10px 0;">
    {'<br>'.join(legend_items)}
    </div>
    <p style="margin: 10px 0 0 0; font-size: 12px; color: #666; font-style: italic;">
    Click on any trajectory for details
    </p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    m.save(output_file)
    print(f"Interactive map saved to: {output_file}")

    print("\nReal Trajectory Summary:")
    print("-" * 40)
    for mode, count in mode_counts.items():
        print(f"  {mode}: {count} trajectories")
    print(f"  Total: {total} trajectories")
    return m


def main():
    preprocessing_dir = Path("data/processed")
    output_file = "all_real_trajectories.html"
    n_per_mode = 500  # match the ~2500 total used for the generated map

    print("Loading real trajectories from interpolated_trips.pkl...")
    real_trajectories = load_real_trajectories(preprocessing_dir, n_per_mode=n_per_mode)
    print(f"Loaded {len(real_trajectories)} trajectories total")

    print("Creating interactive map...")
    map_obj = create_interactive_map(real_trajectories, output_file)
    if map_obj:
        print(f"\n✅ Success! Interactive map created: {output_file}")
    else:
        print("❌ Failed to create map - no valid trajectories found.")


if __name__ == "__main__":
    main()
