#!/usr/bin/env python
"""Generate realistic fake NetMob25 dataset for testing."""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import random
from typing import List, Tuple, Dict
import json


# Paris region bounds (approximate)
PARIS_BOUNDS = {
    'lat_min': 48.75, 'lat_max': 48.95,
    'lon_min': 2.20, 'lon_max': 2.50
}

# Common locations in Paris region
LOCATIONS = {
    'home': [
        (48.8566, 2.3522),  # Central Paris
        (48.8924, 2.3444),  # Montmartre
        (48.8372, 2.3550),  # Latin Quarter
        (48.8619, 2.2887),  # Neuilly
        (48.8147, 2.3633),  # Montparnasse
        (48.8975, 2.3837),  # La Villette
        (48.8265, 2.2380),  # Boulogne
        (48.9042, 2.2656),  # La Défense
        (48.8335, 2.3264),  # Place d'Italie
        (48.8799, 2.3573),  # Belleville
    ],
    'work': [
        (48.8698, 2.3087),  # Champs-Élysées
        (48.8606, 2.3376),  # Louvre
        (48.8867, 2.3431),  # Gare du Nord
        (48.8443, 2.3732),  # Gare de Lyon
        (48.9030, 2.2650),  # La Défense
        (48.8534, 2.3488),  # Châtelet
        (48.8738, 2.2950),  # Arc de Triomphe
        (48.8416, 2.3218),  # Montparnasse
    ],
    'shopping': [
        (48.8606, 2.3376),  # Louvre/Rivoli
        (48.8720, 2.3005),  # Champs-Élysées
        (48.8534, 2.3488),  # Châtelet-Les Halles
        (48.8708, 2.3317),  # Opéra
        (48.8417, 2.3219),  # Montparnasse
    ],
    'leisure': [
        (48.8461, 2.3464),  # Luxembourg Gardens
        (48.8867, 2.3431),  # Sacré-Cœur area
        (48.8606, 2.3522),  # Tuileries
        (48.8529, 2.3499),  # Notre-Dame area
        (48.8738, 2.2950),  # Trocadéro
    ]
}

TRANSPORT_MODES = ['PRIV_CAR_DRIVER', 'WALKING', 'BUS', 'SUBWAY', 'BIKE', 'TRAIN']
TRIP_PURPOSES = ['Work', 'Home', 'Shopping', 'Leisure', 'Education', 'Other']


def generate_individuals(n_users: int = 10) -> pd.DataFrame:
    """Generate individuals dataset."""
    individuals = []
    
    departments = [75, 77, 78, 91, 92, 93, 94, 95]
    
    for i in range(n_users):
        user_id = f"10_{1000 + i}"
        
        # Generate demographics
        age = np.random.choice(range(20, 70), p=np.exp(-np.linspace(0, 3, 50))/np.sum(np.exp(-np.linspace(0, 3, 50))))
        sex = np.random.choice(['Woman', 'Man'])
        
        # Home location
        home_idx = i % len(LOCATIONS['home'])
        home_lat, home_lon = LOCATIONS['home'][home_idx]
        
        individual = {
            'ID': user_id,
            'CODGEO': np.random.choice(departments),
            'AREA_NAME': f'Area_{i}',
            'SEX': sex,
            'AGE': age,
            'DIPLOMA': np.random.choice(['BAC', 'BAC+2', 'BAC+5', 'No diploma'], p=[0.3, 0.3, 0.3, 0.1]),
            'PRO_CAT': np.random.choice(range(1, 9)),
            'TYPE_HOUSE': np.random.choice(['Alone', 'Couple', 'Family', 'Shared']),
            'NBPERS_HOUSE': np.random.choice([1, 2, 3, 4], p=[0.3, 0.4, 0.2, 0.1]),
            'PMR': 0,
            'DRIVING_LICENCE': 1 if age > 18 and np.random.random() > 0.3 else 0,
            'NB_CAR': np.random.choice([0, 1, 2], p=[0.3, 0.5, 0.2]),
            'BIKE': np.random.choice([0, 1], p=[0.6, 0.4]),
            'NAVIGO_SUB': np.random.choice([0, 1], p=[0.3, 0.7]),
            'WEIGHT_INDIV': np.random.uniform(1000, 3000),
            'GPS_RECORD': True,
            'home_lat': home_lat,
            'home_lon': home_lon,
        }
        individuals.append(individual)
    
    return pd.DataFrame(individuals)


def generate_trips(individuals_df: pd.DataFrame, n_trips: int = 1000) -> pd.DataFrame:
    """Generate trips dataset."""
    trips = []
    
    # Distribute trips across users (some users travel more)
    user_weights = np.random.exponential(1, len(individuals_df))
    user_weights = user_weights / user_weights.sum()
    
    trip_key = 1000
    
    for _ in range(n_trips):
        # Select user
        user_idx = np.random.choice(len(individuals_df), p=user_weights)
        user = individuals_df.iloc[user_idx]
        
        # Trip timing
        base_date = datetime(2023, 1, 15)
        day_offset = np.random.randint(0, 7)
        trip_date = base_date + timedelta(days=day_offset)
        
        # Time of day (morning/evening peaks)
        p = np.array([0.01, 0.01, 0.01, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10,
                       0.08, 0.06, 0.05, 0.05, 0.05, 0.06, 0.07, 0.08, 0.09,
                       0.06, 0.04, 0.03, 0.02, 0.01, 0.01])
        p = p / p.sum()
        hour = np.random.choice(
            range(24),
            p=p
        )
        minute = np.random.randint(0, 60)
        start_time = trip_date.replace(hour=hour, minute=minute)
        
        # Trip type and destinations
        is_weekday = trip_date.weekday() < 5
        if is_weekday and hour in range(6, 10):
            purpose_o = 'Home'
            purpose_d = 'Work'
            origin = (user['home_lat'], user['home_lon'])
            destination = random.choice(LOCATIONS['work'])
        elif is_weekday and hour in range(16, 20):
            purpose_o = 'Work'
            purpose_d = 'Home'
            origin = random.choice(LOCATIONS['work'])
            destination = (user['home_lat'], user['home_lon'])
        else:
            purpose_o = np.random.choice(['Home', 'Other'])
            purpose_d = np.random.choice(['Shopping', 'Leisure', 'Other', 'Home'])
            
            if purpose_o == 'Home':
                origin = (user['home_lat'], user['home_lon'])
            else:
                origin = random.choice(LOCATIONS.get(purpose_o.lower(), LOCATIONS['leisure']))
            
            if purpose_d == 'Home':
                destination = (user['home_lat'], user['home_lon'])
            else:
                destination = random.choice(LOCATIONS.get(purpose_d.lower(), LOCATIONS['leisure']))
        
        # Duration based on distance and mode
        distance = np.sqrt((origin[0] - destination[0])**2 + (origin[1] - destination[1])**2)
        
        # Select transport mode based on distance
        if distance < 0.01:
            mode = 'WALKING'
            duration = np.random.uniform(5, 15)
        elif distance < 0.03:
            mode = np.random.choice(['WALKING', 'BIKE', 'BUS', 'SUBWAY'], p=[0.3, 0.2, 0.3, 0.2])
            duration = np.random.uniform(10, 30)
        else:
            mode = np.random.choice(['PRIV_CAR_DRIVER', 'BUS', 'SUBWAY', 'TRAIN'], p=[0.3, 0.2, 0.4, 0.1])
            duration = np.random.uniform(20, 60)
        
        end_time = start_time + timedelta(minutes=int(duration))
        
        trip = {
            'KEY': f'TRIP_{trip_key}',
            'ID': user['ID'],
            'Day_EMG': trip_date.strftime('%A'),
            'Date_EMG': trip_date.strftime('%Y-%m-%d'),
            'Day_Type': 'Normal',
            'ID_Trip_Days': 1,
            'No_Traces': 0,
            'No_Trip': 0,
            'Outside_IDF': 0,
            'Area_O': f'Area_{user_idx}',
            'Area_D': f'Area_{np.random.randint(0, 10)}',
            'Code_INSEE_O': 75000 + np.random.randint(1, 21),
            'Code_INSEE_D': 75000 + np.random.randint(1, 21),
            'Zone_O': 'Paris',
            'Zone_D': 'Paris',
            'Date_O': trip_date.strftime('%Y-%m-%d'),
            'Time_O': start_time,
            'Date_D': trip_date.strftime('%Y-%m-%d'),
            'Time_D': end_time,
            'Duration': duration,
            'Purpose_O': purpose_o,
            'Purpose_D': purpose_d,
            'Main_Mode': mode,
            'Mode_1': mode,
            'Weight_Day': np.random.uniform(1000, 3000),
            'origin_lat': origin[0],
            'origin_lon': origin[1],
            'dest_lat': destination[0],
            'dest_lon': destination[1],
        }
        trips.append(trip)
        trip_key += 1
    
    return pd.DataFrame(trips)


def generate_gps_trajectory(
    origin: Tuple[float, float],
    destination: Tuple[float, float],
    start_time: datetime,
    duration_minutes: float,
    mode: str
) -> pd.DataFrame:
    """Generate realistic GPS trajectory between two points."""
    # Number of GPS points (every 2-3 seconds)
    n_points = int(duration_minutes * 60 / 2.5)
    n_points = max(10, n_points)  # At least 10 points
    
    # Speed profiles by mode (km/h)
    speed_profiles = {
        'WALKING': (3, 5, 0.5),  # (min, mean, std)
        'BIKE': (10, 15, 3),
        'PRIV_CAR_DRIVER': (20, 40, 15),
        'BUS': (15, 25, 10),
        'SUBWAY': (25, 35, 10),
        'TRAIN': (40, 60, 15),
    }
    
    min_speed, mean_speed, std_speed = speed_profiles.get(mode, (10, 20, 5))
    
    # Generate path with some randomness
    t = np.linspace(0, 1, n_points)
    
    # Add some curves to the path
    curve_factor = np.sin(t * np.pi * np.random.uniform(1, 3)) * 0.01
    
    # Interpolate between origin and destination with curves
    lats = origin[0] + (destination[0] - origin[0]) * t + curve_factor * np.random.randn()
    lons = origin[1] + (destination[1] - origin[1]) * t + curve_factor * np.random.randn()
    
    # Add some noise to simulate GPS inaccuracy
    lats += np.random.normal(0, 0.0001, n_points)
    lons += np.random.normal(0, 0.0001, n_points)
    
    # Generate speeds
    speeds = np.maximum(min_speed, np.random.normal(mean_speed, std_speed, n_points))
    
    # Convert to m/s
    speeds_ms = speeds / 3.6
    
    # Generate timestamps
    time_deltas = np.random.uniform(2, 3, n_points)
    time_deltas[0] = 0
    timestamps = start_time + pd.to_timedelta(np.cumsum(time_deltas), unit='s')
    
    gps_df = pd.DataFrame({
        'UTC_TIMESTAMP': timestamps,
        'LOCAL_TIMESTAMP': timestamps,
        'LATITUDE': lats,
        'LONGITUDE': lons,
        'VALID': 'SPS',
        'SPEED': speeds_ms
    })
    
    return gps_df


def main():
    """Generate complete fake dataset."""
    # Create output directory
    output_dir = Path("data/fake_netmob25")
    output_dir.mkdir(parents=True, exist_ok=True)
    gps_dir = output_dir / "gps"
    gps_dir.mkdir(exist_ok=True)
    
    print("Generating fake NetMob25 dataset...")
    
    # Generate individuals
    print("Creating individuals...")
    individuals_df = generate_individuals(n_users=10)
    individuals_df.to_csv(output_dir / "individuals.csv", index=False)
    print(f"Created {len(individuals_df)} individuals")
    
    # Generate trips
    print("Creating trips...")
    trips_df = generate_trips(individuals_df, n_trips=1000)
    trips_df.to_csv(output_dir / "trips.csv", index=False)
    print(f"Created {len(trips_df)} trips")
    
    # Generate GPS traces for each user
    print("Creating GPS traces...")
    for user_id in individuals_df['ID'].unique():
        user_trips = trips_df[trips_df['ID'] == user_id]
        
        if len(user_trips) == 0:
            continue
        
        gps_traces = []
        for _, trip in user_trips.iterrows():
            origin = (trip['origin_lat'], trip['origin_lon'])
            destination = (trip['dest_lat'], trip['dest_lon'])
            
            trajectory = generate_gps_trajectory(
                origin=origin,
                destination=destination,
                start_time=trip['Time_O'],
                duration_minutes=trip['Duration'],
                mode=trip['Main_Mode']
            )
            gps_traces.append(trajectory)
        
        if gps_traces:
            user_gps = pd.concat(gps_traces, ignore_index=True)
            user_gps = user_gps.sort_values('UTC_TIMESTAMP')
            user_gps.to_csv(gps_dir / f"{user_id}.csv", index=False)
            print(f"Created GPS trace for user {user_id}: {len(user_gps)} points")
    
    # Create summary
    summary = {
        'n_individuals': len(individuals_df),
        'n_trips': len(trips_df),
        'date_range': {
            'start': trips_df['Date_EMG'].min(),
            'end': trips_df['Date_EMG'].max()
        },
        'transport_modes': trips_df['Main_Mode'].value_counts().to_dict(),
        'trip_purposes': trips_df['Purpose_D'].value_counts().to_dict(),
        'avg_trip_duration': trips_df['Duration'].mean(),
        'users_with_gps': len(list(gps_dir.glob("*.csv")))
    }
    
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nDataset created successfully in {output_dir}")
    print(f"Summary saved to {output_dir}/summary.json")
    
    # Print summary
    print("\nDataset Summary:")
    print(f"- Individuals: {summary['n_individuals']}")
    print(f"- Trips: {summary['n_trips']}")
    print(f"- Average trip duration: {summary['avg_trip_duration']:.1f} minutes")
    print(f"- Users with GPS traces: {summary['users_with_gps']}")
    print("\nTransport modes:")
    for mode, count in summary['transport_modes'].items():
        print(f"  - {mode}: {count}")


if __name__ == "__main__":
    main()