import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import pickle
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


class NetMob25TrajectoryPreprocessor:
    """
    Preprocessor for NetMob25 dataset to prepare data for trajectory generation models.
    
    This version tracks filtering statistics at each step and uses improved
    categorization and scaling methods.
    """
    
    def __init__(self, data_dir='../data/netmob25/', output_dir='../preprocessing/'):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # IDF bounding box
        self.idf_bounds = {
            'lat_min': 48.21,  # South boundary
            'lat_max': 49.24,  # North boundary  
            'lon_min': 1.45,   # West boundary
            'lon_max': 3.55    # East boundary
        }
        
        # Define modes to filter out
        self.excluded_modes = ['OTHER', 'PLANE', 'LIGHT_COMM_VEHICLE', 'ON_DEMAND']
        
        # Transport mode categories with better names
        self.mode_categories = {
            'ROAD_DRIVER': ['TWO_WHEELER', 'PRIV_CAR_DRIVER'],
            'ROAD_PASSENGER': ['PRIV_CAR_PASSENGER', 'TAXI'],
            'PUBLIC_TRANSPORT': ['BUS', 'TRAIN', 'TRAIN_EXPRESS', 'SUBWAY', 'TRAMWAY'],
            'WALKING': ['WALKING'],
            'MICRO_MOBILITY': ['BIKE', 'ELECT_BIKE', 'ELECT_SCOOTER']
        }
        
        # Create reverse mapping
        self.mode_to_category = {}
        for category, modes in self.mode_categories.items():
            for mode in modes:
                self.mode_to_category[mode] = category
        
        # Filtering statistics
        self.filtering_stats = []
        
        # Scalers
        self.scalers = {}
        
    def compute_trip_statistics(self, trips_df, group_by=None, label=""):
        """Compute statistics for trips"""
        stats = {}
        
        if group_by:
            grouped = trips_df.groupby(group_by)
            
            for name, group in grouped:
                if len(group) > 0:
                    stats[f"{label}_{name}"] = {
                        'nb_trips': len(group),
                        'sum_weight': group['Weight_Day'].sum() if 'Weight_Day' in group.columns else len(group),
                        'distance_avg': group['distance_km'].mean() if 'distance_km' in group.columns else np.nan,
                        'distance_std': group['distance_km'].std() if 'distance_km' in group.columns else np.nan,
                        'speed_avg': group['speed_kmh'].mean() if 'speed_kmh' in group.columns else np.nan,
                        'speed_std': group['speed_kmh'].std() if 'speed_kmh' in group.columns else np.nan,
                        'duration_avg': group['Duration'].mean() if 'Duration' in group.columns else np.nan,
                        'duration_std': group['Duration'].std() if 'Duration' in group.columns else np.nan
                    }
        else:
            stats[label] = {
                'nb_trips': len(trips_df),
                'sum_weight': trips_df['Weight_Day'].sum() if 'Weight_Day' in trips_df.columns else len(trips_df),
                'distance_avg': trips_df['distance_km'].mean() if 'distance_km' in trips_df.columns else np.nan,
                'distance_std': trips_df['distance_km'].std() if 'distance_km' in trips_df.columns else np.nan,
                'speed_avg': trips_df['speed_kmh'].mean() if 'speed_kmh' in trips_df.columns else np.nan,
                'speed_std': trips_df['speed_kmh'].std() if 'speed_kmh' in trips_df.columns else np.nan,
                'duration_avg': trips_df['Duration'].mean() if 'Duration' in trips_df.columns else np.nan,
                'duration_std': trips_df['Duration'].std() if 'Duration' in trips_df.columns else np.nan
            }
        
        return stats
    
    def log_filtering_step(self, step_name, before_df, after_df):
        """Log statistics for filtering step"""
        nb_removed = len(before_df) - len(after_df)
        weight_before = before_df['Weight_Day'].sum() if 'Weight_Day' in before_df.columns else len(before_df)
        weight_after = after_df['Weight_Day'].sum() if 'Weight_Day' in after_df.columns else len(after_df)
        weight_removed = weight_before - weight_after
        
        stats = {
            'step': step_name,
            'trips_before': len(before_df),
            'trips_after': len(after_df),
            'trips_removed': nb_removed,
            'trips_removed_pct': (nb_removed / len(before_df) * 100) if len(before_df) > 0 else 0,
            'weight_before': weight_before,
            'weight_after': weight_after,
            'weight_removed': weight_removed,
            'weight_removed_pct': (weight_removed / weight_before * 100) if weight_before > 0 else 0
        }
        
        self.filtering_stats.append(stats)
        
        print(f"\n{step_name}:")
        print(f"  Trips removed: {nb_removed:,} ({stats['trips_removed_pct']:.1f}%)")
        print(f"  Weight removed: {weight_removed:,.0f} ({stats['weight_removed_pct']:.1f}%)")
        print(f"  Remaining trips: {len(after_df):,}")
    
    def load_and_filter_data(self):
        """Step 1: Load and apply sequential filters with statistics tracking"""
        print("Loading data...")
        
        # Load individuals
        individuals_df = pd.read_csv(self.data_dir / 'individuals_dataset.csv')
        individuals_filtered = individuals_df[individuals_df['GPS_RECORD'] == 1].copy()
        print(f"Individuals with GPS: {len(individuals_filtered)} / {len(individuals_df)}")
        
        # Load trips
        trips_df = pd.read_csv(self.data_dir / 'trips_dataset.csv')
        print(f"Initial trips: {len(trips_df)}")
        
        # Add Weight_Day column if not present (for testing)
        if 'Weight_Day' not in trips_df.columns:
            trips_df['Weight_Day'] = 1.0
        
        # Initial statistics
        initial_stats = self.compute_trip_statistics(trips_df, label="initial")
        
        # Filter 1: Missing date/time values
        trips_filtered = trips_df.copy()
        mask_complete_times = trips_filtered[['Date_O', 'Time_O', 'Date_D', 'Time_D']].notnull().all(axis=1)
        trips_after_filter1 = trips_filtered[mask_complete_times].copy()
        self.log_filtering_step("Filter 1: Missing date/time values", trips_filtered, trips_after_filter1)
        
        # Parse dates and times for further filtering
        trips_after_filter1['datetime_O'] = pd.to_datetime(
            trips_after_filter1['Date_O'] + ' ' + trips_after_filter1['Time_O'],
            format='%Y-%m-%d %H:%M:%S'
        )
        trips_after_filter1['datetime_D'] = pd.to_datetime(
            trips_after_filter1['Date_D'] + ' ' + trips_after_filter1['Time_D'],
            format='%Y-%m-%d %H:%M:%S'
        )
        
        # Filter 3: Trips < 30 seconds
        trips_after_filter1['duration_seconds'] = (
            trips_after_filter1['datetime_D'] - trips_after_filter1['datetime_O']
        ).dt.total_seconds()
        mask_duration_min = trips_after_filter1['duration_seconds'] >= 30
        trips_after_filter3 = trips_after_filter1[mask_duration_min].copy()
        self.log_filtering_step("Filter 3: Trips < 30 seconds", trips_after_filter1, trips_after_filter3)
        
        # Filter 4: Trips >= 3 hours
        mask_duration_max = trips_after_filter3['duration_seconds'] < (3 * 3600)
        trips_after_filter4 = trips_after_filter3[mask_duration_max].copy()
        self.log_filtering_step("Filter 4: Trips >= 3 hours", trips_after_filter3, trips_after_filter4)
        
        # Filter 5: Excluded modes
        mask_valid_modes = ~trips_after_filter4['Main_Mode'].isin(self.excluded_modes)
        trips_after_filter5 = trips_after_filter4[mask_valid_modes].copy()
        self.log_filtering_step("Filter 5: Excluded modes (OTHER, PLANE, etc.)", trips_after_filter4, trips_after_filter5)
        
        # Add trip type (single vs mixed)
        trips_after_filter5['is_multimodal'] = ~trips_after_filter5['Mode_2'].isna()
        trips_after_filter5['trip_type'] = trips_after_filter5.apply(
            lambda x: 'MIXED' if x['is_multimodal'] else x['Main_Mode'], axis=1
        )
        
        # Save filtered data
        individuals_filtered.to_csv(self.output_dir / 'individuals_filtered.csv', index=False)
        trips_after_filter5.to_csv(self.output_dir / 'trips_filtered.csv', index=False)
        
        # Save filtering statistics
        pd.DataFrame(self.filtering_stats).to_csv(self.output_dir / 'filtering_statistics.csv', index=False)
        
        return individuals_filtered, trips_after_filter5
    
    def analyze_modes_and_categories(self, trips_filtered):
        """Analyze trips by mode and category"""
        print("\n=== Mode Analysis ===")
        
        # Single mode analysis
        single_mode_trips = trips_filtered[~trips_filtered['is_multimodal']].copy()
        mode_stats = self.compute_trip_statistics(single_mode_trips, 'Main_Mode', 'mode')
        
        print("\nSingle mode statistics:")
        for mode, stats in sorted(mode_stats.items()):
            if 'mode_' in mode:
                mode_name = mode.replace('mode_', '')
                print(f"\n{mode_name}:")
                print(f"  Trips: {stats['nb_trips']:,}")
                print(f"  Weight: {stats['sum_weight']:,.0f}")
                print(f"  Avg duration: {stats['duration_avg']:.1f} min")
        
        # Mixed mode analysis
        mixed_trips = trips_filtered[trips_filtered['is_multimodal']].copy()
        if len(mixed_trips) > 0:
            mixed_stats = self.compute_trip_statistics(mixed_trips, label='mixed')
            print(f"\nMixed mode trips:")
            print(f"  Trips: {mixed_stats['mixed']['nb_trips']:,}")
            print(f"  Weight: {mixed_stats['mixed']['sum_weight']:,.0f}")
        
        # Category analysis
        trips_filtered['category'] = trips_filtered['Main_Mode'].map(self.mode_to_category)
        trips_filtered.loc[trips_filtered['is_multimodal'], 'category'] = 'MIXED'
        
        category_stats = self.compute_trip_statistics(trips_filtered, 'category', 'category')
        
        print("\n=== Category Analysis ===")
        for cat, stats in sorted(category_stats.items()):
            if 'category_' in cat:
                cat_name = cat.replace('category_', '')
                print(f"\n{cat_name}:")
                print(f"  Trips: {stats['nb_trips']:,}")
                print(f"  Weight: {stats['sum_weight']:,.0f}")
                print(f"  Avg duration: {stats['duration_avg']:.1f} min")
        
        # Save mode and category statistics
        all_stats = {**mode_stats, **category_stats}
        with open(self.output_dir / 'mode_category_statistics.pkl', 'wb') as f:
            pickle.dump(all_stats, f)
        
        return trips_filtered
    
    def load_gps_file(self, user_id):
        """Load GPS data for a specific user"""
        gps_file = self.data_dir / 'gps_dataset' / f'{user_id}.csv'
        if not gps_file.exists():
            return None
        
        gps_data = pd.read_csv(gps_file)
        gps_data['LOCAL_DATETIME_parsed'] = pd.to_datetime(
            gps_data['LOCAL DATETIME'], 
            format='%Y-%m-%d %H:%M:%S'
        )
        return gps_data
    
    def merge_gps_with_trips(self, individuals_filtered, trips_filtered):
        """Step 2: Merge GPS data with trips and filter by IDF bounds"""
        print("\n=== Merging GPS with Trips ===")
        
        merged_trips = []
        trips_outside_idf = 0
        weight_outside_idf = 0
        
        user_ids = individuals_filtered['ID'].unique()
        
        for user_id in tqdm(user_ids, desc="Processing users"):
            gps_data = self.load_gps_file(user_id)
            if gps_data is None:
                continue
            
            user_trips = trips_filtered[trips_filtered['ID'] == user_id]
            
            for _, trip in user_trips.iterrows():
                start_dt = trip['datetime_O']
                end_dt = trip['datetime_D']
                
                # Filter GPS points for this trip
                trip_gps = gps_data[
                    (gps_data['LOCAL_DATETIME_parsed'] >= start_dt) & 
                    (gps_data['LOCAL_DATETIME_parsed'] <= end_dt)
                ].copy()
                
                if len(trip_gps) > 0:
                    # Check if trip is within IDF bounds
                    within_bounds = (
                        (trip_gps['LATITUDE'] >= self.idf_bounds['lat_min']) &
                        (trip_gps['LATITUDE'] <= self.idf_bounds['lat_max']) &
                        (trip_gps['LONGITUDE'] >= self.idf_bounds['lon_min']) &
                        (trip_gps['LONGITUDE'] <= self.idf_bounds['lon_max'])
                    ).all()
                    
                    if not within_bounds:
                        trips_outside_idf += 1
                        weight_outside_idf += trip['Weight_Day']
                        continue
                    
                    # Calculate trip distance and speed
                    coords = trip_gps[['LATITUDE', 'LONGITUDE']].values
                    if len(coords) > 1:
                        # Simple distance calculation (can be improved with haversine)
                        diffs = np.diff(coords, axis=0)
                        distances = np.sqrt((diffs**2).sum(axis=1)) * 111  # Rough km conversion
                        total_distance = distances.sum()
                        avg_speed = (total_distance / (trip['Duration'] / 60)) if trip['Duration'] > 0 else 0
                    else:
                        total_distance = 0
                        avg_speed = 0
                    
                    trip_data = {
                        'user_id': user_id,
                        'trip_id': trip['KEY'],
                        'category': trip['category'],
                        'trip_type': trip['trip_type'],
                        'datetime_O': start_dt,
                        'datetime_D': end_dt,
                        'duration_minutes': trip['Duration'],
                        'distance_km': total_distance,
                        'speed_kmh': avg_speed,
                        'weight': trip['Weight_Day'],
                        'gps_points': trip_gps[['LOCAL_DATETIME_parsed', 'LATITUDE', 
                                               'LONGITUDE', 'SPEED']].values
                    }
                    merged_trips.append(trip_data)
        
        print(f"\nFilter 2: Outside IDF bounds")
        print(f"  Trips removed: {trips_outside_idf:,}")
        print(f"  Weight removed: {weight_outside_idf:,.0f}")
        print(f"  Merged trips: {len(merged_trips)}")
        
        # Add to filtering stats
        self.filtering_stats.append({
            'step': 'Filter 2: Outside IDF bounds',
            'trips_removed': trips_outside_idf,
            'weight_removed': weight_outside_idf
        })
        
        # Save merged data
        with open(self.output_dir / 'merged_trips.pkl', 'wb') as f:
            pickle.dump(merged_trips, f)
        
        return merged_trips
    
    def interpolate_and_resample(self, merged_trips):
        """Step 3: Interpolate to 1s and resample to 2s"""
        print("\n=== Interpolating and Resampling ===")
        
        interpolated_trips = []
        
        for trip in tqdm(merged_trips, desc="Interpolating trips"):
            gps_points = trip['gps_points']
            
            if len(gps_points) < 2:
                continue
            
            # Convert to DataFrame
            gps_df = pd.DataFrame(gps_points, 
                                columns=['timestamp', 'lat', 'lon', 'speed'])
            gps_df['timestamp'] = pd.to_datetime(gps_df['timestamp'])
            gps_df = gps_df.sort_values('timestamp').drop_duplicates('timestamp')
            
            if len(gps_df) < 2:
                continue
            
            # Create 1-second time range
            start_time = gps_df['timestamp'].min()
            end_time = gps_df['timestamp'].max()
            time_range = pd.date_range(start=start_time, end=end_time, freq='1S')
            
            if len(time_range) < 2:
                continue
            
            try:
                # Interpolate to 1s
                interpolated_df = pd.DataFrame(index=time_range)
                gps_df.set_index('timestamp', inplace=True)
                
                for col in ['lat', 'lon', 'speed']:
                    interpolated_df = interpolated_df.join(gps_df[col], how='left')
                    interpolated_df[col] = interpolated_df[col].interpolate(method='linear')
                
                interpolated_df = interpolated_df.ffill().bfill()
                
                # Subsample to 2s
                resampled_df = interpolated_df.iloc[::2].copy()
                
                trip_interpolated = trip.copy()
                trip_interpolated['gps_points'] = resampled_df.reset_index()[
                    ['index', 'lat', 'lon', 'speed']
                ].values
                interpolated_trips.append(trip_interpolated)
                
            except Exception as e:
                print(f"\nError interpolating trip {trip['trip_id']}: {str(e)}")
                continue
        
        print(f"Interpolated {len(interpolated_trips)} trips")
        
        # Save interpolated data
        with open(self.output_dir / 'interpolated_trips.pkl', 'wb') as f:
            pickle.dump(interpolated_trips, f)
        
        return interpolated_trips
    
    def cut_and_pad_sequences(self, interpolated_trips, sequence_length=2000):
        """Step 4: Cut sequences into chunks of fixed length and pad the last chunk"""
        print(f"\n=== Cutting and Padding Sequences (length={sequence_length}) ===")
        
        processed_sequences = []
        sequence_metadata = []
        
        for trip in tqdm(interpolated_trips, desc="Processing sequences"):
            gps_points = trip['gps_points']
            total_points = len(gps_points)
            
            # Cut into chunks of sequence_length
            n_chunks = int(np.ceil(total_points / sequence_length))
            
            for chunk_idx in range(n_chunks):
                start_idx = chunk_idx * sequence_length
                end_idx = min((chunk_idx + 1) * sequence_length, total_points)
                
                chunk_points = gps_points[start_idx:end_idx]
                chunk_length = len(chunk_points)
                
                # Pad if necessary
                if chunk_length < sequence_length:
                    padding_length = sequence_length - chunk_length
                    padding = np.zeros((padding_length, 4))
                    chunk_points = np.vstack([chunk_points, padding])
                    mask = np.zeros(sequence_length, dtype=bool)
                    mask[:chunk_length] = True
                else:
                    mask = np.ones(sequence_length, dtype=bool)
                
                sequence_data = {
                    'user_id': trip['user_id'],
                    'trip_id': trip['trip_id'],
                    'chunk_idx': chunk_idx,
                    'category': trip['category'],
                    'trip_type': trip['trip_type'],
                    'gps_points': chunk_points,
                    'mask': mask,
                    'original_length': chunk_length,
                    'weight': trip['weight']
                }
                processed_sequences.append(sequence_data)
                
                # Metadata for statistics
                sequence_metadata.append({
                    'category': trip['category'],
                    'weight': trip['weight'],
                    'distance_km': trip.get('distance_km', 0) * (chunk_length / total_points),
                    'speed_kmh': trip.get('speed_kmh', 0),
                    'duration_minutes': trip.get('duration_minutes', 0) * (chunk_length / total_points)
                })
        
        print(f"Created {len(processed_sequences)} sequences from {len(interpolated_trips)} trips")
        
        # Compute statistics by category
        metadata_df = pd.DataFrame(sequence_metadata)
        sequence_stats = self.compute_trip_statistics(metadata_df, 'category', 'sequences')
        
        print("\nSequence statistics by category:")
        for cat, stats in sorted(sequence_stats.items()):
            if 'sequences_' in cat:
                cat_name = cat.replace('sequences_', '')
                print(f"\n{cat_name}:")
                print(f"  Sequences: {stats['nb_trips']:,}")
                print(f"  Weight: {stats['sum_weight']:,.0f}")
        
        # Save processed sequences
        with open(self.output_dir / 'processed_sequences.pkl', 'wb') as f:
            pickle.dump(processed_sequences, f)
        
        return processed_sequences
    
    def prepare_vae_data(self, processed_sequences):
        """Step 5: Prepare and scale data for VAE using MinMaxScaler"""
        print("\n=== Preparing VAE Data ===")
        
        n_sequences = len(processed_sequences)
        sequence_length = 2000
        n_features = 3  # lat, lon, speed
        
        # Initialize arrays
        trajectories = np.zeros((n_sequences, sequence_length, n_features))
        masks = np.zeros((n_sequences, sequence_length), dtype=bool)
        categories = []
        weights = []
        
        for i, seq in enumerate(processed_sequences):
            trajectories[i] = seq['gps_points'][:, 1:4]  # lat, lon, speed
            masks[i] = seq['mask']
            categories.append(seq['category'])
            weights.append(seq['weight'])
        
        # Encode categories
        category_encoder = LabelEncoder()
        categories_encoded = category_encoder.fit_transform(categories)
        
        # Scale features using MinMaxScaler
        print("Scaling features with MinMaxScaler...")
        
        # Reshape for scaling
        trajectories_flat = trajectories.reshape(-1, n_features)
        masks_flat = masks.reshape(-1)
        
        # Create scaler and fit only on valid data
        scaler = MinMaxScaler()
        valid_data = trajectories_flat[masks_flat]
        scaler.fit(valid_data)
        
        # Transform all data
        trajectories_scaled = scaler.transform(trajectories_flat).reshape(
            n_sequences, sequence_length, n_features
        )
        
        # Apply mask to scaled data
        for i in range(n_sequences):
            trajectories_scaled[i, ~masks[i]] = 0
        
        # Save scalers
        self.scalers['trajectory'] = scaler
        self.scalers['category_encoder'] = category_encoder
        
        with open(self.output_dir / 'scalers.pkl', 'wb') as f:
            pickle.dump(self.scalers, f)
        
        # Prepare final dataset
        vae_dataset = {
            'trajectories': trajectories_scaled.astype(np.float32),
            'masks': masks,
            'categories': categories_encoded,
            'weights': np.array(weights),
            'sequence_length': sequence_length,
            'feature_names': ['latitude', 'longitude', 'speed'],
            'category_encoder': category_encoder,
            'scaler': scaler
        }
        
        # Save VAE dataset
        print("Saving VAE dataset...")
        np.savez_compressed(
            self.output_dir / 'vae_dataset.npz',
            trajectories=vae_dataset['trajectories'],
            masks=vae_dataset['masks'],
            categories=vae_dataset['categories'],
            weights=vae_dataset['weights']
        )
        
        # Save metadata
        metadata = {
            'sequence_length': sequence_length,
            'n_sequences': n_sequences,
            'n_features': n_features,
            'feature_names': vae_dataset['feature_names'],
            'categories': category_encoder.classes_.tolist(),
            'n_categories': len(category_encoder.classes_),
            'mode_categories': self.mode_categories,
            'scaling_method': 'MinMaxScaler',
            'sequence_method': 'cut_and_pad'
        }
        
        with open(self.output_dir / 'metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"\nVAE dataset prepared:")
        print(f"  Sequences: {n_sequences}")
        print(f"  Sequence length: {sequence_length}")
        print(f"  Features: {vae_dataset['feature_names']}")
        print(f"  Categories: {metadata['categories']}")
        print(f"  Scaling: MinMaxScaler")
        
        return vae_dataset
    
    def save_summary_report(self):
        """Save a comprehensive summary report"""
        report = {
            'filtering_statistics': pd.DataFrame(self.filtering_stats),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save as both CSV and pickle
        report['filtering_statistics'].to_csv(
            self.output_dir / 'preprocessing_summary.csv', index=False
        )
        
        with open(self.output_dir / 'preprocessing_report.pkl', 'wb') as f:
            pickle.dump(report, f)
        
        print("\n=== Preprocessing Summary ===")
        print(report['filtering_statistics'].to_string())
    
    def run_full_pipeline(self):
        """Run the complete preprocessing pipeline"""
        print("Starting NetMob25 Trajectory Preprocessing Pipeline...")
        print(f"Data directory: {self.data_dir}")
        print(f"Output directory: {self.output_dir}")
        
        # Step 1: Load and filter data
        individuals_filtered, trips_filtered = self.load_and_filter_data()
        
        # Analyze modes and categories
        trips_filtered = self.analyze_modes_and_categories(trips_filtered)
        
        # Step 2: Merge GPS with trips
        merged_trips = self.merge_gps_with_trips(individuals_filtered, trips_filtered)
        
        # Step 3: Interpolate and resample
        interpolated_trips = self.interpolate_and_resample(merged_trips)
        
        # Step 4: Cut and pad sequences
        processed_sequences = self.cut_and_pad_sequences(interpolated_trips)
        
        # Step 5: Prepare VAE data
        vae_dataset = self.prepare_vae_data(processed_sequences)
        
        # Save summary report
        self.save_summary_report()
        
        print("\nPreprocessing complete!")
        print(f"All outputs saved to: {self.output_dir}")
        
        return vae_dataset


def main():
    """Main function to run preprocessing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='NetMob25 Trajectory Preprocessing')
    parser.add_argument('--data-dir', type=str, default='../data/netmob25/',
                        help='Path to NetMob25 data directory')
    parser.add_argument('--output-dir', type=str, default='../preprocessing/',
                        help='Path to output directory')
    parser.add_argument('--sequence-length', type=int, default=2000,
                        help='Fixed sequence length for cutting')
    
    args = parser.parse_args()
    
    # Initialize preprocessor
    preprocessor = NetMob25TrajectoryPreprocessor(
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )
    
    # Run full pipeline
    vae_dataset = preprocessor.run_full_pipeline()
    
    print("\nSaved files:")
    print("  - individuals_filtered.csv")
    print("  - trips_filtered.csv")
    print("  - filtering_statistics.csv")
    print("  - mode_category_statistics.pkl")
    print("  - merged_trips.pkl")
    print("  - interpolated_trips.pkl")
    print("  - processed_sequences.pkl")
    print("  - scalers.pkl")
    print("  - vae_dataset.npz")
    print("  - metadata.pkl")
    print("  - preprocessing_summary.csv")
    print("  - preprocessing_report.pkl")


if __name__ == "__main__":
    main()