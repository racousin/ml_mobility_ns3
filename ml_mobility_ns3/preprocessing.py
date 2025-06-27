import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import pickle
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, LabelEncoder
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


class NetMob25Preprocessor:
    """
    Preprocessor for NetMob25 dataset to prepare data for Conditional VAE training.
    
    Transport mode handling:
    - Single mode trips: Uses the Main_Mode value
    - Multimodal trips: Uses "mixed" when Mode_2 is not null
    
    This allows the VAE to distinguish between single-mode and multimodal mobility patterns.
    """
    
    def __init__(self, data_dir='../data/netmob25/', output_dir='../preprocessing/'):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # IDF bounding box with margin
        self.idf_bounds = {
            'lat_min': 48.21,  # South boundary (around Fontainebleau in Seine-et-Marne)
            'lat_max': 49.24,  # North boundary (around Chantilly in Val-d'Oise)  
            'lon_min': 1.45,   # West boundary (around Dreux area in Yvelines)
            'lon_max': 3.55    # East boundary (around Coulommiers in Seine-et-Marne)
        }
        
        self.scalers = {}
        
    def load_and_filter_data(self):
        """Step 1: Load individuals and trips data with filtering"""
        print("Loading individuals and trips data...")
        
        # Load individuals
        individuals_df = pd.read_csv(self.data_dir / 'individuals_dataset.csv')
        individuals_filtered = individuals_df[individuals_df['GPS_RECORD'] == 1].copy()
        print(f"Filtered individuals with GPS: {len(individuals_filtered)} / {len(individuals_df)}")
        
        # Load trips
        trips_df = pd.read_csv(self.data_dir / 'trips_dataset.csv')
        trips_filtered = trips_df[
            trips_df[['Date_O', 'Time_O', 'Date_D', 'Time_D']].notnull().all(axis=1)
        ].copy()
        print(f"Filtered trips with valid times: {len(trips_filtered)} / {len(trips_df)}")
        
        # Save filtered data
        individuals_filtered.to_csv(self.output_dir / 'individuals_filtered.csv', index=False)
        trips_filtered.to_csv(self.output_dir / 'trips_filtered.csv', index=False)
        
        return individuals_filtered, trips_filtered
    
    def load_gps_file(self, user_id):
        """Load GPS data for a specific user"""
        gps_file = self.data_dir / 'gps_dataset' / f'{user_id}.csv'
        if not gps_file.exists():
            return None
        
        gps_data = pd.read_csv(gps_file)
        # Parse timestamps
        gps_data['LOCAL_DATETIME_parsed'] = pd.to_datetime(
            gps_data['LOCAL DATETIME'], 
            format='%Y-%m-%d %H:%M:%S'
        )
        return gps_data
    
    def analyze_multimodal_trips(self, trips_filtered):
        """Analyze multimodal trips in the dataset"""
        print("\nAnalyzing multimodal trips...")
        
        # Count single vs multimodal trips
        single_mode_trips = trips_filtered[trips_filtered['Mode_2'].isna()]
        multimodal_trips = trips_filtered[~trips_filtered['Mode_2'].isna()]
        
        print(f"  Single mode trips: {len(single_mode_trips)} ({len(single_mode_trips)/len(trips_filtered)*100:.1f}%)")
        print(f"  Multimodal trips: {len(multimodal_trips)} ({len(multimodal_trips)/len(trips_filtered)*100:.1f}%)")
        
        if len(multimodal_trips) > 0:
            # Analyze mode combinations
            print(f"\n  Most common multimodal combinations:")
            
            # Get mode columns
            mode_cols = [f'Mode_{i}' for i in range(1, 6) if f'Mode_{i}' in multimodal_trips.columns]
            
            # Count combinations
            combinations = []
            for _, trip in multimodal_trips.iterrows():
                modes = []
                for col in mode_cols:
                    if pd.notna(trip.get(col)):
                        modes.append(trip[col])
                if modes:
                    combinations.append(' -> '.join(modes))
            
            combo_counts = Counter(combinations)
            
            for combo, count in combo_counts.most_common(10):
                print(f"    {combo}: {count} trips ({count/len(multimodal_trips)*100:.1f}%)")
        
        return single_mode_trips, multimodal_trips
    
    def merge_gps_with_trips(self, individuals_filtered, trips_filtered):
        """Step 2: Merge GPS data with trips based on time windows"""
        print("\nMerging GPS data with trips...")
        
        # First analyze multimodal trips
        single_mode_trips, multimodal_trips = self.analyze_multimodal_trips(trips_filtered)
        
        merged_trips = []
        
        # Get unique user IDs with GPS data
        user_ids = individuals_filtered['ID'].unique()
        
        for user_id in tqdm(user_ids, desc="Processing users"):
            # Load GPS data for user
            gps_data = self.load_gps_file(user_id)
            if gps_data is None:
                continue
            
            # Get trips for this user
            user_trips = trips_filtered[trips_filtered['ID'] == user_id]
            
            for _, trip in user_trips.iterrows():
                # Parse trip times
                start_time = f"{trip['Date_O']} {trip['Time_O']}"
                end_time = f"{trip['Date_D']} {trip['Time_D']}"
                
                start_dt = pd.to_datetime(start_time, format='%Y-%m-%d %H:%M:%S')
                end_dt = pd.to_datetime(end_time, format='%Y-%m-%d %H:%M:%S')
                
                # Filter GPS points for this trip
                trip_gps = gps_data[
                    (gps_data['LOCAL_DATETIME_parsed'] >= start_dt) & 
                    (gps_data['LOCAL_DATETIME_parsed'] <= end_dt)
                ].copy()
                
                if len(trip_gps) > 0:
                    # Determine transport mode: use Main_Mode if single mode, "mixed" if multimodal
                    if pd.isna(trip.get('Mode_2')):
                        transport_mode = trip['Main_Mode']
                    else:
                        transport_mode = 'mixed'
                    
                    trip_data = {
                        'user_id': user_id,
                        'trip_id': trip['KEY'],
                        'transport_mode': transport_mode,
                        'start_time': start_dt,
                        'end_time': end_dt,
                        'duration_minutes': trip['Duration'],
                        'gps_points': trip_gps[['LOCAL_DATETIME_parsed', 'LATITUDE', 
                                               'LONGITUDE', 'SPEED']].values
                    }
                    merged_trips.append(trip_data)
        
        print(f"Merged {len(merged_trips)} trips with GPS data")
        
        # Count transport modes in merged trips
        mode_counts = {}
        for trip in merged_trips:
            mode = trip['transport_mode']
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
        
        if 'mixed' in mode_counts:
            print(f"  Including {mode_counts['mixed']} mixed-mode trips")
        
        # Save merged data
        with open(self.output_dir / 'merged_trips.pkl', 'wb') as f:
            pickle.dump(merged_trips, f)
        
        return merged_trips
    
    def check_and_handle_duplicates(self, gps_df):
        """Check for duplicate timestamps and handle them"""
        # Check for duplicates
        duplicated_mask = gps_df['timestamp'].duplicated(keep=False)
        
        if duplicated_mask.any():
            n_duplicates = duplicated_mask.sum()
            duplicate_groups = gps_df[duplicated_mask].groupby('timestamp')
            
            # Check if duplicates have different values
            differences = []
            for timestamp, group in duplicate_groups:
                if len(group) > 1:
                    # Check if lat/lon/speed values differ
                    for col in ['lat', 'lon', 'speed']:
                        unique_vals = group[col].unique()
                        if len(unique_vals) > 1:
                            differences.append({
                                'timestamp': timestamp,
                                'column': col,
                                'values': unique_vals.tolist()
                            })
            
            if differences:
                print(f"\n  Found {n_duplicates} duplicate timestamps with different values:")
                for diff in differences[:5]:  # Show first 5 differences
                    print(f"    {diff['timestamp']}: {diff['column']} = {diff['values']}")
                if len(differences) > 5:
                    print(f"    ... and {len(differences) - 5} more differences")
            
            # Handle duplicates by keeping the first occurrence
            gps_df = gps_df.drop_duplicates(subset=['timestamp'], keep='first')
        
        return gps_df
    
    def interpolate_and_resample(self, merged_trips):
        """Step 3: Interpolate to 1s and then subsample to 2s"""
        print("\nInterpolating and resampling GPS data...")
        
        interpolated_trips = []
        n_trips_with_duplicates = 0
        
        for trip in tqdm(merged_trips, desc="Interpolating trips"):
            gps_points = trip['gps_points']
            
            if len(gps_points) < 2:
                continue
            
            # Convert to DataFrame for easier handling
            gps_df = pd.DataFrame(gps_points, 
                                columns=['timestamp', 'lat', 'lon', 'speed'])
            gps_df['timestamp'] = pd.to_datetime(gps_df['timestamp'])
            gps_df = gps_df.sort_values('timestamp')
            
            # Check and handle duplicates
            original_len = len(gps_df)
            gps_df = self.check_and_handle_duplicates(gps_df)
            if len(gps_df) < original_len:
                n_trips_with_duplicates += 1
            
            # Skip if too few points after deduplication
            if len(gps_df) < 2:
                continue
            
            # Create 1-second time range
            start_time = gps_df['timestamp'].min()
            end_time = gps_df['timestamp'].max()
            time_range = pd.date_range(start=start_time, end=end_time, freq='1S')
            
            # Skip if time range is too short
            if len(time_range) < 2:
                continue
            
            try:
                # Interpolate to 1s
                interpolated_df = pd.DataFrame(index=time_range)
                
                # Set original data
                gps_df.set_index('timestamp', inplace=True)
                
                # Merge and interpolate
                for col in ['lat', 'lon', 'speed']:
                    # Use join instead of direct assignment to avoid reindex issues
                    temp_series = gps_df[col]
                    interpolated_df = interpolated_df.join(temp_series, how='left')
                    interpolated_df[col] = interpolated_df[col].interpolate(method='linear')
                
                # Fill any remaining NaN values
                interpolated_df = interpolated_df.ffill().bfill()
                
                # Subsample to 2s
                resampled_df = interpolated_df.iloc[::2].copy()
                
                # Update trip data
                trip_interpolated = trip.copy()
                trip_interpolated['gps_points'] = resampled_df.reset_index()[
                    ['index', 'lat', 'lon', 'speed']
                ].values
                interpolated_trips.append(trip_interpolated)
                
            except Exception as e:
                print(f"\n  Error interpolating trip {trip['trip_id']}: {str(e)}")
                continue
        
        print(f"Interpolated {len(interpolated_trips)} trips")
        if n_trips_with_duplicates > 0:
            print(f"  Found and handled duplicates in {n_trips_with_duplicates} trips")
        
        # Save interpolated data
        with open(self.output_dir / 'interpolated_trips.pkl', 'wb') as f:
            pickle.dump(interpolated_trips, f)
        
        return interpolated_trips
    
    def filter_idf_bounds(self, interpolated_trips):
        """Step 4: Filter GPS points outside IDF bounds"""
        print("\nFiltering trips to IDF bounds...")
        
        filtered_trips = []
        
        for trip in tqdm(interpolated_trips, desc="Filtering bounds"):
            gps_points = trip['gps_points']
            
            # Filter points within IDF bounds
            mask = (
                (gps_points[:, 1] >= self.idf_bounds['lat_min']) &
                (gps_points[:, 1] <= self.idf_bounds['lat_max']) &
                (gps_points[:, 2] >= self.idf_bounds['lon_min']) &
                (gps_points[:, 2] <= self.idf_bounds['lon_max'])
            )
            
            # Find first and last valid indices
            valid_indices = np.where(mask)[0]
            
            if len(valid_indices) > 0:
                start_idx = valid_indices[0]
                end_idx = valid_indices[-1] + 1
                
                trip_filtered = trip.copy()
                trip_filtered['gps_points'] = gps_points[start_idx:end_idx]
                
                if len(trip_filtered['gps_points']) > 0:
                    filtered_trips.append(trip_filtered)
        
        print(f"Filtered to {len(filtered_trips)} trips within IDF bounds")
        
        # Save filtered data
        with open(self.output_dir / 'filtered_trips.pkl', 'wb') as f:
            pickle.dump(filtered_trips, f)
        
        return filtered_trips
    
    def compute_sequence_length(self, filtered_trips, quantile=0.95):
        """Step 5: Compute the quantile sequence length"""
        print(f"\nComputing {quantile*100}% quantile sequence length...")
        
        sequence_lengths = [len(trip['gps_points']) for trip in filtered_trips]
        
        trip_sequence_length = int(np.quantile(sequence_lengths, quantile))
        
        print(f"Sequence length statistics:")
        print(f"  Min: {np.min(sequence_lengths)}")
        print(f"  Max: {np.max(sequence_lengths)}")
        print(f"  Mean: {np.mean(sequence_lengths):.1f}")
        print(f"  {quantile*100}% quantile: {trip_sequence_length}")
        
        # Save sequence length
        with open(self.output_dir / 'sequence_length.txt', 'w') as f:
            f.write(str(trip_sequence_length))
        
        return trip_sequence_length
    
    def pad_or_truncate_sequences(self, filtered_trips, trip_sequence_length):
        """Step 6: Pad or truncate sequences to fixed length"""
        print(f"\nPadding/truncating sequences to length {trip_sequence_length}...")
        
        processed_trips = []
        
        for trip in tqdm(filtered_trips, desc="Processing sequences"):
            gps_points = trip['gps_points']
            current_length = len(gps_points)
            
            if current_length > trip_sequence_length:
                # Truncate
                processed_points = gps_points[:trip_sequence_length]
                mask = np.ones(trip_sequence_length, dtype=bool)
            else:
                # Pad with zeros and create mask
                padding_length = trip_sequence_length - current_length
                padding = np.zeros((padding_length, 4))  # 4 features: time, lat, lon, speed
                processed_points = np.vstack([gps_points, padding])
                
                # Create mask (1 for valid data, 0 for padding)
                mask = np.zeros(trip_sequence_length, dtype=bool)
                mask[:current_length] = True
            
            trip_processed = trip.copy()
            trip_processed['gps_points'] = processed_points
            trip_processed['mask'] = mask
            trip_processed['original_length'] = current_length
            
            processed_trips.append(trip_processed)
        
        # Save processed data
        with open(self.output_dir / 'processed_trips.pkl', 'wb') as f:
            pickle.dump(processed_trips, f)
        
        return processed_trips
    
    def prepare_vae_data(self, processed_trips, trip_sequence_length):
        """Step 7: Prepare and scale data for VAE"""
        print("\nPreparing data for VAE...")
        
        # Extract features
        n_trips = len(processed_trips)
        n_features = 3  # lat, lon, speed
        
        # Initialize arrays
        trajectories = np.zeros((n_trips, trip_sequence_length, n_features))
        masks = np.zeros((n_trips, trip_sequence_length), dtype=bool)
        transport_modes = []
        trip_lengths = []
        
        for i, trip in enumerate(processed_trips):
            # Extract trajectory features (excluding timestamp)
            trajectories[i] = trip['gps_points'][:, 1:4]  # lat, lon, speed
            masks[i] = trip['mask']
            transport_modes.append(trip['transport_mode'])
            trip_lengths.append(trip['original_length'])
        
        # Encode transport modes
        mode_encoder = LabelEncoder()
        transport_modes_encoded = mode_encoder.fit_transform(transport_modes)
        
        # Scale features
        print("Scaling features...")
        
        # Reshape for scaling
        trajectories_flat = trajectories.reshape(-1, n_features)
        masks_flat = masks.reshape(-1)
        
        # Create scaler and fit only on valid data
        scaler = StandardScaler()
        valid_data = trajectories_flat[masks_flat]
        scaler.fit(valid_data)
        
        # Transform all data
        trajectories_scaled = scaler.transform(trajectories_flat).reshape(
            n_trips, trip_sequence_length, n_features
        )
        
        # Apply mask to scaled data (set padded values to 0)
        for i in range(n_trips):
            trajectories_scaled[i, ~masks[i]] = 0
        
        # Save scalers
        self.scalers['trajectory'] = scaler
        self.scalers['mode_encoder'] = mode_encoder
        
        with open(self.output_dir / 'scalers.pkl', 'wb') as f:
            pickle.dump(self.scalers, f)
        
        # Prepare final dataset
        vae_dataset = {
            'trajectories': trajectories_scaled.astype(np.float32),
            'masks': masks,
            'transport_modes': transport_modes_encoded,
            'trip_lengths': np.array(trip_lengths),
            'sequence_length': trip_sequence_length,
            'feature_names': ['latitude', 'longitude', 'speed'],
            'mode_encoder': mode_encoder,
            'scaler': scaler
        }
        
        # Save VAE dataset
        print("Saving VAE dataset...")
        np.savez_compressed(
            self.output_dir / 'vae_dataset.npz',
            trajectories=vae_dataset['trajectories'],
            masks=vae_dataset['masks'],
            transport_modes=vae_dataset['transport_modes'],
            trip_lengths=vae_dataset['trip_lengths']
        )
        
        # Save metadata
        metadata = {
            'sequence_length': trip_sequence_length,
            'n_trips': n_trips,
            'n_features': n_features,
            'feature_names': vae_dataset['feature_names'],
            'transport_modes': mode_encoder.classes_.tolist(),
            'n_transport_modes': len(mode_encoder.classes_),
            'note': 'Transport mode "mixed" indicates multimodal trips (where Mode_2 is not null)'
        }
        
        with open(self.output_dir / 'metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"\nVAE dataset prepared:")
        print(f"  Trajectories shape: {trajectories_scaled.shape}")
        print(f"  Number of trips: {n_trips}")
        print(f"  Sequence length: {trip_sequence_length}")
        print(f"  Features: {vae_dataset['feature_names']}")
        print(f"  Transport modes: {len(mode_encoder.classes_)}")
        
        return vae_dataset
    
    def run_full_pipeline(self, quantile=0.95):
        """Run the complete preprocessing pipeline"""
        print("Starting NetMob25 VAE preprocessing pipeline...")
        print(f"Data directory: {self.data_dir}")
        print(f"Output directory: {self.output_dir}")
        
        # Step 1: Load and filter data
        individuals_filtered, trips_filtered = self.load_and_filter_data()
        
        # Step 2: Merge GPS with trips
        merged_trips = self.merge_gps_with_trips(individuals_filtered, trips_filtered)
        
        # Step 3: Interpolate and resample
        interpolated_trips = self.interpolate_and_resample(merged_trips)
        
        # Step 4: Filter to IDF bounds
        filtered_trips = self.filter_idf_bounds(interpolated_trips)
        
        # Step 5: Compute sequence length
        trip_sequence_length = self.compute_sequence_length(filtered_trips, quantile)
        
        # Step 6: Pad or truncate sequences
        processed_trips = self.pad_or_truncate_sequences(filtered_trips, trip_sequence_length)
        
        # Step 7: Prepare VAE data
        vae_dataset = self.prepare_vae_data(processed_trips, trip_sequence_length)
        
        print("\nPreprocessing complete!")
        print(f"All outputs saved to: {self.output_dir}")
        
        return vae_dataset


    def analyze_duplicates_in_dataset(self, merged_trips):
        """Analyze duplicate timestamps across all trips"""
        print("\nAnalyzing duplicates in dataset...")
        
        total_duplicates = 0
        trips_with_duplicates = 0
        duplicate_stats = []
        
        for trip in tqdm(merged_trips, desc="Analyzing duplicates"):
            gps_points = trip['gps_points']
            
            if len(gps_points) < 2:
                continue
                
            # Convert to DataFrame
            gps_df = pd.DataFrame(gps_points, 
                                columns=['timestamp', 'lat', 'lon', 'speed'])
            gps_df['timestamp'] = pd.to_datetime(gps_df['timestamp'])
            
            # Check for duplicates
            duplicated_mask = gps_df['timestamp'].duplicated(keep=False)
            
            if duplicated_mask.any():
                trips_with_duplicates += 1
                n_duplicates = duplicated_mask.sum()
                total_duplicates += n_duplicates
                
                duplicate_stats.append({
                    'user_id': trip['user_id'],
                    'trip_id': trip['trip_id'],
                    'n_duplicates': n_duplicates,
                    'total_points': len(gps_df),
                    'duplicate_ratio': n_duplicates / len(gps_df)
                })
        
        print(f"\nDuplicate Analysis Summary:")
        print(f"  Total trips analyzed: {len(merged_trips)}")
        print(f"  Trips with duplicates: {trips_with_duplicates} ({trips_with_duplicates/len(merged_trips)*100:.1f}%)")
        print(f"  Total duplicate points: {total_duplicates}")
        
        if duplicate_stats:
            # Show top trips with most duplicates
            duplicate_stats.sort(key=lambda x: x['n_duplicates'], reverse=True)
            print(f"\n  Top 5 trips with most duplicates:")
            for stat in duplicate_stats[:5]:
                print(f"    User {stat['user_id']}, Trip {stat['trip_id']}: "
                      f"{stat['n_duplicates']} duplicates out of {stat['total_points']} points "
                      f"({stat['duplicate_ratio']*100:.1f}%)")
        
        return duplicate_stats
    
    def run_from_checkpoint(self, checkpoint_name, quantile=0.95):
        """Resume pipeline from a specific checkpoint"""
        checkpoints = {
            'filtered': 'individuals_filtered.csv',
            'merged': 'merged_trips.pkl',
            'interpolated': 'interpolated_trips.pkl',
            'bounded': 'filtered_trips.pkl',
            'processed': 'processed_trips.pkl'
        }
        
        if checkpoint_name not in checkpoints:
            raise ValueError(f"Unknown checkpoint: {checkpoint_name}. "
                           f"Available: {list(checkpoints.keys())}")
        
        checkpoint_file = self.output_dir / checkpoints[checkpoint_name]
        if not checkpoint_file.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")
        
        print(f"Resuming from checkpoint: {checkpoint_name}")
        
        # Load data based on checkpoint
        if checkpoint_name == 'filtered':
            individuals_filtered = pd.read_csv(self.output_dir / 'individuals_filtered.csv')
            trips_filtered = pd.read_csv(self.output_dir / 'trips_filtered.csv')
            
            # Continue from step 2
            merged_trips = self.merge_gps_with_trips(individuals_filtered, trips_filtered)
            interpolated_trips = self.interpolate_and_resample(merged_trips)
            filtered_trips = self.filter_idf_bounds(interpolated_trips)
            trip_sequence_length = self.compute_sequence_length(filtered_trips, quantile)
            processed_trips = self.pad_or_truncate_sequences(filtered_trips, trip_sequence_length)
            vae_dataset = self.prepare_vae_data(processed_trips, trip_sequence_length)
            
        elif checkpoint_name == 'merged':
            with open(checkpoint_file, 'rb') as f:
                merged_trips = pickle.load(f)
            
            # Optionally analyze duplicates
            self.analyze_duplicates_in_dataset(merged_trips)
            
            # Continue from step 3
            interpolated_trips = self.interpolate_and_resample(merged_trips)
            filtered_trips = self.filter_idf_bounds(interpolated_trips)
            trip_sequence_length = self.compute_sequence_length(filtered_trips, quantile)
            processed_trips = self.pad_or_truncate_sequences(filtered_trips, trip_sequence_length)
            vae_dataset = self.prepare_vae_data(processed_trips, trip_sequence_length)
            
        elif checkpoint_name == 'interpolated':
            with open(checkpoint_file, 'rb') as f:
                interpolated_trips = pickle.load(f)
            
            # Continue from step 4
            filtered_trips = self.filter_idf_bounds(interpolated_trips)
            trip_sequence_length = self.compute_sequence_length(filtered_trips, quantile)
            processed_trips = self.pad_or_truncate_sequences(filtered_trips, trip_sequence_length)
            vae_dataset = self.prepare_vae_data(processed_trips, trip_sequence_length)
            
        elif checkpoint_name == 'bounded':
            with open(checkpoint_file, 'rb') as f:
                filtered_trips = pickle.load(f)
            
            # Continue from step 5
            trip_sequence_length = self.compute_sequence_length(filtered_trips, quantile)
            processed_trips = self.pad_or_truncate_sequences(filtered_trips, trip_sequence_length)
            vae_dataset = self.prepare_vae_data(processed_trips, trip_sequence_length)
            
        elif checkpoint_name == 'processed':
            with open(checkpoint_file, 'rb') as f:
                processed_trips = pickle.load(f)
            
            # Load sequence length
            with open(self.output_dir / 'sequence_length.txt', 'r') as f:
                trip_sequence_length = int(f.read().strip())
            
            # Continue from step 7
            vae_dataset = self.prepare_vae_data(processed_trips, trip_sequence_length)
        
        print("\nProcessing complete from checkpoint!")
        return vae_dataset
    
    @staticmethod
    def load_vae_dataset(preprocessing_dir='../preprocessing/'):
        """Load the preprocessed VAE dataset"""
        preprocessing_dir = Path(preprocessing_dir)
        
        # Load main data
        data = np.load(preprocessing_dir / 'vae_dataset.npz')
        
        # Load metadata
        with open(preprocessing_dir / 'metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        
        # Load scalers
        with open(preprocessing_dir / 'scalers.pkl', 'rb') as f:
            scalers = pickle.load(f)
        
        return {
            'trajectories': data['trajectories'],
            'masks': data['masks'],
            'transport_modes': data['transport_modes'],
            'trip_lengths': data['trip_lengths'],
            'metadata': metadata,
            'scalers': scalers
        }
    
    @staticmethod
    def print_dataset_info(vae_data):
        """Print information about the VAE dataset"""
        print("\nVAE Dataset Information:")
        print(f"  Trajectories shape: {vae_data['trajectories'].shape}")
        print(f"  Number of trips: {len(vae_data['trajectories'])}")
        print(f"  Sequence length: {vae_data['metadata']['sequence_length']}")
        print(f"  Features: {vae_data['metadata']['feature_names']}")
        print(f"  Transport modes: {vae_data['metadata']['transport_modes']}")
        print(f"  Number of modes: {vae_data['metadata']['n_transport_modes']}")
        
        if 'note' in vae_data['metadata']:
            print(f"\n  Note: {vae_data['metadata']['note']}")
        
        # Trip length statistics
        lengths = vae_data['trip_lengths']
        print(f"\n  Trip length statistics:")
        print(f"    Min: {np.min(lengths)}")
        print(f"    Max: {np.max(lengths)}")
        print(f"    Mean: {np.mean(lengths):.1f}")
        print(f"    Std: {np.std(lengths):.1f}")
        
        # Transport mode distribution
        print(f"\n  Transport mode distribution:")
        mode_counts = np.bincount(vae_data['transport_modes'])
        for i, mode in enumerate(vae_data['metadata']['transport_modes']):
            if i < len(mode_counts):
                print(f"    {mode}: {mode_counts[i]} ({mode_counts[i]/len(vae_data['transport_modes'])*100:.1f}%)")


def main():
    """Main function to run preprocessing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='NetMob25 VAE Preprocessing')
    parser.add_argument('--data-dir', type=str, default='../data/netmob25/',
                        help='Path to NetMob25 data directory')
    parser.add_argument('--output-dir', type=str, default='../preprocessing/',
                        help='Path to output directory')
    parser.add_argument('--quantile', type=float, default=0.95,
                        help='Quantile for sequence length (default: 0.95)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        choices=['filtered', 'merged', 'interpolated', 'bounded', 'processed'],
                        help='Resume from checkpoint')
    parser.add_argument('--analyze-duplicates', action='store_true',
                        help='Analyze duplicates in the dataset')
    
    args = parser.parse_args()
    
    # Initialize preprocessor
    preprocessor = NetMob25Preprocessor(
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )
    
    if args.checkpoint:
        # Resume from checkpoint
        vae_dataset = preprocessor.run_from_checkpoint(args.checkpoint, args.quantile)
    else:
        # Run full pipeline
        vae_dataset = preprocessor.run_full_pipeline(quantile=args.quantile)
    
    # Print summary
    print("\nSaved files:")
    print("  - individuals_filtered.csv")
    print("  - trips_filtered.csv")
    print("  - merged_trips.pkl")
    print("  - interpolated_trips.pkl")
    print("  - filtered_trips.pkl")
    print("  - processed_trips.pkl")
    print("  - sequence_length.txt")
    print("  - scalers.pkl")
    print("  - vae_dataset.npz")
    print("  - metadata.pkl")


if __name__ == "__main__":
    main()