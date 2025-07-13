import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import pickle
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from typing import Dict, Tuple, List
import logging

from ml_mobility_ns3.metrics.stat_metrics import StatMetrics

logger = logging.getLogger(__name__)


class TrajectoryPreprocessor:
    """
    Preprocessor for NetMob25 dataset to prepare data for trajectory generation models.
    """
    
    def __init__(self, config):
        self.config = config
        self.data_dir = Path(config.data_dir)
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.idf_bounds = dict(config.idf_bounds)
        self.excluded_modes = list(config.excluded_modes)
        self.sequence_length = config.sequence_length
        
        # Transport mode categories
        self.mode_categories = {
            'CAR': ['TWO_WHEELER', 'PRIV_CAR_DRIVER', 'PRIV_CAR_PASSENGER', 'TAXI'],
            'PUBLIC_TRANSPORT': ['BUS', 'TRAIN', 'TRAIN_EXPRESS', 'SUBWAY', 'TRAMWAY'],
            'WALKING': ['WALKING'],
            'BIKE': ['BIKE', 'ELECT_BIKE', 'ELECT_SCOOTER']
        }
        
        # Create reverse mapping
        self.mode_to_category = {}
        for category, modes in self.mode_categories.items():
            for mode in modes:
                self.mode_to_category[mode] = category
        
        self.filtering_stats = []
        self.scalers = {}
        self.metrics = StatMetrics()
        
    def process(self) -> Dict:
        """Run the complete preprocessing pipeline."""
        logger.info("Starting preprocessing pipeline")
        
        # Step 1: Load and filter data
        individuals, trips = self._load_and_filter_data()
        
        # Step 2: Merge with GPS data
        merged_trips = self._merge_gps_with_trips(individuals, trips)
        
        # Step 3: Interpolate and resample
        interpolated_trips = self._interpolate_and_resample(merged_trips)
        
        # Step 4: Clip speed outliers
        clipped_trips = self._clip_speed_outliers(interpolated_trips, percentile=99)
        
        # Step 5: Cut and pad sequences
        sequences = self._cut_and_pad_sequences(clipped_trips)
        
        # Step 6: Prepare final dataset
        dataset = self._prepare_dataset(sequences)
        
        # Save outputs
        self._save_outputs(dataset)
        
        # Save summary report
        self._save_summary_report()
        
        return dataset
    
    def _log_filtering_step(self, step_name: str, before_df: pd.DataFrame, 
                           after_df: pd.DataFrame):
        """Log statistics for filtering step."""
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
        
        logger.info(f"{step_name}:")
        logger.info(f"  Trips removed: {nb_removed:,} ({stats['trips_removed_pct']:.1f}%)")
        logger.info(f"  Weight removed: {weight_removed:,.0f} ({stats['weight_removed_pct']:.1f}%)")
        logger.info(f"  Remaining trips: {len(after_df):,}")
    
    def _load_and_filter_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and apply sequential filters with statistics tracking."""
        logger.info("Loading data...")
        
        # Load individuals
        individuals_df = pd.read_csv(self.data_dir / 'individuals_dataset.csv')
        individuals_filtered = individuals_df[individuals_df['GPS_RECORD'] == 1].copy()
        logger.info(f"Individuals with GPS: {len(individuals_filtered)} / {len(individuals_df)}")
        
        # Load trips
        trips_df = pd.read_csv(self.data_dir / 'trips_dataset.csv')
        logger.info(f"Initial trips: {len(trips_df)}")
        
        # Add Weight_Day column if not present
        if 'Weight_Day' not in trips_df.columns:
            trips_df['Weight_Day'] = 1.0
        
        # Filter 1: Missing date/time values
        trips_filtered = trips_df.copy()
        mask_complete_times = trips_filtered[['Date_O', 'Time_O', 'Date_D', 'Time_D']].notnull().all(axis=1)
        trips_after_filter1 = trips_filtered[mask_complete_times].copy()
        self._log_filtering_step("Filter 1: Missing date/time values", trips_filtered, trips_after_filter1)
        
        # Parse dates and times
        trips_after_filter1['datetime_O'] = pd.to_datetime(
            trips_after_filter1['Date_O'] + ' ' + trips_after_filter1['Time_O'],
            format='%Y-%m-%d %H:%M:%S'
        )
        trips_after_filter1['datetime_D'] = pd.to_datetime(
            trips_after_filter1['Date_D'] + ' ' + trips_after_filter1['Time_D'],
            format='%Y-%m-%d %H:%M:%S'
        )
        
        # Filter 2: Trips < 30 seconds
        trips_after_filter1['duration_seconds'] = (
            trips_after_filter1['datetime_D'] - trips_after_filter1['datetime_O']
        ).dt.total_seconds()
        mask_duration_min = trips_after_filter1['duration_seconds'] >= 30
        trips_after_filter2 = trips_after_filter1[mask_duration_min].copy()
        self._log_filtering_step("Filter 2: Trips < 30 seconds", trips_after_filter1, trips_after_filter2)
        
        # Filter 3: Trips >= 3 hours
        mask_duration_max = trips_after_filter2['duration_seconds'] < (3 * 3600)
        trips_after_filter3 = trips_after_filter2[mask_duration_max].copy()
        self._log_filtering_step("Filter 3: Trips >= 3 hours", trips_after_filter2, trips_after_filter3)
        
        # Filter 4: Excluded modes
        mask_valid_modes = ~trips_after_filter3['Main_Mode'].isin(self.excluded_modes)
        trips_after_filter4 = trips_after_filter3[mask_valid_modes].copy()
        self._log_filtering_step("Filter 4: Excluded modes", trips_after_filter3, trips_after_filter4)
        
        # Add trip type and category
        trips_after_filter4['is_multimodal'] = ~trips_after_filter4['Mode_2'].isna()
        trips_after_filter4['trip_type'] = trips_after_filter4.apply(
            lambda x: 'MIXED' if x['is_multimodal'] else x['Main_Mode'], axis=1
        )
        trips_after_filter4['category'] = trips_after_filter4['Main_Mode'].map(self.mode_to_category)
        trips_after_filter4.loc[trips_after_filter4['is_multimodal'], 'category'] = 'MIXED'
        
        # Save filtered data
        individuals_filtered.to_csv(self.output_dir / 'individuals_filtered.csv', index=False)
        trips_after_filter4.to_csv(self.output_dir / 'trips_filtered.csv', index=False)
        
        return individuals_filtered, trips_after_filter4
    
    def _clip_speed_outliers(self, interpolated_trips: List[Dict], percentile: int = 99) -> List[Dict]:
        """
        Clip speed outliers to the specified percentile threshold for each transport mode type.
        
        Args:
            interpolated_trips: List of trip dictionaries with GPS points
            percentile: Percentile threshold for clipping (default: 90)
        
        Returns:
            List of trips with clipped speeds
        """
        from collections import defaultdict
        
        logger.info(f"Clipping speed outliers to {percentile}th percentile by transport mode...")
        
        # Group trips by transport mode type
        trips_by_mode = defaultdict(list)
        for trip in interpolated_trips:
            trips_by_mode[trip['trip_type']].append(trip)
        
        # Calculate percentile thresholds for each mode
        mode_thresholds = {}
        for mode, mode_trips in trips_by_mode.items():
            # Collect all speed values for this mode
            all_speeds = []
            for trip in mode_trips:
                gps_points = trip['gps_points']
                if len(gps_points) > 0:
                    # Speed is in column 3 (index 3) of gps_points
                    speeds = gps_points[:, 3]
                    # Filter out any invalid speeds (negative or extremely high)
                    valid_speeds = speeds[(speeds >= 0) & (speeds < 500)]  # reasonable upper bound
                    all_speeds.extend(valid_speeds)
            
            if all_speeds:
                threshold = np.percentile(all_speeds, percentile)
                mode_thresholds[mode] = threshold
                logger.info(f"  {mode}: {percentile}th percentile speed = {threshold:.2f} km/h")
            else:
                mode_thresholds[mode] = float('inf')  # No clipping if no valid speeds
        
        # Apply clipping to each trip
        clipped_trips = []
        total_clipped_points = 0
        total_points = 0
        
        for trip in interpolated_trips:
            mode = trip['trip_type']
            threshold = mode_thresholds.get(mode, float('inf'))
            
            trip_clipped = trip.copy()
            gps_points = trip['gps_points'].copy()
            
            if len(gps_points) > 0:
                # Count points before clipping
                speeds_before = gps_points[:, 3]
                points_to_clip = np.sum(speeds_before > threshold)
                total_clipped_points += points_to_clip
                total_points += len(speeds_before)
                
                # Clip speeds to threshold
                gps_points[:, 3] = np.clip(gps_points[:, 3], 0, threshold)
                
                # Update trip data
                trip_clipped['gps_points'] = gps_points
                
                # Recalculate average speed for the trip
                if len(gps_points) > 0:
                    # Calculate new average speed from clipped GPS points
                    valid_speeds = gps_points[:, 3]
                    trip_clipped['speed_kmh'] = np.mean(valid_speeds) if len(valid_speeds) > 0 else 0
            
            clipped_trips.append(trip_clipped)
        
        # Log clipping statistics
        clipping_percentage = (total_clipped_points / total_points * 100) if total_points > 0 else 0
        logger.info(f"Speed clipping completed:")
        logger.info(f"  Total GPS points processed: {total_points:,}")
        logger.info(f"  Points clipped: {total_clipped_points:,} ({clipping_percentage:.2f}%)")
        
        return clipped_trips

    def _load_gps_file(self, user_id: int) -> pd.DataFrame:
        """Load GPS data for a specific user."""
        gps_file = self.data_dir / 'gps_dataset' / f'{user_id}.csv'
        if not gps_file.exists():
            return None
        
        gps_data = pd.read_csv(gps_file)
        gps_data['LOCAL_DATETIME_parsed'] = pd.to_datetime(
            gps_data['LOCAL DATETIME'], 
            format='%Y-%m-%d %H:%M:%S'
        )
        return gps_data
    
    def _merge_gps_with_trips(self, individuals: pd.DataFrame, trips: pd.DataFrame) -> List[Dict]:
        """Merge GPS data with trips and filter by bounds."""
        logger.info("Merging GPS with trips...")
        
        merged_trips = []
        trips_outside_bounds = 0
        weight_outside_bounds = 0
        total_trips_to_process = 0
        total_weight_to_process = 0
        
        user_ids = individuals['ID'].unique()
        
        for user_id in tqdm(user_ids, desc="Processing users"):
            gps_data = self._load_gps_file(user_id)
            if gps_data is None:
                continue
            
            user_trips = trips[trips['ID'] == user_id]
            
            for _, trip in user_trips.iterrows():
                start_dt = trip['datetime_O']
                end_dt = trip['datetime_D']
                
                # Filter GPS points for this trip
                trip_gps = gps_data[
                    (gps_data['LOCAL_DATETIME_parsed'] >= start_dt) & 
                    (gps_data['LOCAL_DATETIME_parsed'] <= end_dt)
                ].copy()
                
                if len(trip_gps) > 0:
                    total_trips_to_process += 1
                    total_weight_to_process += trip['Weight_Day']
                    
                    # Check if trip is within bounds
                    within_bounds = (
                        (trip_gps['LATITUDE'] >= self.idf_bounds['lat_min']) &
                        (trip_gps['LATITUDE'] <= self.idf_bounds['lat_max']) &
                        (trip_gps['LONGITUDE'] >= self.idf_bounds['lon_min']) &
                        (trip_gps['LONGITUDE'] <= self.idf_bounds['lon_max'])
                    ).all()
                    
                    if not within_bounds:
                        trips_outside_bounds += 1
                        weight_outside_bounds += trip['Weight_Day']
                        continue
                    
                    # Calculate metrics
                    gps_array = trip_gps[['LOCAL_DATETIME_parsed', 'LATITUDE', 
                                         'LONGITUDE', 'SPEED']].values
                    avg_speed, bird_distance, total_distance = self.metrics.compute_gps_metrics_numpy(gps_array)
                    
                    trip_data = {
                        'user_id': user_id,
                        'trip_id': trip['KEY'],
                        'category': trip['category'],
                        'trip_type': trip['trip_type'],
                        'datetime_O': start_dt,
                        'datetime_D': end_dt,
                        'duration_minutes': trip['Duration'],
                        'distance_km': total_distance,
                        'bird_distance_km': bird_distance,
                        'speed_kmh': avg_speed,
                        'weight': trip['Weight_Day'],
                        'gps_points': gps_array
                    }
                    merged_trips.append(trip_data)
        
        # Log statistics
        logger.info(f"Filter 5: Outside bounds")
        logger.info(f"  Trips with GPS data: {total_trips_to_process:,}")
        logger.info(f"  Trips removed: {trips_outside_bounds:,}")
        logger.info(f"  Remaining trips: {len(merged_trips):,}")
        
        # Filter speed outliers
        merged_trips = self._filter_speed_outliers(merged_trips)
        
        # Save merged trips
        with open(self.output_dir / 'merged_trips.pkl', 'wb') as f:
            pickle.dump(merged_trips, f)
        
        return merged_trips
    
    def _filter_speed_outliers(self, merged_trips: List[Dict], percentile: int = 99) -> List[Dict]:
        """Filter trips with speeds above percentile threshold for each mode."""
        from collections import defaultdict
        
        trips_by_mode = defaultdict(list)
        for trip in merged_trips:
            trips_by_mode[trip['trip_type']].append(trip)
        
        filtered_trips = []
        
        for mode, mode_trips in sorted(trips_by_mode.items()):
            speeds = [t['speed_kmh'] for t in mode_trips]
            
            if speeds:
                threshold = np.percentile(speeds, percentile)
                
                for trip in mode_trips:
                    if trip['speed_kmh'] <= threshold:
                        filtered_trips.append(trip)
        
        logger.info(f"Speed outlier filtering: {len(merged_trips)} -> {len(filtered_trips)} trips")
        
        return filtered_trips
    
    def _interpolate_and_resample(self, merged_trips: List[Dict]) -> List[Dict]:
        """Interpolate to 1s and resample to 2s."""
        logger.info("Interpolating and resampling...")
        
        interpolated_trips = []
        
        for trip in tqdm(merged_trips, desc="Interpolating trips"):
            gps_points = trip['gps_points']
            
            if len(gps_points) < 2:
                continue
            
            # Convert to DataFrame
            gps_df = pd.DataFrame(gps_points, columns=['timestamp', 'lat', 'lon', 'speed'])
            gps_df['timestamp'] = pd.to_datetime(gps_df['timestamp'])
            
            # Ensure numeric types for lat, lon, speed
            gps_df['lat'] = pd.to_numeric(gps_df['lat'], errors='coerce')
            gps_df['lon'] = pd.to_numeric(gps_df['lon'], errors='coerce')
            gps_df['speed'] = pd.to_numeric(gps_df['speed'], errors='coerce')
            
            # Remove any rows with NaN values after conversion
            gps_df = gps_df.dropna()
            
            # Sort and remove duplicates
            gps_df = gps_df.sort_values('timestamp').drop_duplicates('timestamp')
            
            if len(gps_df) < 2:
                continue
            
            # Create 1-second time range (use lowercase 's' for seconds)
            start_time = gps_df['timestamp'].min()
            end_time = gps_df['timestamp'].max()
            time_range = pd.date_range(start=start_time, end=end_time, freq='1s')
            
            if len(time_range) < 2:
                continue
            
            try:
                # Set timestamp as index
                gps_df.set_index('timestamp', inplace=True)
                
                # Create interpolated dataframe with numeric columns
                interpolated_df = pd.DataFrame(index=time_range)
                
                # Join and interpolate each column
                for col in ['lat', 'lon', 'speed']:
                    # Join the column
                    interpolated_df[col] = gps_df[col]
                    
                    # Interpolate (now guaranteed to be numeric)
                    interpolated_df[col] = interpolated_df[col].interpolate(method='linear')
                
                # Fill any remaining NaN values at the edges
                interpolated_df = interpolated_df.ffill().bfill()
                
                # Subsample to 2s (every other row)
                resampled_df = interpolated_df.iloc[::2].copy()
                
                # Update trip with interpolated data
                trip_interpolated = trip.copy()
                trip_interpolated['gps_points'] = resampled_df.reset_index().rename(columns={'index': 'timestamp'})[
                    ['timestamp', 'lat', 'lon', 'speed']
                ].values
                interpolated_trips.append(trip_interpolated)
                
            except Exception as e:
                logger.warning(f"Error interpolating trip {trip['trip_id']}: {str(e)}")
                continue
        
        logger.info(f"Interpolated {len(interpolated_trips)} trips")
        
        # Save interpolated data
        with open(self.output_dir / 'interpolated_trips.pkl', 'wb') as f:
            pickle.dump(interpolated_trips, f)
        
        return interpolated_trips
    
    def _cut_and_pad_sequences(self, interpolated_trips: List[Dict]) -> List[Dict]:
        """Cut sequences into chunks of fixed length and pad the last chunk."""
        logger.info(f"Cutting and padding sequences (length={self.sequence_length})...")
        
        processed_sequences = []
        
        for trip in tqdm(interpolated_trips, desc="Processing sequences"):
            gps_points = trip['gps_points']
            total_points = len(gps_points)
            
            # Cut into chunks
            n_chunks = int(np.ceil(total_points / self.sequence_length))
            
            for chunk_idx in range(n_chunks):
                start_idx = chunk_idx * self.sequence_length
                end_idx = min((chunk_idx + 1) * self.sequence_length, total_points)
                
                chunk_points = gps_points[start_idx:end_idx]
                chunk_length = len(chunk_points)
                
                # Pad if necessary
                if chunk_length < self.sequence_length:
                    padding_length = self.sequence_length - chunk_length
                    padding = np.zeros((padding_length, 4))
                    chunk_points = np.vstack([chunk_points, padding])
                    mask = np.zeros(self.sequence_length, dtype=bool)
                    mask[:chunk_length] = True
                else:
                    mask = np.ones(self.sequence_length, dtype=bool)
                
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
        
        logger.info(f"Created {len(processed_sequences)} sequences from {len(interpolated_trips)} trips")
        
        # Save processed sequences
        with open(self.output_dir / 'processed_sequences.pkl', 'wb') as f:
            pickle.dump(processed_sequences, f)
        
        return processed_sequences
    
    def _prepare_dataset(self, sequences: List[Dict]) -> Dict:
        """Prepare and scale data for model training."""
        logger.info("Preparing dataset...")
        
        n_sequences = len(sequences)
        n_features = 3  # lat, lon, speed
        
        # Initialize arrays
        trajectories = np.zeros((n_sequences, self.sequence_length, n_features))
        masks = np.zeros((n_sequences, self.sequence_length), dtype=bool)
        categories = []
        lengths = []
        weights = []
        
        for i, seq in enumerate(sequences):
            trajectories[i] = seq['gps_points'][:, 1:4]  # lat, lon, speed
            masks[i] = seq['mask']
            categories.append(seq['category'])
            lengths.append(seq['original_length'])
            weights.append(seq['weight'])
        
        # Encode categories
        category_encoder = LabelEncoder()
        categories_encoded = category_encoder.fit_transform(categories)
        
        # Scale features
        logger.info("Scaling features...")
        trajectories_flat = trajectories.reshape(-1, n_features)
        masks_flat = masks.reshape(-1)
        
        scaler = MinMaxScaler()
        valid_data = trajectories_flat[masks_flat]
        scaler.fit(valid_data)
        
        trajectories_scaled = scaler.transform(trajectories_flat).reshape(
            n_sequences, self.sequence_length, n_features
        )
        
        # Apply mask to scaled data
        for i in range(n_sequences):
            trajectories_scaled[i, ~masks[i]] = 0
        
        # Save scalers
        self.scalers['trajectory'] = scaler
        self.scalers['category_encoder'] = category_encoder
        
        dataset = {
            'trajectories': trajectories_scaled.astype(np.float32),
            'masks': masks,
            'categories': categories_encoded,
            'lengths': np.array(lengths),
            'weights': np.array(weights),
            'transport_modes': category_encoder.classes_.tolist()
        }
        
        return dataset
    
    def _save_outputs(self, dataset: Dict):
        """Save dataset and metadata."""
        # Save dataset
        np.savez_compressed(
            self.output_dir / 'dataset.npz',
            trajectories=dataset['trajectories'],
            masks=dataset['masks'],
            categories=dataset['categories'],
            lengths=dataset['lengths'],
            weights=dataset['weights']
        )
        
        # Save scalers
        with open(self.output_dir / 'scalers.pkl', 'wb') as f:
            pickle.dump(self.scalers, f)
        
        # Save metadata
        metadata = {
            'sequence_length': self.sequence_length,
            'n_sequences': len(dataset['trajectories']),
            'n_features': 3,
            'feature_names': ['latitude', 'longitude', 'speed'],
            'transport_modes': dataset['transport_modes'],
            'n_transport_modes': len(dataset['transport_modes']),
            'mode_categories': self.mode_categories,
            'scaling_method': 'MinMaxScaler',
            'idf_bounds': self.idf_bounds,
            'excluded_modes': self.excluded_modes,
            'filtering_stats': self.filtering_stats
        }
        
        with open(self.output_dir / 'metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"Dataset saved to {self.output_dir}")
    
    def _save_summary_report(self):
            """Save preprocessing summary report with dataset statistics."""
            # Load the saved dataset to compute statistics
            dataset_path = self.output_dir / 'dataset.npz'
            data = np.load(dataset_path)
            
            trajectories = data['trajectories']
            masks = data['masks']
            categories = data['categories']
            lengths = data['lengths']
            
            # Compute statistics per category using shared function
            category_stats = self.metrics.compute_category_statistics(
                trajectories, masks, categories, lengths
            )
            
            report = {
                'filtering_statistics': pd.DataFrame(self.filtering_stats),
                'category_statistics': category_stats,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            report['filtering_statistics'].to_csv(
                self.output_dir / 'preprocessing_summary.csv', index=False
            )
            
            with open(self.output_dir / 'preprocessing_report.pkl', 'wb') as f:
                pickle.dump(report, f)
            
            # Print category statistics using shared function
            self.metrics.print_category_statistics(category_stats, "Dataset Statistics by Category")
            
            logger.info("Preprocessing summary saved")