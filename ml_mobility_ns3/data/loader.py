import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class NetMob25Loader:
    """Simple data loader for NetMob25 dataset."""
    
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.individuals_df = None
        self.trips_df = None
        self.gps_data = {}
        
    def load_individuals(self) -> pd.DataFrame:
        """Load individuals dataset."""
        path = self.data_dir / "individuals_dataset.csv"
        logger.info(f"Loading individuals from {path}")
        self.individuals_df = pd.read_csv(path)
        
        # Drop temporary columns if they exist (from fake data generation)
        temp_cols = ['home_lat', 'home_lon']
        existing_temp_cols = [col for col in temp_cols if col in self.individuals_df.columns]
        if existing_temp_cols:
            self.individuals_df = self.individuals_df.drop(columns=existing_temp_cols)
        
        return self.individuals_df
    
    def load_trips(self) -> pd.DataFrame:
        """Load trips dataset."""
        path = self.data_dir / "trips_dataset.csv"
        logger.info(f"Loading trips from {path}")
        self.trips_df = pd.read_csv(path)
        # Convert time columns to datetime
        time_cols = ['Date_O', 'Time_O', 'Date_D', 'Time_D']
        for col in time_cols:
            if col in self.trips_df.columns:
                self.trips_df[col] = pd.to_datetime(self.trips_df[col], errors='coerce')
        
        # Drop temporary columns if they exist (from fake data generation)
        temp_cols = ['origin_lat', 'origin_lon', 'dest_lat', 'dest_lon']
        existing_temp_cols = [col for col in temp_cols if col in self.trips_df.columns]
        if existing_temp_cols:
            self.trips_df = self.trips_df.drop(columns=existing_temp_cols)
        
        return self.trips_df
    
    def load_gps_trace(self, user_id: str) -> pd.DataFrame:
        """Load GPS trace for a specific user."""
        path = self.data_dir / "gps_dataset" / f"{user_id}.csv"
        if not path.exists():
            logger.warning(f"GPS file not found for user {user_id}")
            return pd.DataFrame()
        
        df = pd.read_csv(path)
        df['UTC DATETIME'] = pd.to_datetime(df['UTC DATETIME'])
        return df
    
    def get_user_trips(self, user_id: str) -> pd.DataFrame:
        """Get all trips for a specific user."""
        if self.trips_df is None:
            self.load_trips()
        return self.trips_df[self.trips_df['ID'] == user_id]
    
    def get_trip_gps_points(self, user_id: str, trip_key: str) -> pd.DataFrame:
        """Get GPS points for a specific trip."""
        trips = self.get_user_trips(user_id)
        trip = trips[trips['KEY'] == trip_key]

        if trip.empty:
            return pd.DataFrame()
        
        gps = self.load_gps_trace(user_id)

        if gps.empty:
            return pd.DataFrame()
        
        return gps

        # Filter GPS points by trip time window
        start_time = trip.iloc[0]['Time_O']
        end_time = trip.iloc[0]['Time_D']

        mask = (gps['UTC DATETIME'] >= start_time) & (gps['UTC DATETIME'] <= end_time)

        return gps[mask]
    
    def sample_trajectories(self, n_samples: int = 100, min_points: int = 10) -> List[pd.DataFrame]:
        """Sample random trajectories from the dataset."""
        if self.trips_df is None:
            self.load_trips()
            
        trajectories = []
        sampled_trips = self.trips_df.sample(n=min(n_samples * 2, len(self.trips_df)))
        
        for _, trip in sampled_trips.iterrows():
            if len(trajectories) >= n_samples:
                break
                
            gps_points = self.get_trip_gps_points(trip['ID'], trip['KEY'])
            if len(gps_points) >= min_points:
                trajectories.append(gps_points)
                
        logger.info(f"Sampled {len(trajectories)} trajectories")
        return trajectories