# cache_manager.py
import joblib
import hashlib
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

class ForecastCache:
    def __init__(self, cache_dir="models/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
    
    def get_cache_key(self, df, horizon, model_name):
        """Create unique key based on data and parameters"""
        # Hash the dataframe content
        df_hash = hashlib.md5(
            pd.util.hash_pandas_object(df, index=True).values
        ).hexdigest()
        
        # Combine with parameters
        key_string = f"{df_hash}_{horizon}_{model_name}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get_cached_forecast(self, df, horizon, model_name, max_age_hours=24):
        """Retrieve cached forecast if it exists and is fresh"""
        cache_key = self.get_cache_key(df, horizon, model_name)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if cache_file.exists():
            # Check age
            file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if file_age < timedelta(hours=max_age_hours):
                try:
                    cached_data = joblib.load(cache_file)
                    return cached_data
                except:
                    pass
        return None
    
    def save_forecast(self, df, horizon, model_name, forecast_df):
        """Save forecast to cache"""
        cache_key = self.get_cache_key(df, horizon, model_name)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        # Only save if not too big
        if len(forecast_df) < 1000:
            joblib.dump(forecast_df, cache_file)
    
    def clear_old_cache(self, max_age_days=7):
        """Remove cache files older than max_age_days"""
        now = datetime.now()
        for cache_file in self.cache_dir.glob("*.pkl"):
            file_age = now - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if file_age > timedelta(days=max_age_days):
                cache_file.unlink()
    
    def get_cache_stats(self):
        """Get cache statistics"""
        files = list(self.cache_dir.glob("*.pkl"))
        return {
            "total_cached": len(files),
            "cache_size_mb": sum(f.stat().st_size for f in files) / (1024 * 1024),
            "oldest_cache": min((f.stat().st_mtime for f in files), default=None)
        }