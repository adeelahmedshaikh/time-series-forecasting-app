# advanced_features.py
import pandas as pd
import numpy as np

def add_advanced_features(df):
    """Add sophisticated time-series features for better predictions"""
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # ===== TIME-BASED FEATURES =====
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['weekofyear'] = df['date'].dt.isocalendar().week.astype(int)
    df['days_in_month'] = df['date'].dt.days_in_month
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
    
    # Cyclical encoding for time features
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    
    # ===== LAG FEATURES (Multiple windows) =====
    for lag in [1, 2, 3, 7, 14, 21, 30]:
        df[f'lag_{lag}'] = df['value'].shift(lag)
    
    # ===== ROLLING STATISTICS =====
    for window in [3, 7, 14, 30]:
        df[f'rolling_mean_{window}'] = df['value'].rolling(window).mean()
        df[f'rolling_std_{window}'] = df['value'].rolling(window).std()
        df[f'rolling_min_{window}'] = df['value'].rolling(window).min()
        df[f'rolling_max_{window}'] = df['value'].rolling(window).max()
        df[f'rolling_range_{window}'] = df[f'rolling_max_{window}'] - df[f'rolling_min_{window}']
    
    # ===== EXPONENTIAL WEIGHTED =====
    df['ewm_7'] = df['value'].ewm(span=7, adjust=False).mean()
    df['ewm_30'] = df['value'].ewm(span=30, adjust=False).mean()
    
    # ===== RATE OF CHANGE =====
    df['pct_change_1'] = df['value'].pct_change(1)
    df['pct_change_7'] = df['value'].pct_change(7)
    df['pct_change_30'] = df['value'].pct_change(30)
    
    # ===== DIFFERENCING (for stationarity) =====
    df['diff_1'] = df['value'].diff(1)
    df['diff_7'] = df['value'].diff(7)
    
    # ===== CUMULATIVE FEATURES =====
    df['cumsum'] = df['value'].cumsum()
    df['cummax'] = df['value'].cummax()
    df['cummin'] = df['value'].cummin()
    
    # ===== INTERACTION FEATURES =====
    df['lag1_x_lag2'] = df['lag_1'] * df['lag_2']
    df['lag1_plus_lag2'] = df['lag_1'] + df['lag_2']
    
    # ===== TARGET ENCODING (simple version) =====
    # Previous day's value as percentage of rolling mean
    df['value_vs_rolling7'] = df['value'] / (df['rolling_mean_7'] + 1e-6)
    
    return df

def get_feature_importance_names():
    """Return list of all feature names for reference"""
    basic_features = ['lag_1', 'lag_2', 'lag_3', 'rolling_mean_3', 'rolling_std_3']
    time_features = ['dayofweek', 'month', 'quarter', 'is_weekend', 'month_sin', 'month_cos']
    advanced_lags = ['lag_7', 'lag_14', 'lag_30']
    rolling_features = ['rolling_mean_7', 'rolling_mean_14', 'rolling_std_7', 'rolling_range_7']
    rate_features = ['pct_change_1', 'pct_change_7']
    
    return basic_features + time_features + advanced_lags + rolling_features + rate_features