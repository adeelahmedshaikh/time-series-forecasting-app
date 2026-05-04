# validation.py
import pandas as pd
import streamlit as st
from datetime import datetime

def validate_data(df, min_rows=14, max_missing_percent=30):
    """
    Validate dataset before forecasting
    Returns: (is_valid, error_message)
    """
    
    # Check if dataframe is empty
    if df is None or df.empty:
        return False, "Dataset is empty. Please upload valid data."
    
    # Check minimum rows (need at least min_rows for lag features)
    if len(df) < min_rows:
        return False, f"Need at least {min_rows} data points. You have {len(df)}. Please add more historical data."
    
    # Check for too many missing values
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isnull().sum().sum()
    missing_percent = (missing_cells / total_cells) * 100 if total_cells > 0 else 100
    
    if missing_percent > max_missing_percent:
        return False, f"Too many missing values ({missing_percent:.1f}%). Please clean your data first (remove rows with N/A)."
    
    # Check date column exists and is datetime
    if 'date' not in df.columns:
        return False, "Date column not found. Please ensure your data has a date column."
    
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        try:
            df['date'] = pd.to_datetime(df['date'])
        except:
            return False, "Date column must be convertible to datetime format. Example: 2024-01-01"
    
    # Check value column exists and is numeric
    if 'value' not in df.columns:
        return False, "Target column not found. Please ensure your data has a numeric column to forecast."
    
    if not pd.api.types.is_numeric_dtype(df['value']):
        return False, "Target column must be numeric (numbers only). Remove any text or special characters."
    
    # Check for constant values (no variation)
    if df['value'].std() == 0:
        return False, "Target column has no variation (all values are the same). Cannot forecast."
    
    # Check for infinite values
    if df['value'].isin([float('inf'), float('-inf')]).any():
        return False, "Dataset contains infinite values. Please remove or replace them."
    
    # Check date range is reasonable
    date_range = (df['date'].max() - df['date'].min()).days
    if date_range < 7:
        st.warning(f"⚠️ Your data only spans {date_range} days. Forecasts may be less accurate.")
    
    return True, "Data is valid"

def get_data_quality_report(df):
    """Generate a data quality report"""
    if df.empty or len(df) < 2:
        return {"error": "Insufficient data for report"}
    
    # Check for gaps in dates
    date_diff = df['date'].diff().dropna()
    if not date_diff.empty:
        most_common_freq = date_diff.mode()[0] if not date_diff.mode().empty else None
        gaps = (date_diff > most_common_freq).sum() if most_common_freq else 0
        has_gaps = gaps > 0
    else:
        has_gaps = False
    
    report = {
        "total_rows": len(df),
        "missing_values": int(df.isnull().sum().sum()),
        "missing_percent": float((df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100),
        "date_range": f"{df['date'].min().date()} to {df['date'].max().date()}",
        "days_span": (df['date'].max() - df['date'].min()).days,
        "value_mean": float(df['value'].mean()),
        "value_std": float(df['value'].std()),
        "value_min": float(df['value'].min()),
        "value_max": float(df['value'].max()),
        "has_date_gaps": has_gaps,
        "gap_count": int(gaps) if has_gaps else 0
    }
    return report

def check_date_continuity(df):
    """Check if dates have gaps"""
    if len(df) < 2:
        return False
    
    date_diff = df['date'].diff().dropna()
    if date_diff.empty:
        return False
    
    expected_diff = date_diff.mode()[0] if not date_diff.mode().empty else None
    if expected_diff is None:
        return False
    
    gaps = (date_diff > expected_diff).sum()
    return gaps == 0

def suggest_data_fixes(df):
    """Suggest fixes for common data issues"""
    suggestions = []
    
    # Check for missing values
    if df.isnull().sum().sum() > 0:
        suggestions.append("• Remove rows with missing values using df.dropna()")
    
    # Check for duplicates
    if df.duplicated(subset=['date']).any():
        suggestions.append("• Remove duplicate dates (keep first occurrence)")
    
    # Check for outliers
    if len(df) > 0:
        z_scores = abs((df['value'] - df['value'].mean()) / df['value'].std())
        if (z_scores > 5).any():
            suggestions.append("• Consider removing extreme outliers (values > 5 standard deviations)")
    
    # Check for negative values in positive-only data
    if (df['value'] < 0).any():
        suggestions.append("• Negative values detected. Consider if these are valid for your use case")
    
    return suggestions