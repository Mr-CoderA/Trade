"""Data validation utilities."""

import numpy as np
import pandas as pd
from typing import Tuple, List
from utils.logger import setup_logger

logger = setup_logger(__name__)


def detect_outliers(data: pd.Series, method: str = 'iqr', threshold: float = 3.0) -> np.ndarray:
    """Detect outliers in time series data.
    
    Args:
        data: Time series data
        method: 'iqr' (Interquartile Range) or 'zscore'
        threshold: Threshold for outlier detection
        
    Returns:
        Boolean array indicating outliers
    """
    if method == 'iqr':
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        outliers = (data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))
    elif method == 'zscore':
        outliers = np.abs((data - data.mean()) / data.std()) > threshold
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return outliers.values


def validate_price_data(df: pd.DataFrame, required_cols: List[str] = None) -> Tuple[bool, str]:
    """Validate price data for quality issues.
    
    Args:
        df: Price dataframe
        required_cols: List of required columns
        
    Returns:
        Tuple of (is_valid, message)
    """
    if required_cols is None:
        required_cols = ['open', 'high', 'low', 'close', 'volume']
    
    # Check required columns
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        return False, f"Missing columns: {missing_cols}"
    
    # Check for NaN values
    if df[required_cols].isna().any().any():
        return False, "Data contains NaN values"
    
    # Check for negative prices
    if (df[['open', 'high', 'low', 'close']] <= 0).any().any():
        return False, "Data contains non-positive prices"
    
    # Check OHLC relationships
    if (df['high'] < df['low']).any():
        return False, "High prices lower than low prices"
    
    if (df['high'] < df['close']).any() or (df['low'] > df['close']).any():
        return False, "Close prices outside high-low range"
    
    # Check volume
    if (df['volume'] < 0).any():
        return False, "Data contains negative volumes"
    
    return True, "Data validation passed"


def fill_missing_data(df: pd.DataFrame, method: str = 'forward_fill') -> pd.DataFrame:
    """Fill missing data in price dataframe.
    
    Args:
        df: Price dataframe
        method: 'forward_fill', 'interpolate', or 'drop'
        
    Returns:
        Dataframe with filled missing values
    """
    df = df.copy()
    
    if method == 'forward_fill':
        df = df.fillna(method='ffill').fillna(method='bfill')
    elif method == 'interpolate':
        df = df.interpolate(method='linear')
    elif method == 'drop':
        df = df.dropna()
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return df


def normalize_data(data: pd.DataFrame, columns: List[str] = None) -> Tuple[pd.DataFrame, dict]:
    """Normalize data for ML models.
    
    Args:
        data: Input dataframe
        columns: Columns to normalize
        
    Returns:
        Tuple of (normalized_data, normalization_params)
    """
    data = data.copy()
    if columns is None:
        columns = data.columns
    
    normalization_params = {}
    
    for col in columns:
        mean = data[col].mean()
        std = data[col].std()
        data[col] = (data[col] - mean) / std
        normalization_params[col] = {'mean': mean, 'std': std}
    
    return data, normalization_params


def denormalize_data(data: pd.DataFrame, normalization_params: dict) -> pd.DataFrame:
    """Denormalize data using saved normalization parameters.
    
    Args:
        data: Normalized data
        normalization_params: Dictionary of normalization parameters
        
    Returns:
        Denormalized data
    """
    data = data.copy()
    
    for col, params in normalization_params.items():
        if col in data.columns:
            data[col] = data[col] * params['std'] + params['mean']
    
    return data
