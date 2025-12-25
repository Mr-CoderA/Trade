"""Data preprocessing module."""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from utils.logger import setup_logger
from utils.validators import detect_outliers, validate_price_data, fill_missing_data

logger = setup_logger(__name__)


class DataPreprocessor:
    """Preprocess and clean price data."""
    
    def __init__(self, outlier_method: str = 'iqr', fill_method: str = 'forward_fill'):
        """Initialize preprocessor.
        
        Args:
            outlier_method: Method for outlier detection ('iqr' or 'zscore')
            fill_method: Method for filling missing values
        """
        self.outlier_method = outlier_method
        self.fill_method = fill_method
    
    def preprocess(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        """Complete preprocessing pipeline.
        
        Args:
            data: Raw price data
            
        Returns:
            Tuple of (processed_data, preprocessing_info)
        """
        preprocessing_info = {}
        
        logger.info("Starting data preprocessing...")
        
        # Validate data
        is_valid, message = validate_price_data(data)
        preprocessing_info['validation'] = {'is_valid': is_valid, 'message': message}
        
        if not is_valid:
            logger.warning(f"Data validation failed: {message}")
        
        # Create copy to avoid modifying original
        data = data.copy()
        
        # Fill missing values
        logger.info(f"Filling missing values using {self.fill_method}...")
        data = fill_missing_data(data, method=self.fill_method)
        preprocessing_info['missing_values_filled'] = True
        
        # Remove duplicates
        initial_rows = len(data)
        data = data[~data.index.duplicated(keep='first')]
        duplicates_removed = initial_rows - len(data)
        preprocessing_info['duplicates_removed'] = duplicates_removed
        logger.info(f"Removed {duplicates_removed} duplicate rows")
        
        # Detect and handle outliers
        outliers_by_col = {}
        for col in ['open', 'high', 'low', 'close']:
            if col in data.columns:
                outliers = detect_outliers(data[col], method=self.outlier_method)
                outliers_by_col[col] = outliers.sum()
                
                if outliers.sum() > 0:
                    logger.warning(f"Found {outliers.sum()} outliers in {col}")
                    # Replace outliers with interpolated values
                    data.loc[outliers, col] = np.nan
                    data[col] = data[col].interpolate(method='linear')
        
        preprocessing_info['outliers_detected'] = outliers_by_col
        
        # Sort by index
        data = data.sort_index()
        
        # Remove any remaining NaN values
        initial_len = len(data)
        data = data.dropna()
        removed = initial_len - len(data)
        preprocessing_info['rows_with_nan_removed'] = removed
        
        logger.info(f"Data preprocessing complete. Final rows: {len(data)}")
        return data, preprocessing_info
    
    def add_derived_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add derived features to price data.
        
        Args:
            data: Price dataframe
            
        Returns:
            DataFrame with additional features
        """
        data = data.copy()
        
        # Price features
        data['returns'] = data['close'].pct_change()
        data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        data['price_range'] = data['high'] - data['low']
        data['hl_ratio'] = data['high'] / (data['low'] + 1e-8)
        
        # OHLC relationships
        data['close_position'] = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
        
        # Volatility
        data['volatility_10'] = data['returns'].rolling(10).std()
        data['volatility_30'] = data['returns'].rolling(30).std()
        
        # Volume features
        if 'volume' in data.columns:
            data['volume_change'] = data['volume'].pct_change()
            data['volume_ma'] = data['volume'].rolling(20).mean()
        
        logger.info("Added derived features to data")
        return data


def engineer_features(data: pd.DataFrame, lookback_window: int = 60) -> pd.DataFrame:
    """Engineer features for ML models.
    
    Args:
        data: Price dataframe
        lookback_window: Number of days to look back for features
        
    Returns:
        DataFrame with engineered features
    """
    data = data.copy()
    
    # Historical statistics
    for window in [5, 10, 20, 60]:
        data[f'returns_mean_{window}'] = data['returns'].rolling(window).mean()
        data[f'returns_std_{window}'] = data['returns'].rolling(window).std()
        data[f'close_min_{window}'] = data['close'].rolling(window).min()
        data[f'close_max_{window}'] = data['close'].rolling(window).max()
    
    # Momentum indicators
    data['momentum_5'] = data['close'].diff(5)
    data['momentum_10'] = data['close'].diff(10)
    data['momentum_20'] = data['close'].diff(20)
    
    # Rate of change
    data['roc_5'] = data['close'].pct_change(5)
    data['roc_10'] = data['close'].pct_change(10)
    data['roc_20'] = data['close'].pct_change(20)
    
    # Drop initial NaN rows
    data = data.dropna()
    
    logger.info(f"Engineered features. Data shape: {data.shape}")
    return data
