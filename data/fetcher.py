"""Data fetching module for multiple data sources."""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional, Tuple
from abc import ABC, abstractmethod
import requests
from utils.logger import setup_logger
from config import config

logger = setup_logger(__name__)


class DataFetcher(ABC):
    """Abstract base class for data fetchers."""
    
    @abstractmethod
    def fetch(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch price data."""
        pass


class YFinanceFetcher(DataFetcher):
    """Fetch data using yfinance library."""
    
    def fetch(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch OHLCV data from yfinance.
        
        Args:
            symbol: Currency pair (e.g., 'AED=X' for AED/USD)
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            logger.info(f"Fetching data for {symbol} from {start_date} to {end_date}")
            data = yf.download(symbol, start=start_date, end=end_date, progress=False)
            
            if data.empty:
                logger.warning(f"No data fetched for {symbol}")
                return pd.DataFrame()
            
            # Standardize column names
            data.columns = [col.lower() for col in data.columns]
            logger.info(f"Successfully fetched {len(data)} rows for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data from yfinance: {e}")
            return pd.DataFrame()


class AlphaVantageFetcher(DataFetcher):
    """Fetch data using Alpha Vantage API."""
    
    def __init__(self, api_key: str):
        """Initialize with API key."""
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
    
    def fetch(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch OHLCV data from Alpha Vantage.
        
        Args:
            symbol: Currency pair (e.g., 'AED', 'CNY')
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            logger.info(f"Fetching data for {symbol} from Alpha Vantage")
            
            params = {
                'function': 'FX_DAILY',
                'from_symbol': symbol.split('/')[0],
                'to_symbol': symbol.split('/')[1],
                'apikey': self.api_key,
                'outputsize': 'full'
            }
            
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data_dict = response.json()
            
            if 'Time Series FX (Daily)' not in data_dict:
                logger.error(f"Error in API response: {data_dict}")
                return pd.DataFrame()
            
            ts_data = data_dict['Time Series FX (Daily)']
            
            # Convert to DataFrame
            data = pd.DataFrame.from_dict(ts_data, orient='index')
            data.columns = ['open', 'high', 'low', 'close']
            data.index = pd.to_datetime(data.index)
            data = data.astype(float)
            
            # Filter by date range
            data = data[(data.index >= start_date) & (data.index <= end_date)]
            
            logger.info(f"Successfully fetched {len(data)} rows for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data from Alpha Vantage: {e}")
            return pd.DataFrame()


class ScreenOCRFetcher(DataFetcher):
    """Placeholder for screen OCR data fetcher."""
    
    def fetch(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch data from screen captures using OCR.
        
        Note: This is a placeholder. Implementation in ocr module.
        """
        logger.warning("ScreenOCRFetcher.fetch() not yet implemented")
        return pd.DataFrame()


def get_data_fetcher(source: str = None) -> DataFetcher:
    """Factory function to get appropriate data fetcher.
    
    Args:
        source: Data source ('yfinance', 'alpha_vantage', 'screen_ocr')
        
    Returns:
        DataFetcher instance
    """
    if source is None:
        source = config.get_nested('data.data_source', 'yfinance')
    
    if source == 'yfinance':
        return YFinanceFetcher()
    elif source == 'alpha_vantage':
        api_key = config.get_nested('data.alpha_vantage_api_key')
        if not api_key:
            logger.warning("Alpha Vantage API key not configured, falling back to yfinance")
            return YFinanceFetcher()
        return AlphaVantageFetcher(api_key)
    elif source == 'screen_ocr':
        return ScreenOCRFetcher()
    else:
        logger.warning(f"Unknown data source: {source}, using yfinance")
        return YFinanceFetcher()


def fetch_historical_data(symbol: str, days_back: int = 365) -> Tuple[pd.DataFrame, bool]:
    """Fetch historical data for a given symbol.
    
    Args:
        symbol: Currency pair
        days_back: Number of days to fetch
        
    Returns:
        Tuple of (DataFrame, is_valid)
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    fetcher = get_data_fetcher()
    data = fetcher.fetch(symbol, start_date, end_date)
    
    is_valid = not data.empty
    return data, is_valid
