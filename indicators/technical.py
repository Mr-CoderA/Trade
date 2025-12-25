"""Technical indicators module for trading signals."""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
from utils.logger import setup_logger
from config import config

logger = setup_logger(__name__)


class TechnicalIndicators:
    """Calculate various technical indicators."""
    
    @staticmethod
    def moving_average(data: pd.Series, period: int) -> pd.Series:
        """Simple moving average.
        
        Args:
            data: Price series
            period: Window period
            
        Returns:
            Moving average series
        """
        return data.rolling(window=period).mean()
    
    @staticmethod
    def exponential_moving_average(data: pd.Series, period: int) -> pd.Series:
        """Exponential moving average.
        
        Args:
            data: Price series
            period: Window period
            
        Returns:
            EMA series
        """
        return data.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index.
        
        Args:
            data: Price series (typically close)
            period: Period for RSI calculation
            
        Returns:
            RSI series (0-100)
        """
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD (Moving Average Convergence Divergence).
        
        Args:
            data: Price series (typically close)
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period
            
        Returns:
            Tuple of (MACD line, Signal line, Histogram)
        """
        ema_fast = data.ewm(span=fast, adjust=False).mean()
        ema_slow = data.ewm(span=slow, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(data: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands.
        
        Args:
            data: Price series (typically close)
            period: Moving average period
            std_dev: Number of standard deviations
            
        Returns:
            Tuple of (Upper Band, Middle Band, Lower Band)
        """
        middle_band = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        
        upper_band = middle_band + (std_dev * std)
        lower_band = middle_band - (std_dev * std)
        
        return upper_band, middle_band, lower_band
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average True Range.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: ATR period
            
        Returns:
            ATR series
        """
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    @staticmethod
    def volume_analysis(volume: pd.Series, period: int = 20) -> pd.Series:
        """Volume moving average and analysis.
        
        Args:
            volume: Volume series
            period: Moving average period
            
        Returns:
            Volume MA series
        """
        return volume.rolling(window=period).mean()
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Stochastic Oscillator.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            k_period: K line period
            d_period: D line period
            
        Returns:
            Tuple of (K line, D line)
        """
        low_min = low.rolling(window=k_period).min()
        high_max = high.rolling(window=k_period).max()
        
        k_line = 100 * ((close - low_min) / (high_max - low_min + 1e-10))
        d_line = k_line.rolling(window=d_period).mean()
        
        return k_line, d_line
    
    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average Directional Index.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: ADX period
            
        Returns:
            ADX series
        """
        # Positive and negative directional indicators
        up_move = high.diff()
        down_move = -low.diff()
        
        pos_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
        neg_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)
        
        # True range
        tr = TechnicalIndicators.atr(high, low, close, 1)
        
        # Smoothed values
        pos_di = 100 * (pos_dm.rolling(period).sum() / tr.rolling(period).sum())
        neg_di = 100 * (neg_dm.rolling(period).sum() / tr.rolling(period).sum())
        
        # ADX
        dx = 100 * abs(pos_di - neg_di) / (pos_di + neg_di + 1e-10)
        adx = dx.rolling(period).mean()
        
        return adx


def calculate_all_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate all technical indicators for a dataframe.
    
    Args:
        data: Price dataframe with OHLCV columns
        
    Returns:
        DataFrame with all indicators
    """
    result = data.copy()
    config_indicators = config.get('indicators', {})
    
    logger.info("Calculating technical indicators...")
    
    # Moving Averages
    ma_periods = config_indicators.get('moving_averages', [20, 50, 200])
    for period in ma_periods:
        result[f'ma_{period}'] = TechnicalIndicators.moving_average(result['close'], period)
        result[f'ema_{period}'] = TechnicalIndicators.exponential_moving_average(result['close'], period)
    
    # RSI
    rsi_period = config_indicators.get('rsi_period', 14)
    result['rsi'] = TechnicalIndicators.rsi(result['close'], rsi_period)
    
    # MACD
    macd_config = config_indicators.get('macd', {})
    macd_line, signal_line, histogram = TechnicalIndicators.macd(
        result['close'],
        fast=macd_config.get('fast', 12),
        slow=macd_config.get('slow', 26),
        signal=macd_config.get('signal', 9)
    )
    result['macd'] = macd_line
    result['macd_signal'] = signal_line
    result['macd_hist'] = histogram
    
    # Bollinger Bands
    bb_period = config_indicators.get('bollinger_bands_period', 20)
    bb_std = config_indicators.get('bollinger_bands_std', 2.0)
    upper_bb, middle_bb, lower_bb = TechnicalIndicators.bollinger_bands(result['close'], bb_period, bb_std)
    result['bb_upper'] = upper_bb
    result['bb_middle'] = middle_bb
    result['bb_lower'] = lower_bb
    
    # ATR
    result['atr'] = TechnicalIndicators.atr(result['high'], result['low'], result['close'], 14)
    
    # Stochastic
    k_line, d_line = TechnicalIndicators.stochastic(result['high'], result['low'], result['close'])
    result['stoch_k'] = k_line
    result['stoch_d'] = d_line
    
    # ADX
    result['adx'] = TechnicalIndicators.adx(result['high'], result['low'], result['close'])
    
    # Volume analysis
    if 'volume' in result.columns:
        vol_period = config_indicators.get('volume_period', 20)
        result['volume_ma'] = TechnicalIndicators.volume_analysis(result['volume'], vol_period)
    
    logger.info(f"Calculated {len(result.columns) - len(data.columns)} indicators")
    return result
