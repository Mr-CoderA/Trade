"""Common utilities and helper functions."""

import numpy as np
import pandas as pd
from typing import Union, List, Tuple
from datetime import datetime, timedelta


def calculate_returns(prices: pd.Series) -> pd.Series:
    """Calculate daily returns from price series.
    
    Args:
        prices: Price series
        
    Returns:
        Returns series (percentage change)
    """
    return prices.pct_change()


def calculate_log_returns(prices: pd.Series) -> pd.Series:
    """Calculate log returns from price series.
    
    Args:
        prices: Price series
        
    Returns:
        Log returns series
    """
    return np.log(prices / prices.shift(1))


def calculate_volatility(returns: pd.Series, window: int = 30) -> pd.Series:
    """Calculate rolling volatility of returns.
    
    Args:
        returns: Returns series
        window: Rolling window size
        
    Returns:
        Volatility series
    """
    return returns.rolling(window).std() * np.sqrt(252)  # Annualized


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """Calculate Sharpe ratio of returns.
    
    Args:
        returns: Returns series
        risk_free_rate: Risk-free rate (annual)
        
    Returns:
        Sharpe ratio
    """
    excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
    return np.sqrt(252) * (excess_returns.mean() / excess_returns.std())


def calculate_max_drawdown(returns: pd.Series) -> float:
    """Calculate maximum drawdown from returns.
    
    Args:
        returns: Returns series
        
    Returns:
        Maximum drawdown (as decimal, e.g., -0.20 for -20%)
    """
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()


def calculate_win_rate(trades: List[dict]) -> float:
    """Calculate win rate from list of trades.
    
    Args:
        trades: List of trade dictionaries with 'pnl' key
        
    Returns:
        Win rate (percentage of profitable trades)
    """
    if not trades:
        return 0.0
    
    wins = sum(1 for trade in trades if trade.get('pnl', 0) > 0)
    return wins / len(trades)


def calculate_profit_factor(trades: List[dict]) -> float:
    """Calculate profit factor (gross profit / gross loss).
    
    Args:
        trades: List of trade dictionaries with 'pnl' key
        
    Returns:
        Profit factor
    """
    if not trades:
        return 0.0
    
    gross_profit = sum(max(0, trade.get('pnl', 0)) for trade in trades)
    gross_loss = abs(sum(min(0, trade.get('pnl', 0)) for trade in trades))
    
    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 0.0
    
    return gross_profit / gross_loss


def format_timestamp(dt: datetime = None) -> str:
    """Format datetime to ISO string.
    
    Args:
        dt: Datetime object, defaults to current time
        
    Returns:
        ISO formatted string
    """
    if dt is None:
        dt = datetime.utcnow()
    
    return dt.strftime('%Y-%m-%d %H:%M:%S')


def parse_timestamp(ts: str) -> datetime:
    """Parse timestamp string to datetime.
    
    Args:
        ts: Timestamp string
        
    Returns:
        Datetime object
    """
    return datetime.strptime(ts, '%Y-%m-%d %H:%M:%S')


def seconds_to_hms(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format.
    
    Args:
        seconds: Number of seconds
        
    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def get_business_days(start_date: datetime, end_date: datetime) -> int:
    """Calculate business days between two dates.
    
    Args:
        start_date: Start date
        end_date: End date
        
    Returns:
        Number of business days
    """
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')
    return len(date_range)
