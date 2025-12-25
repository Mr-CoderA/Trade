"""Utilities package."""

from utils.logger import setup_logger
from utils.validators import detect_outliers, validate_price_data, fill_missing_data, normalize_data, denormalize_data
from utils.common import (
    calculate_returns, calculate_log_returns, calculate_volatility,
    calculate_sharpe_ratio, calculate_max_drawdown, calculate_win_rate,
    calculate_profit_factor, format_timestamp, parse_timestamp,
    seconds_to_hms, get_business_days
)
from utils.gpu_utils import (
    check_gpu_availability, get_gpu_details, setup_gpu_memory_growth,
    set_gpu_compute_capability, benchmark_gpu_vs_cpu, print_gpu_info,
    check_xgboost_gpu_support, check_lightgbm_gpu_support, diagnose_gpu_setup
)

__all__ = [
    'setup_logger',
    'detect_outliers',
    'validate_price_data',
    'fill_missing_data',
    'normalize_data',
    'denormalize_data',
    'calculate_returns',
    'calculate_log_returns',
    'calculate_volatility',
    'calculate_sharpe_ratio',
    'calculate_max_drawdown',
    'calculate_win_rate',
    'calculate_profit_factor',
    'format_timestamp',
    'parse_timestamp',
    'seconds_to_hms',
    'get_business_days',
    'check_gpu_availability',
    'get_gpu_details',
    'setup_gpu_memory_growth',
    'set_gpu_compute_capability',
    'benchmark_gpu_vs_cpu',
    'print_gpu_info',
    'check_xgboost_gpu_support',
    'check_lightgbm_gpu_support',
    'diagnose_gpu_setup',
]
