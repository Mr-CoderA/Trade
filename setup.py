#!/usr/bin/env python3
"""
Complete setup and initialization script for AED/CNY Trading System.
This script verifies all dependencies and initializes the system.
"""

import sys
import subprocess
import os
from pathlib import Path

def print_header(text):
    """Print formatted header."""
    print("\n" + "="*80)
    print(f" {text}")
    print("="*80)

def print_success(text):
    """Print success message."""
    print(f"✓ {text}")

def print_error(text):
    """Print error message."""
    print(f"✗ {text}")

def print_warning(text):
    """Print warning message."""
    print(f"⚠ {text}")

def check_python_version():
    """Check Python version."""
    print_header("Checking Python Version")
    
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print_success(f"Python {version.major}.{version.minor}.{version.micro} detected")
        return True
    else:
        print_error(f"Python 3.8+ required, found {version.major}.{version.minor}")
        return False

def check_venv():
    """Check if virtual environment is activated."""
    print_header("Checking Virtual Environment")
    
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print_success("Virtual environment is activated")
        return True
    else:
        print_warning("Virtual environment not activated")
        print("Run: venv\\Scripts\\activate")
        return False

def verify_dependencies():
    """Verify all required packages are installed."""
    print_header("Verifying Dependencies")
    
    critical_packages = [
        'numpy', 'pandas', 'scikit-learn', 'tensorflow',
        'xgboost', 'lightgbm', 'yfinance', 'dash', 'plotly'
    ]
    
    missing = []
    for package in critical_packages:
        try:
            __import__(package)
            print_success(f"{package} installed")
        except ImportError:
            print_error(f"{package} not installed")
            missing.append(package)
    
    if missing:
        print_warning(f"Missing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    return True

def check_data_directory():
    """Check if data directory exists and is writable."""
    print_header("Checking Data Directory")
    
    data_dir = Path('data')
    if data_dir.exists() and data_dir.is_dir():
        print_success("Data directory exists")
        return True
    else:
        print_error("Data directory not found")
        return False

def check_models_directory():
    """Check if models directory exists."""
    print_header("Checking Models Directory")
    
    models_dir = Path('models')
    if not models_dir.exists():
        models_dir.mkdir(parents=True, exist_ok=True)
        print_success("Created models directory")
    else:
        print_success("Models directory exists")
    
    return True

def check_logs_directory():
    """Check if logs directory exists."""
    print_header("Checking Logs Directory")
    
    logs_dir = Path('logs')
    if not logs_dir.exists():
        logs_dir.mkdir(parents=True, exist_ok=True)
        print_success("Created logs directory")
    else:
        print_success("Logs directory exists")
    
    return True

def verify_config():
    """Verify configuration file exists."""
    print_header("Verifying Configuration")
    
    config_file = Path('config/settings.yaml')
    if config_file.exists():
        print_success("Configuration file found")
        return True
    else:
        print_error("Configuration file not found")
        return False

def test_imports():
    """Test critical module imports."""
    print_header("Testing Module Imports")
    
    modules = [
        ('config', 'config'),
        ('utils', 'utils'),
        ('data', 'data'),
        ('indicators', 'indicators'),
        ('models', 'models'),
        ('risk', 'risk'),
        ('monitoring', 'monitoring'),
        ('backtesting', 'backtesting'),
    ]
    
    success = True
    for module_name, display_name in modules:
        try:
            __import__(module_name)
            print_success(f"{display_name} module imported")
        except ImportError as e:
            print_error(f"{display_name} module import failed: {e}")
            success = False
    
    return success

def test_data_fetching():
    """Test data fetching capability."""
    print_header("Testing Data Fetching")
    
    try:
        import yfinance as yf
        from datetime import datetime, timedelta
        
        # Attempt to fetch small sample
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        print("Fetching 30 days of sample data...")
        data = yf.download('AED=X', start=start_date, end=end_date, progress=False)
        
        if len(data) > 0:
            print_success(f"Successfully fetched {len(data)} rows of data")
            return True
        else:
            print_error("No data fetched")
            return False
            
    except Exception as e:
        print_error(f"Data fetching test failed: {e}")
        return False

def print_next_steps():
    """Print next steps for user."""
    print_header("Next Steps")
    
    print("""
1. CONFIGURE THE SYSTEM:
   - Edit config/settings.yaml
   - Set your account balance, risk parameters, and model settings
   - Choose data source (yfinance, alpha_vantage, or screen_ocr)

2. TRAIN THE MODEL:
   - Run: python main.py
   - This will fetch 5 years of historical data, train models, and backtest

3. VIEW THE DASHBOARD:
   - Run: python ui/dashboard.py
   - Open browser to http://localhost:8050
   - Monitor real-time signals and performance

4. CUSTOMIZE STRATEGY:
   - Modify indicators in indicators/technical.py
   - Adjust model in models/ml_models.py
   - Change signal logic in monitoring/signal_generator.py

5. LIVE TRADING (FUTURE):
   - Add broker API integration
   - Implement order execution
   - Start paper trading before live trading

6. MONITOR PERFORMANCE:
   - Check logs in logs/trading_system.log
   - Review backtesting results
   - Track prediction accuracy over time
    """)

def run_diagnostics():
    """Run all diagnostic checks."""
    print_header("AED/CNY Trading System - Initialization Diagnostics")
    
    checks = [
        ("Python Version", check_python_version),
        ("Virtual Environment", check_venv),
        ("Dependencies", verify_dependencies),
        ("Data Directory", check_data_directory),
        ("Models Directory", check_models_directory),
        ("Logs Directory", check_logs_directory),
        ("Configuration", verify_config),
        ("Module Imports", test_imports),
        ("Data Fetching", test_data_fetching),
    ]
    
    results = {}
    for check_name, check_func in checks:
        try:
            results[check_name] = check_func()
        except Exception as e:
            print_error(f"{check_name} check failed with error: {e}")
            results[check_name] = False
    
    # Summary
    print_header("Diagnostic Summary")
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"\nChecks Passed: {passed}/{total}")
    
    if passed == total:
        print_success("All checks passed! System is ready to use.")
        print_next_steps()
        return 0
    else:
        print_warning("Some checks failed. Please resolve issues above.")
        return 1

if __name__ == '__main__':
    exit_code = run_diagnostics()
    sys.exit(exit_code)
