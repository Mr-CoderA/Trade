@echo off
REM Setup script for AED/CNY Trading System on Windows

echo.
echo ========================================
echo AED/CNY Trading System - Setup
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/
    exit /b 1
)

echo [1/4] Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    exit /b 1
)

echo [2/4] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    exit /b 1
)

echo [3/4] Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    exit /b 1
)

echo [4/4] Running diagnostics...
python setup.py
if errorlevel 1 (
    echo WARNING: Some diagnostics failed, but setup is complete
    exit /b 0
)

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Virtual environment is ready. To activate:
echo   venv\Scripts\activate
echo.
echo Then run the trading system:
echo   python main.py
echo.
echo Or start the dashboard:
echo   python ui/dashboard.py
echo.
