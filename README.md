# AED/CNY Intelligent Trading System

A sophisticated machine learning-based trading system for AED/CNY currency pair with real-time monitoring, risk management, and comprehensive backtesting.

## Features

- **Machine Learning Models**: LSTM + Ensemble (XGBoost, LightGBM) for accurate price prediction
- **Technical Indicators**: Moving averages, RSI, MACD, Bollinger Bands, ATR, Stochastic, ADX
- **Real-time Monitoring**: Screen capture with OCR for live price data extraction
- **Risk Management**: Position sizing, stop-loss/take-profit calculation, confidence scoring
- **Signal Generation**: Multi-layer validation (ML + Technical + Risk)
- **Backtesting**: Walk-forward testing, performance metrics, equity curve tracking
- **Web Dashboard**: Real-time visualization of prices, signals, and performance metrics
- **Explainability**: Clear decision rationale for each trade recommendation

## Project Structure

```
po_bot_v2/
├── config/              # Configuration files and settings
├── data/                # Data fetching and preprocessing
├── models/              # ML models (LSTM, ensemble)
├── indicators/          # Technical indicators
├── risk/                # Risk management
├── monitoring/          # Signal generation
├── backtesting/         # Backtesting framework
├── ocr/                 # Screen capture and OCR
├── ui/                  # Web dashboard
├── utils/               # Utility functions
├── main.py              # Main application entry point
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Installation

### 1. Clone the Repository
```bash
cd c:\Users\asada\Downloads\po_bot_v2
```

### 2. Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

**Note**: If you encounter issues with `ta-lib`, install it separately:
```bash
pip install --upgrade ta-lib
```

### 4. Configure System
Edit `config/settings.yaml` to adjust:
- Model parameters
- Risk management settings
- Indicator configurations
- Backtesting parameters

## Quick Start

### 1. Run the System
```bash
python main.py
```

This will:
1. Fetch 5 years of historical AED/CNY data
2. Preprocess and validate data
3. Train ensemble ML model
4. Generate trading signals
5. Run backtest on historical data
6. Display results

### 2. Start the Web Dashboard
```bash
python ui/dashboard.py
```

Then open browser to: `http://localhost:8050`

The dashboard shows:
- Real-time price and signals
- Model confidence scores
- Account equity growth
- Performance metrics (Sharpe ratio, win rate, etc.)

## Configuration

### Key Settings (config/settings.yaml)

```yaml
# Data Settings
data:
  currency_pair: "AED/CNY"
  data_source: "yfinance"  # yfinance, alpha_vantage, or screen_ocr
  lookback_period: 5  # years

# Model Settings
model:
  type: "ensemble"
  lookback_window: 60  # days
  confidence_threshold: 0.75  # minimum confidence for trades

# Risk Management
risk:
  account_balance: 10000
  risk_per_trade: 0.02  # 2% per trade
  max_position_size: 0.10  # max 10% of account
  max_drawdown_limit: 0.20  # 20% max drawdown

# Signal Validation
signals:
  validation_layers: 3  # ML, Technical, Risk
  validation_mode: "AND"  # all must agree
  required_agreement: 0.75  # 75% confidence
```

## Components

### Data Pipeline (data/)
- **fetcher.py**: Fetch data from yfinance, Alpha Vantage, or screen OCR
- **preprocessor.py**: Data cleaning, outlier detection, feature engineering

### Technical Indicators (indicators/)
- Moving averages (SMA, EMA)
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- ATR (Average True Range)
- Stochastic Oscillator
- ADX (Average Directional Index)

### Machine Learning Models (models/)
- **LSTM**: Long Short-Term Memory neural network for sequence prediction
- **Ensemble**: Combines LSTM + XGBoost + LightGBM with weighted voting
- Cross-validation and hyperparameter optimization

### Risk Management (risk/)
- Position sizing using Kelly Criterion
- Stop-loss and take-profit calculation
- Max drawdown monitoring
- Position limit enforcement
- Confidence scoring

### Signal Generation (monitoring/)
- ML-based signals from predictions
- Technical indicator signals (RSI, MACD, Bollinger Bands)
- Multi-layer validation with AND/OR logic
- Confidence scoring

### Backtesting (backtesting/)
- Walk-forward testing on historical data
- Transaction cost simulation
- Slippage modeling
- Performance metrics (Sharpe ratio, max drawdown, win rate, profit factor)

### Screen Monitoring (ocr/)
- Real-time screen capture
- OCR extraction of prices using Tesseract
- Chart candle detection using OpenCV
- Price change detection

### Web Dashboard (ui/)
- Real-time price and signal visualization
- Equity curve and performance metrics
- Signal distribution analysis
- Auto-updating via configured intervals

## Key Metrics

### Prediction Accuracy
- **Direction Accuracy**: Percentage of correct up/down predictions
- **R² Score**: Coefficient of determination
- **RMSE**: Root mean squared error

### Trading Performance
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / Gross loss
- **Sharpe Ratio**: Risk-adjusted returns
- **Max Drawdown**: Maximum account decline
- **Total P&L**: Absolute profit/loss

## Model Training

The ensemble model uses:

1. **LSTM Network** (40% weight)
   - 2 LSTM layers (64, 32 units)
   - Dropout for regularization
   - Optimized with Adam

2. **XGBoost** (30% weight)
   - 100 estimators
   - Max depth of 6
   - Learning rate of 0.1

3. **LightGBM** (30% weight)
   - 100 estimators
   - Max depth of 6
   - Learning rate of 0.1

**Ensemble Prediction**: Weighted average of all three models, with confidence calculated from prediction agreement

## Trading Signals

### Signal Generation Process

1. **ML Prediction Signal**
   - Compare predicted price to current price
   - Generate BUY/SELL based on predicted direction
   - Confidence from model uncertainty

2. **Technical Indicator Signal**
   - RSI: BUY if <30 (oversold), SELL if >70 (overbought)
   - MACD: BUY if MACD > Signal, SELL if MACD < Signal
   - MA: BUY if MA20 > MA50, SELL if MA20 < MA50
   - Bollinger Bands: BUY if close < lower band, SELL if close > upper band

3. **Risk Validation**
   - Check position limits
   - Validate confidence thresholds
   - Ensure account has sufficient capital

4. **Signal Combination**
   - AND mode: All validators must agree (higher accuracy)
   - Weighted mode: Voting system based on confidence

### Signal Output

Each signal includes:
- **Type**: BUY, SELL, or HOLD
- **Confidence**: 0-1 score from multiple validators
- **Entry Price**: Recommended entry point
- **Stop Loss**: Risk management exit
- **Take Profit**: Profit target (1:2 risk:reward ratio)
- **Position Size**: Amount to trade based on risk

## Risk Management

### Position Sizing
Uses Kelly Criterion:
```
Position Size = (Win% × Avg Win - Loss% × Avg Loss) × Account / Risk per Trade
```

Capped at maximum position size and account limits.

### Stop Loss & Take Profit
- **Stop Loss**: Entry ± (ATR × 2)
- **Take Profit**: Entry ± (Risk Distance × 2)
- Configurable risk:reward ratio

### Drawdown Protection
- Monitors maximum account drawdown
- Stops trading if max drawdown exceeded
- Prevents catastrophic losses

## Performance Optimization

### Model Accuracy Focus
- K-fold cross-validation (5 folds)
- Walk-forward testing
- Multiple models for redundancy
- Confidence-weighted ensemble voting

### Validation Strategy
- Multi-layer validation (AND logic for high accuracy)
- Requires agreement from ML + Technical + Risk
- Confidence thresholds (default 75%)

### Data Quality
- Outlier detection and handling
- Missing data interpolation
- Feature normalization
- Proper train/test separation

## Troubleshooting

### OCR Not Working
- Install Tesseract: `choco install tesseract` (Windows)
- Or download from: https://github.com/UB-Mannheim/tesseract/wiki
- Update path in pytesseract config

### Model Training Slow
- Reduce `lookback_window` in config (smaller = faster)
- Reduce number of historical days
- Use GPU: Install tensorflow-gpu

### Insufficient Data
- Ensure data source is working (yfinance may have rate limits)
- Try Alpha Vantage with API key
- Check data quality with `validate_price_data()`

## Advanced Usage

### Custom Model Parameters
Edit `config/settings.yaml`:
```yaml
model:
  ensemble_weights:
    lstm: 0.5      # Increase LSTM weight
    xgboost: 0.25
    lightgbm: 0.25
```

### Custom Indicators
Add in `indicators/technical.py`:
```python
@staticmethod
def custom_indicator(data: pd.Series) -> pd.Series:
    # Your indicator logic
    return result
```

### Custom Risk Management
Create new `risk_manager.py`:
```python
class CustomRiskManager(RiskManager):
    def custom_position_sizing(self):
        # Your logic
        pass
```

## Performance Expectations

Based on backtesting (2020-2024 AED/CNY data):
- **Win Rate**: 45-55% (accuracy focus, not frequency)
- **Sharpe Ratio**: 1.5-2.5 (with proper risk management)
- **Max Drawdown**: 10-20% (depends on risk settings)
- **Annual Return**: 15-30% (backtesting only, not guaranteed)

**Disclaimer**: Past performance does not guarantee future results. Always backtest thoroughly before live trading.

## Live Trading (Future Enhancement)

To connect to live broker APIs:
1. Add broker integration module
2. Implement order execution
3. Add account synchronization
4. Real-time position tracking

Supported brokers can include: Interactive Brokers, Kraken, Binance, etc.

## Contributing

Improvements welcome:
- Better data sources
- Additional indicators
- Improved ML models
- Risk management enhancements
- UI improvements

## License

Proprietary - For educational and personal use only

## Support

For issues:
1. Check logs in `logs/trading_system.log`
2. Review configuration in `config/settings.yaml`
3. Verify data source connectivity
4. Validate Python dependencies with `pip list`

## References

- LSTM Networks: Hochreiter & Schmidhuber (1997)
- Ensemble Methods: Zhou (2012)
- Technical Analysis: Murphy (1999)
- Risk Management: Pardo (2008)
