"""Main trading system application."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional
import pickle
import os

from utils.logger import setup_logger
from utils import format_timestamp
from utils.gpu_utils import diagnose_gpu_setup, setup_gpu_memory_growth
from config import config

from data import fetch_historical_data, DataPreprocessor, engineer_features
from indicators import calculate_all_indicators
from models import EnsembleModel, evaluate_model
from risk import RiskManager, TradeSignal
from monitoring import SignalGenerator, extract_signal_summary
from backtesting import run_backtest
from ocr import PriceOCR

logger = setup_logger(__name__)


class TradingSystem:
    """Main trading system orchestrator."""
    
    def __init__(self, symbol: str = 'AED=X'):
        """Initialize trading system.
        
        Args:
            symbol: Trading symbol (default AED=X for AED/USD, need to map to AED/CNY)
        """
        self.symbol = symbol
        self.config = config
        
        # Components
        self.preprocessor = DataPreprocessor()
        self.model = EnsembleModel(
            lookback_window=config.get_nested('model.lookback_window', 60),
            forecast_horizon=config.get_nested('model.forecast_horizon', 5)
        )
        self.risk_manager = RiskManager()
        self.signal_generator = SignalGenerator()
        self.price_ocr = PriceOCR()
        
        # Data
        self.historical_data = None
        self.latest_signals = None
        self.trades = []
        self.performance_metrics = {}
        
        logger.info("Trading system initialized")
    
    def prepare_data(self, days_back: int = 1825) -> bool:
        """Prepare historical data for model training.
        
        Args:
            days_back: Number of days of historical data to fetch
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Fetching historical data for {self.symbol} ({days_back} days)...")
        
        try:
            data, is_valid = fetch_historical_data(self.symbol, days_back)
            
            if not is_valid:
                logger.error("Failed to fetch historical data")
                return False
            
            logger.info(f"Preprocessing data ({len(data)} rows)...")
            data, preprocess_info = self.preprocessor.preprocess(data)
            
            logger.info(f"Engineering features...")
            data = self.preprocessor.add_derived_features(data)
            data = engineer_features(data)
            
            logger.info(f"Calculating technical indicators...")
            data = calculate_all_indicators(data)
            
            # Drop NaN rows
            data = data.dropna()
            
            self.historical_data = data
            logger.info(f"Data preparation complete. Final shape: {data.shape}")
            return True
            
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            return False
    
    def train_model(self, epochs: int = 50) -> bool:
        """Train ML ensemble model.
        
        Args:
            epochs: Number of training epochs
            
        Returns:
            True if successful, False otherwise
        """
        if self.historical_data is None:
            logger.error("No data available for training")
            return False
        
        try:
            logger.info("Preparing training data...")
            
            # Prepare features and target
            features = self.model.prepare_features(self.historical_data)
            target = self.historical_data['close'].values
            
            # Create sequences
            X_seq, y_seq = self.model.create_sequences(features, target)
            
            if len(X_seq) == 0:
                logger.error("No sequences created from data")
                return False
            
            logger.info(f"Training data shape: {X_seq.shape}")
            logger.info("Training ensemble model...")
            
            self.model.train(X_seq, y_seq, epochs=epochs)
            
            # Evaluate on training data
            predictions, confidences = self.model.predict(X_seq)
            metrics = evaluate_model(y_seq[:, 0], predictions)
            
            logger.info(f"Training complete. Metrics: {metrics}")
            self.performance_metrics['training'] = metrics
            
            return True
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return False
    
    def generate_signals(self) -> Optional[pd.DataFrame]:
        """Generate trading signals for latest data.
        
        Returns:
            DataFrame with signals, or None if failed
        """
        if self.historical_data is None:
            logger.error("No data available for signal generation")
            return None
        
        try:
            logger.info("Generating trading signals...")
            
            # Prepare features
            features = self.model.prepare_features(self.historical_data)
            
            # Create sequences
            X_seq, _ = self.model.create_sequences(features, self.historical_data['close'].values)
            
            # Make predictions
            predictions, confidences = self.model.predict(X_seq)
            
            # Generate signals
            signals = self.signal_generator.generate_signals_batch(
                self.historical_data.iloc[-len(X_seq):],
                predictions,
                confidences
            )
            
            self.latest_signals = signals
            
            # Extract summary
            summary = extract_signal_summary(signals)
            logger.info(f"Signal generation complete. Summary: {summary}")
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return None
    
    def backtest_strategy(self) -> Optional[Dict]:
        """Run backtest on historical data.
        
        Returns:
            Dictionary with backtest metrics, or None if failed
        """
        if self.historical_data is None or self.latest_signals is None:
            logger.error("Missing data for backtesting")
            return None
        
        try:
            logger.info("Running backtest...")
            
            initial_balance = config.get_nested('backtesting.initial_balance', 10000)
            bt, metrics = run_backtest(self.historical_data, self.latest_signals, initial_balance)
            
            logger.info(f"Backtest results: {metrics}")
            self.performance_metrics['backtest'] = metrics
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error in backtest: {e}")
            return None
    
    def get_latest_signal(self) -> Optional[Dict]:
        """Get the latest trading signal.
        
        Returns:
            Dictionary with latest signal information
        """
        if self.latest_signals is None or len(self.latest_signals) == 0:
            return None
        
        latest = self.latest_signals.iloc[-1]
        
        return {
            'timestamp': latest['timestamp'],
            'close': latest['close'],
            'signal': latest['final_signal'],
            'confidence': latest['final_confidence'],
            'prediction': latest['prediction']
        }
    
    def save_model(self, filepath: str = 'models/ensemble_model.pkl'):
        """Save trained model to file.
        
        Args:
            filepath: Path to save model
        """
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'wb') as f:
                pickle.dump(self.model, f)
            logger.info(f"Model saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def load_model(self, filepath: str = 'models/ensemble_model.pkl'):
        """Load trained model from file.
        
        Args:
            filepath: Path to load model
        """
        try:
            with open(filepath, 'rb') as f:
                self.model = pickle.load(f)
            logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
    
    def get_system_status(self) -> Dict:
        """Get current system status.
        
        Returns:
            Dictionary with system information
        """
        status = {
            'timestamp': format_timestamp(),
            'symbol': self.symbol,
            'data_loaded': self.historical_data is not None,
            'model_trained': self.model.models['lstm'] is not None,
            'signals_generated': self.latest_signals is not None,
            'latest_signal': self.get_latest_signal(),
            'performance_metrics': self.performance_metrics,
            'open_positions': len(self.risk_manager.open_positions),
            'closed_trades': len(self.risk_manager.closed_trades),
            'account_balance': self.risk_manager.current_balance
        }
        
        return status


def main():
    """Main entry point for trading system."""
    logger.info("Starting AED/CNY Trading System...")
    
    # Run GPU diagnostics
    logger.info("\nInitializing GPU acceleration...")
    gpu_status = diagnose_gpu_setup()
    setup_gpu_memory_growth()
    
    try:
        # Initialize system
        system = TradingSystem(symbol='AED=X')  # Using AED/USD as proxy for AED/CNY
        
        # Step 1: Prepare data
        if not system.prepare_data(days_back=1825):
            logger.error("Failed to prepare data")
            return
        
        # Step 2: Train model
        if not system.train_model(epochs=50):
            logger.error("Failed to train model")
            return
        
        # Step 3: Generate signals
        signals = system.generate_signals()
        if signals is None:
            logger.error("Failed to generate signals")
            return
        
        # Step 4: Backtest strategy
        backtest_results = system.backtest_strategy()
        if backtest_results is None:
            logger.error("Failed to run backtest")
            return
        
        # Step 5: Save model
        system.save_model()
        
        # Print system status
        status = system.get_system_status()
        logger.info(f"System Status: {status}")
        
        print("\n" + "="*80)
        print("TRADING SYSTEM INITIALIZED SUCCESSFULLY")
        print("="*80)
        print(f"Symbol: {status['symbol']}")
        print(f"Latest Signal: {status['latest_signal']}")
        print(f"Account Balance: ${status['account_balance']:.2f}")
        print(f"Backtest P&L: ${backtest_results['total_pnl']:.2f}")
        print(f"Win Rate: {backtest_results['win_rate']:.2%}")
        print(f"Sharpe Ratio: {backtest_results['sharpe_ratio']:.2f}")
        print("="*80 + "\n")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise


if __name__ == '__main__':
    main()
