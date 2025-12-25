"""
Unified Main Entry Point - AED/CNY Trading System
Supports both backtesting and live trading on Pocket Option
"""

import sys
import argparse
import pandas as pd
import numpy as np
import pickle
import time
from datetime import datetime, timedelta
from typing import Dict, Optional
from pathlib import Path

from utils.logger import setup_logger
from utils import format_timestamp
from utils.gpu_utils import diagnose_gpu_setup, setup_gpu_memory_growth
from config import config
from data.fetcher import YFinanceFetcher
from data.preprocessor import DataPreprocessor
from indicators.technical import TechnicalIndicators
from models.ml_models import EnsembleModel
from risk.risk_manager import RiskManager
from monitoring.signal_generator import SignalGenerator
from backtesting.backtest import Backtest
from config.live_trading_config import LiveTradingConfig
from brokers.pocket_option import PocketOptionBroker, AccountType
from brokers.live_executor import LiveTradingExecutor

logger = setup_logger(__name__)


class TradingSystem:
    """Unified trading system orchestrator - supports both backtest and live trading."""
    
    def __init__(self, live_trading: bool = False):
        """Initialize trading system.
        
        Args:
            live_trading: If True, execute real trades. If False, backtest only.
        """
        self.live_trading = live_trading
        self.symbol = 'AED=X'
        
        # Data & Models
        self.data = None
        self.model = None
        self.preprocessor = None
        
        # Risk & Signals
        self.risk_manager = None
        self.signal_generator = None
        
        # Live Trading (only if enabled)
        self.broker = None
        self.live_executor = None
        self.live_config = None
        
        # Tracking
        self.latest_signals = None
        self.performance_metrics = {}
        
        logger.info(f"Trading System initialized (Mode: {'LIVE' if live_trading else 'BACKTEST'})")
    
    # ==================== GPU INITIALIZATION ====================
    def initialize_gpu(self) -> bool:
        """Initialize GPU acceleration.
        
        Returns:
            True if GPU setup successful
        """
        logger.info("Initializing GPU acceleration...")
        gpu_status = diagnose_gpu_setup()
        setup_gpu_memory_growth()
        
        if gpu_status:
            logger.info("✓ GPU acceleration ENABLED")
        else:
            logger.warning("⚠ GPU not detected - will use CPU")
        
        return True
    
    # ==================== DATA PIPELINE ====================
    def fetch_data(self) -> bool:
        """Fetch historical data.
        
        Returns:
            True if successful
        """
        logger.info("Fetching historical data...")
        
        try:
            fetcher = YFinanceFetcher()
            lookback = config.get_nested('data.lookback_period', 5)
            
            self.data = fetcher.fetch(
                self.symbol,
                years=lookback
            )
            
            if self.data is None or len(self.data) < 100:
                logger.error("Failed to fetch sufficient data")
                return False
            
            logger.info(f"✓ Fetched {len(self.data)} candles")
            return True
            
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            return False
    
    def preprocess_data(self) -> bool:
        """Preprocess data.
        
        Returns:
            True if successful
        """
        logger.info("Preprocessing data...")
        
        try:
            self.preprocessor = DataPreprocessor()
            self.data = self.preprocessor.preprocess(self.data)
            
            if self.data is None or len(self.data) == 0:
                logger.error("Data preprocessing resulted in empty dataset")
                return False
            
            logger.info(f"✓ Data preprocessed ({len(self.data)} records)")
            return True
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            return False
    
    def calculate_indicators(self) -> bool:
        """Calculate technical indicators.
        
        Returns:
            True if successful
        """
        logger.info("Calculating technical indicators...")
        
        try:
            ti = TechnicalIndicators(self.data)
            ti.calculate_all_indicators()
            self.data = ti.data
            
            logger.info("✓ Indicators calculated")
            return True
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return False
    
    # ==================== MODEL TRAINING ====================
    def train_model(self) -> bool:
        """Train ML ensemble model.
        
        Returns:
            True if successful
        """
        logger.info("Training ensemble model (LSTM + XGBoost + LightGBM)...")
        
        try:
            self.model = EnsembleModel(
                lookback_window=config.get_nested('model.lookback_window', 60),
                forecast_horizon=config.get_nested('model.forecast_horizon', 5)
            )
            
            # Prepare features
            feature_cols = [col for col in self.data.columns 
                          if col not in ['open', 'high', 'low', 'close', 'volume']]
            X = self.data[feature_cols].values
            y = self.data['close'].values
            
            # Create sequences
            X_seq, y_seq = self.model.create_sequences(X, y)
            
            if len(X_seq) == 0:
                logger.error("No sequences created from data")
                return False
            
            # Train model
            epochs = config.get_nested('model.epochs', 150)
            self.model.train(X_seq, y_seq, epochs=epochs)
            
            logger.info("✓ Model training complete")
            return True
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return False
    
    # ==================== RISK & SIGNAL SETUP ====================
    def setup_risk_management(self) -> bool:
        """Setup risk management.
        
        Returns:
            True if successful
        """
        logger.info("Setting up risk management...")
        
        try:
            self.risk_manager = RiskManager(
                account_balance=config.get_nested('risk.account_balance', 10000),
                risk_per_trade=config.get_nested('risk.risk_per_trade', 0.01),
                max_position_size=config.get_nested('risk.max_position_size', 0.05)
            )
            
            logger.info("✓ Risk management configured")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up risk management: {e}")
            return False
    
    def setup_signal_generator(self) -> bool:
        """Setup signal generator.
        
        Returns:
            True if successful
        """
        logger.info("Setting up signal generator...")
        
        try:
            self.signal_generator = SignalGenerator(
                confidence_threshold=config.get_nested('model.confidence_threshold', 0.75),
                validation_mode="AND"
            )
            
            logger.info("✓ Signal generator configured")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up signal generator: {e}")
            return False
    
    # ==================== BACKTESTING ====================
    def run_backtest(self) -> bool:
        """Run backtesting.
        
        Returns:
            True if successful
        """
        logger.info("Running backtest...")
        
        try:
            # Generate signals for all data
            signals = []
            for i in range(len(self.data) - 1):
                current = self.data.iloc[i]
                next_price = self.data.iloc[i+1]['close']
                
                # Generate signal
                signal, confidence, _ = self.signal_generator.generate_signal(
                    ml_prediction=0.5,
                    technical_signal="BUY" if next_price > current['close'] else "SELL",
                    risk_approved=True
                )
                
                signals.append(signal)
            
            logger.info(f"✓ Backtest complete - Generated {len(signals)} signals")
            return True
            
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            return False
    
    # ==================== LIVE TRADING SETUP ====================
    def setup_live_trading(self) -> bool:
        """Setup live trading connection.
        
        Returns:
            True if successful
        """
        if not self.live_trading:
            return True
        
        logger.info("Setting up live trading...")
        
        try:
            # Load or setup configuration
            self.live_config = LiveTradingConfig()
            
            # Try to load existing config
            if not self.live_config.load():
                # Interactive setup if no config exists
                if not self.live_config.setup_account_interactive():
                    logger.error("Live trading setup cancelled")
                    return False
            
            # Load credentials
            if not self.live_config.load_credentials():
                logger.error("Failed to load credentials")
                return False
            
            # Connect to broker
            self.broker = PocketOptionBroker(
                email=self.live_config.broker_email,
                password=self.live_config.broker_password,
                account_type=self.live_config.account_type
            )
            
            if not self.broker.connect():
                logger.error("Failed to connect to Pocket Option")
                return False
            
            # Setup live executor
            self.live_executor = LiveTradingExecutor(
                broker=self.broker,
                account_balance=self.live_config.initial_balance,
                risk_per_trade=self.live_config.risk_per_trade,
                max_position_size=self.live_config.max_position_size,
                max_daily_trades=self.live_config.max_daily_trades,
                demo_mode=self.live_config.demo_mode
            )
            
            logger.info(f"✓ Live trading connected ({self.live_config.account_type.value})")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up live trading: {e}")
            return False
    
    # ==================== SYSTEM INITIALIZATION ====================
    def run(self) -> bool:
        """Run complete system pipeline.
        
        Returns:
            True if successful
        """
        try:
            logger.info("=" * 80)
            logger.info("TRADING SYSTEM STARTUP")
            logger.info("=" * 80)
            
            # GPU setup
            if not self.initialize_gpu():
                return False
            
            # Data pipeline
            if not self.fetch_data():
                return False
            
            if not self.preprocess_data():
                return False
            
            if not self.calculate_indicators():
                return False
            
            # Model training
            if not self.train_model():
                return False
            
            # Risk and signals
            if not self.setup_risk_management():
                return False
            
            if not self.setup_signal_generator():
                return False
            
            # Backtesting
            if not self.run_backtest():
                return False
            
            # Live trading setup (if enabled)
            if self.live_trading:
                if not self.setup_live_trading():
                    logger.error("Live trading setup failed - falling back to demo mode")
                    self.live_trading = False
            
            logger.info("=" * 80)
            logger.info("TRADING SYSTEM INITIALIZED SUCCESSFULLY")
            logger.info("=" * 80)
            
            if self.live_trading:
                logger.warning("⚠️  LIVE TRADING MODE - Real money at risk!")
                logger.info(f"Account Type: {self.live_config.account_type.value}")
                logger.info(f"Balance: ${self.live_config.initial_balance}")
            else:
                logger.info("Running in DEMO/BACKTEST mode (no real money)")
            
            return True
            
        except Exception as e:
            logger.error(f"System initialization failed: {str(e)}")
            return False
    
    # ==================== UTILITY METHODS ====================
    def get_latest_signal(self) -> Optional[Dict]:
        """Get latest trading signal.
        
        Returns:
            Dictionary with signal info or None
        """
        if self.latest_signals is None or len(self.latest_signals) == 0:
            return None
        
        latest = self.latest_signals.iloc[-1]
        
        return {
            'timestamp': latest['timestamp'],
            'close': latest['close'],
            'signal': latest['final_signal'],
            'confidence': latest['final_confidence']
        }
    
    def get_system_status(self) -> Dict:
        """Get current system status.
        
        Returns:
            Dictionary with system information
        """
        status = {
            'timestamp': format_timestamp(),
            'symbol': self.symbol,
            'mode': 'LIVE' if self.live_trading else 'BACKTEST',
            'data_loaded': self.data is not None,
            'model_trained': self.model is not None,
            'signals_generated': self.latest_signals is not None,
            'latest_signal': self.get_latest_signal(),
            'performance_metrics': self.performance_metrics
        }
        
        if self.live_trading and self.broker:
            status['broker_connected'] = True
            status['account_type'] = self.live_config.account_type.value if self.live_config else 'Unknown'
        
        return status
    
    def save_model(self, filepath: str = 'models/ensemble_model.pkl') -> bool:
        """Save trained model.
        
        Args:
            filepath: Path to save model
            
        Returns:
            True if successful
        """
        try:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'wb') as f:
                pickle.dump(self.model, f)
            logger.info(f"Model saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def load_model(self, filepath: str = 'models/ensemble_model.pkl') -> bool:
        """Load trained model.
        
        Args:
            filepath: Path to load model
            
        Returns:
            True if successful
        """
        try:
            with open(filepath, 'rb') as f:
                self.model = pickle.load(f)
            logger.info(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False


def main():
    """Main entry point for unified trading system."""
    parser = argparse.ArgumentParser(
        description="AED/CNY Trading System with Live Trading Support"
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Enable live trading on Pocket Option"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Demo/backtest mode only (default)"
    )
    
    args = parser.parse_args()
    
    # Determine mode
    live_mode = args.live and not args.demo
    
    # Initialize and run system
    system = TradingSystem(live_trading=live_mode)
    
    if system.run():
        logger.info("✓ System ready for trading")
        
        # Save trained model
        system.save_model()
        
        if live_mode:
            logger.warning("\n" + "="*80)
            logger.warning("⚠️  LIVE TRADING MODE - IMPORTANT SAFETY REMINDERS")
            logger.warning("="*80)
            logger.warning("1. Start with DEMO account only")
            logger.warning("2. Test thoroughly before using real money")
            logger.warning("3. Monitor trades actively - don't leave unattended")
            logger.warning("4. Never trade more than you can afford to lose")
            logger.warning("5. Check daily loss limits are enabled")
            logger.warning("6. Press Ctrl+C to stop and disconnect safely")
            logger.warning("="*80 + "\n")
            
            # Keep system running
            try:
                logger.info("System running. Press Ctrl+C to stop...")
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("\nShutting down...")
                if system.broker:
                    try:
                        system.broker.disconnect()
                        logger.info("Broker disconnected")
                    except:
                        pass
        else:
            # Backtest mode - print summary
            status = system.get_system_status()
            print("\n" + "="*80)
            print("TRADING SYSTEM SUMMARY")
            print("="*80)
            print(f"Mode: {status['mode']}")
            print(f"Symbol: {status['symbol']}")
            print(f"Data Loaded: {status['data_loaded']}")
            print(f"Model Trained: {status['model_trained']}")
            print(f"Signals Generated: {status['signals_generated']}")
            print("="*80 + "\n")
            logger.info("Backtest complete. System ready for next run.")
    else:
        logger.error("✗ System initialization failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
