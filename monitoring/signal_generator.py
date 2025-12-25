"""Trading signal generation engine."""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
from datetime import datetime
from risk import TradeSignal, calculate_confidence_score
from utils.logger import setup_logger
from config import config

logger = setup_logger(__name__)


class SignalGenerator:
    """Generate trading signals from ML predictions and technical indicators."""
    
    def __init__(self):
        """Initialize signal generator."""
        self.signal_config = config.get('signals', {})
        self.validation_layers = self.signal_config.get('validation_layers', 3)
        self.validation_mode = self.signal_config.get('validation_mode', 'AND')
    
    def generate_ml_signal(self, prediction: float, confidence: float, 
                          last_close: float, threshold: float = 0.01) -> Tuple[TradeSignal, float]:
        """Generate signal from ML model prediction.
        
        Args:
            prediction: Predicted price
            confidence: Model confidence (0-1)
            last_close: Last close price
            threshold: Minimum price change to signal (1% default)
            
        Returns:
            Tuple of (signal, confidence)
        """
        price_change = (prediction - last_close) / last_close
        
        if abs(price_change) < threshold:
            return TradeSignal.HOLD, confidence * 0.5  # Lower confidence for borderline
        elif price_change > threshold:
            return TradeSignal.BUY, confidence
        else:
            return TradeSignal.SELL, confidence
    
    def generate_technical_signal(self, data: pd.Series) -> Tuple[TradeSignal, float]:
        """Generate signal from technical indicators.
        
        Args:
            data: Row of data with indicators
            
        Returns:
            Tuple of (signal, confidence)
        """
        signals = []
        
        # RSI signal
        if 'rsi' in data.index:
            rsi = data['rsi']
            if rsi < 30:
                signals.append((TradeSignal.BUY, 0.7))
            elif rsi > 70:
                signals.append((TradeSignal.SELL, 0.7))
            else:
                signals.append((TradeSignal.HOLD, 0.5))
        
        # MACD signal
        if 'macd' in data.index and 'macd_signal' in data.index:
            if data['macd'] > data['macd_signal']:
                signals.append((TradeSignal.BUY, 0.6))
            elif data['macd'] < data['macd_signal']:
                signals.append((TradeSignal.SELL, 0.6))
            else:
                signals.append((TradeSignal.HOLD, 0.4))
        
        # Moving average signal
        if 'ma_20' in data.index and 'ma_50' in data.index:
            if data['ma_20'] > data['ma_50']:
                signals.append((TradeSignal.BUY, 0.5))
            elif data['ma_20'] < data['ma_50']:
                signals.append((TradeSignal.SELL, 0.5))
            else:
                signals.append((TradeSignal.HOLD, 0.4))
        
        # Bollinger Bands signal
        if 'close' in data.index and 'bb_lower' in data.index and 'bb_upper' in data.index:
            if data['close'] < data['bb_lower']:
                signals.append((TradeSignal.BUY, 0.6))
            elif data['close'] > data['bb_upper']:
                signals.append((TradeSignal.SELL, 0.6))
            else:
                signals.append((TradeSignal.HOLD, 0.4))
        
        if not signals:
            return TradeSignal.HOLD, 0.0
        
        # Combine signals
        signal_counts = {TradeSignal.BUY: 0, TradeSignal.SELL: 0, TradeSignal.HOLD: 0}
        confidence_sum = 0
        
        for sig, conf in signals:
            signal_counts[sig] += conf
            confidence_sum += conf
        
        # Determine majority signal
        max_signal = max(signal_counts.items(), key=lambda x: x[1])
        avg_confidence = confidence_sum / len(signals)
        
        return max_signal[0], avg_confidence
    
    def generate_combined_signal(self, ml_signal: TradeSignal, ml_confidence: float,
                               technical_signal: TradeSignal, tech_confidence: float,
                               risk_valid: bool) -> Tuple[TradeSignal, float]:
        """Combine signals from multiple sources.
        
        Args:
            ml_signal: ML model signal
            ml_confidence: ML confidence
            technical_signal: Technical indicator signal
            tech_confidence: Technical confidence
            risk_valid: Risk validation result
            
        Returns:
            Tuple of (final_signal, confidence)
        """
        risk_score = 1.0 if risk_valid else 0.0
        
        if self.validation_mode == 'AND':
            # All validators must agree
            if ml_signal == technical_signal and ml_signal != TradeSignal.HOLD:
                confidence = calculate_confidence_score(ml_confidence, tech_confidence, risk_score)
                return ml_signal, confidence
            else:
                return TradeSignal.HOLD, 0.0
        
        else:  # WEIGHTED voting
            # Weighted majority voting
            signal_votes = {
                TradeSignal.BUY: 0,
                TradeSignal.SELL: 0,
                TradeSignal.HOLD: 0
            }
            
            signal_votes[ml_signal] += ml_confidence * 0.5
            signal_votes[technical_signal] += tech_confidence * 0.3
            
            if risk_valid:
                # Risk validation favors the stronger signal
                if ml_confidence > tech_confidence:
                    signal_votes[ml_signal] += 0.2
                else:
                    signal_votes[technical_signal] += 0.2
            
            final_signal = max(signal_votes.items(), key=lambda x: x[1])[0]
            confidence = calculate_confidence_score(ml_confidence, tech_confidence, risk_score)
            
            return final_signal, confidence
    
    def generate_signals_batch(self, data: pd.DataFrame, predictions: np.ndarray, 
                              confidences: np.ndarray) -> pd.DataFrame:
        """Generate signals for a batch of data points.
        
        Args:
            data: Historical data with indicators
            predictions: ML predictions
            confidences: ML confidences
            
        Returns:
            DataFrame with signals and metadata
        """
        signals_list = []
        
        for i in range(len(data)):
            row = data.iloc[i]
            pred = predictions[i]
            conf = confidences[i]
            
            # ML signal
            ml_sig, ml_conf = self.generate_ml_signal(pred, conf, row['close'])
            
            # Technical signal
            tech_sig, tech_conf = self.generate_technical_signal(row)
            
            # Risk check (basic validation)
            risk_valid = True  # TODO: Integrate with RiskManager
            
            # Combined signal
            final_sig, final_conf = self.generate_combined_signal(
                ml_sig, ml_conf, tech_sig, tech_conf, risk_valid
            )
            
            signals_list.append({
                'timestamp': row.name,
                'close': row['close'],
                'ml_signal': ml_sig.name,
                'ml_confidence': ml_conf,
                'technical_signal': tech_sig.name,
                'technical_confidence': tech_conf,
                'final_signal': final_sig.name,
                'final_confidence': final_conf,
                'prediction': pred
            })
        
        return pd.DataFrame(signals_list)


def extract_signal_summary(signals_df: pd.DataFrame) -> Dict:
    """Extract summary statistics from signals.
    
    Args:
        signals_df: Signals dataframe
        
    Returns:
        Dictionary with summary stats
    """
    total_signals = len(signals_df)
    buy_signals = (signals_df['final_signal'] == 'BUY').sum()
    sell_signals = (signals_df['final_signal'] == 'SELL').sum()
    hold_signals = (signals_df['final_signal'] == 'HOLD').sum()
    
    avg_confidence = signals_df['final_confidence'].mean()
    max_confidence = signals_df['final_confidence'].max()
    min_confidence = signals_df['final_confidence'].min()
    
    high_confidence_signals = (signals_df['final_confidence'] >= 0.75).sum()
    
    return {
        'total_signals': total_signals,
        'buy_signals': buy_signals,
        'sell_signals': sell_signals,
        'hold_signals': hold_signals,
        'buy_percentage': 100 * buy_signals / total_signals if total_signals > 0 else 0,
        'sell_percentage': 100 * sell_signals / total_signals if total_signals > 0 else 0,
        'hold_percentage': 100 * hold_signals / total_signals if total_signals > 0 else 0,
        'avg_confidence': avg_confidence,
        'max_confidence': max_confidence,
        'min_confidence': min_confidence,
        'high_confidence_signals': high_confidence_signals,
        'high_confidence_percentage': 100 * high_confidence_signals / total_signals if total_signals > 0 else 0,
    }
