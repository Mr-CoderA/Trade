"""Risk management module for position sizing and trade validation."""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from enum import Enum
from utils.logger import setup_logger
from config import config

logger = setup_logger(__name__)


class TradeSignal(Enum):
    """Trading signal types."""
    BUY = 1
    SELL = -1
    HOLD = 0


class RiskManager:
    """Manage risk for trading system."""
    
    def __init__(self, account_balance: float = None):
        """Initialize risk manager.
        
        Args:
            account_balance: Initial account balance in USD
        """
        self.account_balance = account_balance or config.get_nested('risk.account_balance', 10000)
        self.current_balance = self.account_balance
        self.risk_config = config.get('risk', {})
        self.open_positions = []
        self.closed_trades = []
    
    def calculate_position_size(self, entry_price: float, stop_loss_price: float, 
                                confidence: float = 1.0) -> float:
        """Calculate optimal position size based on Kelly Criterion.
        
        Args:
            entry_price: Entry price
            stop_loss_price: Stop loss price
            confidence: Model confidence (0-1)
            
        Returns:
            Position size in base currency
        """
        risk_per_trade = self.risk_config.get('risk_per_trade', 0.02)
        max_position_size = self.risk_config.get('max_position_size', 0.10)
        
        # Risk amount
        risk_amount = self.current_balance * risk_per_trade * confidence
        
        # Price distance to stop loss
        price_distance = abs(entry_price - stop_loss_price)
        
        if price_distance == 0:
            logger.warning("Entry price equals stop loss price, position size = 0")
            return 0
        
        # Position size in base currency
        position_size = risk_amount / price_distance
        
        # Apply max position size limit
        max_position = self.current_balance * max_position_size
        position_size = min(position_size, max_position)
        
        logger.info(f"Calculated position size: {position_size:.6f} (confidence: {confidence:.2%})")
        return position_size
    
    def calculate_stop_loss(self, entry_price: float, atr: float = None, 
                           direction: str = 'long') -> float:
        """Calculate stop loss price using ATR.
        
        Args:
            entry_price: Entry price
            atr: Average True Range value
            direction: 'long' or 'short'
            
        Returns:
            Stop loss price
        """
        if atr is None:
            atr = 0.02 * entry_price  # Default to 2% if no ATR
        
        multiplier = self.risk_config.get('stop_loss_atr_multiplier', 2.0)
        stop_loss_distance = atr * multiplier
        
        if direction == 'long':
            stop_loss = entry_price - stop_loss_distance
        else:  # short
            stop_loss = entry_price + stop_loss_distance
        
        return stop_loss
    
    def calculate_take_profit(self, entry_price: float, stop_loss_price: float, 
                             direction: str = 'long') -> float:
        """Calculate take profit price using risk:reward ratio.
        
        Args:
            entry_price: Entry price
            stop_loss_price: Stop loss price
            direction: 'long' or 'short'
            
        Returns:
            Take profit price
        """
        risk_reward_ratio = self.risk_config.get('take_profit_ratio', 2.0)
        risk_distance = abs(entry_price - stop_loss_price)
        reward_distance = risk_distance * risk_reward_ratio
        
        if direction == 'long':
            take_profit = entry_price + reward_distance
        else:  # short
            take_profit = entry_price - reward_distance
        
        return take_profit
    
    def check_max_drawdown(self, current_equity: float) -> bool:
        """Check if max drawdown limit is exceeded.
        
        Args:
            current_equity: Current account equity
            
        Returns:
            True if within limits, False if exceeded
        """
        max_drawdown_limit = self.risk_config.get('max_drawdown_limit', 0.20)
        current_drawdown = (self.account_balance - current_equity) / self.account_balance
        
        if current_drawdown > max_drawdown_limit:
            logger.warning(f"Max drawdown exceeded: {current_drawdown:.2%} > {max_drawdown_limit:.2%}")
            return False
        
        return True
    
    def check_position_limit(self) -> bool:
        """Check if max open positions limit is reached.
        
        Returns:
            True if can open more positions, False if limit reached
        """
        max_positions = self.risk_config.get('max_open_positions', 3)
        
        if len(self.open_positions) >= max_positions:
            logger.warning(f"Max open positions reached: {len(self.open_positions)}")
            return False
        
        return True
    
    def validate_trade(self, signal: TradeSignal, confidence: float, 
                      entry_price: float, atr: float = None) -> Tuple[bool, Dict]:
        """Validate trade based on risk criteria.
        
        Args:
            signal: Trading signal (BUY/SELL/HOLD)
            confidence: Model confidence (0-1)
            entry_price: Entry price
            atr: Average True Range
            
        Returns:
            Tuple of (is_valid, trade_info)
        """
        trade_info = {
            'signal': signal.name,
            'confidence': confidence,
            'entry_price': entry_price,
            'valid': False,
            'reason': ''
        }
        
        # Check if HOLD signal
        if signal == TradeSignal.HOLD:
            trade_info['reason'] = 'HOLD signal'
            return False, trade_info
        
        # Check confidence threshold
        min_confidence = config.get_nested('signals.required_agreement', 0.75)
        if confidence < min_confidence:
            trade_info['reason'] = f'Confidence {confidence:.2%} < minimum {min_confidence:.2%}'
            return False, trade_info
        
        # Check position limit
        if not self.check_position_limit():
            trade_info['reason'] = 'Max open positions reached'
            return False, trade_info
        
        # Calculate stop loss and take profit
        stop_loss = self.calculate_stop_loss(entry_price, atr, 
                                            'long' if signal == TradeSignal.BUY else 'short')
        take_profit = self.calculate_take_profit(entry_price, stop_loss,
                                                 'long' if signal == TradeSignal.BUY else 'short')
        
        # Calculate position size
        position_size = self.calculate_position_size(entry_price, stop_loss, confidence)
        
        if position_size == 0:
            trade_info['reason'] = 'Position size calculated as zero'
            return False, trade_info
        
        trade_info['valid'] = True
        trade_info['stop_loss'] = stop_loss
        trade_info['take_profit'] = take_profit
        trade_info['position_size'] = position_size
        
        logger.info(f"Trade validated: {signal.name} {position_size:.6f} @ {entry_price:.6f}")
        return True, trade_info


def calculate_position_accuracy(predicted_direction: int, actual_direction: int) -> float:
    """Calculate accuracy of directional prediction.
    
    Args:
        predicted_direction: 1 (up), -1 (down), 0 (no change)
        actual_direction: 1 (up), -1 (down), 0 (no change)
        
    Returns:
        Accuracy score (1.0 = correct, 0.0 = wrong)
    """
    if predicted_direction == actual_direction:
        return 1.0
    else:
        return 0.0


def calculate_confidence_score(ml_confidence: float, technical_agreement: float, 
                               risk_check: float) -> float:
    """Calculate overall confidence from multiple validators.
    
    Args:
        ml_confidence: ML model confidence (0-1)
        technical_agreement: Technical indicators agreement (0-1)
        risk_check: Risk validation result (0-1)
        
    Returns:
        Overall confidence score (0-1)
    """
    # Weighted average of validators
    weights = {
        'ml': 0.5,
        'technical': 0.3,
        'risk': 0.2
    }
    
    confidence = (ml_confidence * weights['ml'] + 
                 technical_agreement * weights['technical'] + 
                 risk_check * weights['risk'])
    
    return min(max(confidence, 0.0), 1.0)  # Clamp between 0 and 1
