"""Backtesting framework for strategy validation."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from utils.logger import setup_logger
from utils import calculate_returns, calculate_sharpe_ratio, calculate_max_drawdown, calculate_win_rate, calculate_profit_factor
from config import config

logger = setup_logger(__name__)


class Backtest:
    """Backtest trading strategy."""
    
    def __init__(self, initial_balance: float = None, transaction_cost: float = None, 
                 slippage: float = None):
        """Initialize backtest.
        
        Args:
            initial_balance: Starting account balance
            transaction_cost: Commission percentage (0.001 = 0.1%)
            slippage: Slippage percentage
        """
        bt_config = config.get('backtesting', {})
        self.initial_balance = initial_balance or bt_config.get('initial_balance', 10000)
        self.transaction_cost = transaction_cost or bt_config.get('transaction_cost', 0.001)
        self.slippage = slippage or bt_config.get('slippage', 0.0002)
        
        self.balance = self.initial_balance
        self.trades = []
        self.equity_curve = [self.initial_balance]
        self.cash = self.initial_balance
        self.positions = {}  # symbol -> {quantity, entry_price}
    
    def execute_trade(self, symbol: str, signal: str, price: float, position_size: float, 
                     stop_loss: float, take_profit: float, confidence: float, 
                     timestamp: datetime = None):
        """Execute a trade.
        
        Args:
            symbol: Trading symbol
            signal: 'BUY' or 'SELL'
            price: Entry price
            position_size: Position size
            stop_loss: Stop loss price
            take_profit: Take profit price
            confidence: Trade confidence
            timestamp: Trade timestamp
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Apply slippage
        slipped_price = price * (1 + self.slippage if signal == 'BUY' else 1 - self.slippage)
        
        # Calculate transaction cost
        transaction_fee = abs(position_size * slipped_price) * self.transaction_cost
        
        if signal == 'BUY':
            # Check if we have cash
            cost = position_size * slipped_price + transaction_fee
            if cost > self.cash:
                logger.warning(f"Insufficient cash for BUY: {cost:.2f} > {self.cash:.2f}")
                return
            
            self.cash -= cost
            self.positions[symbol] = {
                'quantity': position_size,
                'entry_price': slipped_price,
                'entry_time': timestamp,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'confidence': confidence
            }
            
            logger.info(f"BUY {position_size:.6f} @ {slipped_price:.6f}")
            
        elif signal == 'SELL':
            if symbol not in self.positions:
                logger.warning(f"No position to sell for {symbol}")
                return
            
            pos = self.positions[symbol]
            quantity = pos['quantity']
            entry_price = pos['entry_price']
            
            # Calculate P&L
            pnl = quantity * (slipped_price - entry_price) - transaction_fee
            
            self.cash += quantity * slipped_price - transaction_fee
            
            holding_time = (timestamp - pos['entry_time']).days if pos['entry_time'] else 0
            
            self.trades.append({
                'symbol': symbol,
                'entry_price': entry_price,
                'exit_price': slipped_price,
                'quantity': quantity,
                'entry_time': pos['entry_time'],
                'exit_time': timestamp,
                'holding_days': holding_time,
                'pnl': pnl,
                'pnl_pct': pnl / (quantity * entry_price) if quantity * entry_price > 0 else 0,
                'confidence': pos['confidence']
            })
            
            del self.positions[symbol]
            
            logger.info(f"SELL {quantity:.6f} @ {slipped_price:.6f}, P&L: {pnl:.2f}")
    
    def update_equity(self, symbol: str, current_price: float):
        """Update equity based on current prices.
        
        Args:
            symbol: Trading symbol
            current_price: Current price
        """
        total_equity = self.cash
        
        for sym, pos in self.positions.items():
            unrealized_pnl = pos['quantity'] * (current_price - pos['entry_price'])
            total_equity += pos['quantity'] * current_price
        
        self.balance = total_equity
        self.equity_curve.append(total_equity)
    
    def get_performance_metrics(self) -> Dict:
        """Calculate performance metrics.
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'total_pnl_pct': 0,
                'avg_trade_pnl': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0
            }
        
        trades_df = pd.DataFrame(self.trades)
        
        total_pnl = trades_df['pnl'].sum()
        total_pnl_pct = total_pnl / self.initial_balance
        
        winning_trades = (trades_df['pnl'] > 0).sum()
        losing_trades = (trades_df['pnl'] < 0).sum()
        total_trades = len(trades_df)
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        avg_trade_pnl = trades_df['pnl'].mean()
        
        # Sharpe Ratio
        equity_series = pd.Series(self.equity_curve)
        returns = calculate_returns(equity_series)
        sharpe = calculate_sharpe_ratio(returns) if len(returns) > 1 else 0
        
        # Max Drawdown
        max_dd = calculate_max_drawdown(returns) if len(returns) > 1 else 0
        
        # Profit Factor
        pf = calculate_profit_factor(self.trades)
        
        # Avg holding time
        avg_holding_days = trades_df['holding_days'].mean()
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_pnl_pct': total_pnl_pct,
            'avg_trade_pnl': avg_trade_pnl,
            'profit_factor': pf,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'avg_holding_days': avg_holding_days,
            'final_balance': self.balance,
            'return_pct': (self.balance - self.initial_balance) / self.initial_balance
        }


def run_backtest(data: pd.DataFrame, signals: pd.DataFrame, initial_balance: float = 10000) -> Tuple[Backtest, Dict]:
    """Run backtest on historical data with signals.
    
    Args:
        data: Price data with OHLCV
        signals: Signals dataframe
        initial_balance: Initial account balance
        
    Returns:
        Tuple of (Backtest object, performance metrics)
    """
    logger.info(f"Starting backtest with initial balance: {initial_balance}")
    
    bt = Backtest(initial_balance=initial_balance)
    
    # Align data and signals
    data = data.copy()
    data = data.join(signals.set_index('timestamp'), how='inner')
    
    for idx, row in data.iterrows():
        # Check for exits first
        for symbol, pos in list(bt.positions.items()):
            # Stop loss
            if row['low'] <= pos['stop_loss']:
                bt.execute_trade(symbol, 'SELL', pos['stop_loss'], pos['quantity'],
                               0, 0, pos['confidence'], idx)
            # Take profit
            elif row['high'] >= pos['take_profit']:
                bt.execute_trade(symbol, 'SELL', pos['take_profit'], pos['quantity'],
                               0, 0, pos['confidence'], idx)
        
        # Check for entries
        if pd.notna(row.get('final_signal')) and row['final_signal'] in ['BUY', 'SELL']:
            signal = row['final_signal']
            confidence = row.get('final_confidence', 0.5)
            
            if signal == 'BUY':
                # Calculate position size (simple fixed size for now)
                position_size = (bt.cash * 0.05) / row['close']
                stop_loss = row['close'] * 0.98
                take_profit = row['close'] * 1.02
                
                bt.execute_trade('AED/CNY', 'BUY', row['close'], position_size,
                               stop_loss, take_profit, confidence, idx)
        
        # Update equity
        bt.update_equity('AED/CNY', row['close'])
    
    # Close any remaining positions
    if bt.positions:
        last_price = data.iloc[-1]['close']
        for symbol in list(bt.positions.keys()):
            bt.execute_trade(symbol, 'SELL', last_price, bt.positions[symbol]['quantity'],
                           0, 0, bt.positions[symbol]['confidence'], data.index[-1])
    
    metrics = bt.get_performance_metrics()
    logger.info(f"Backtest complete. Final balance: {metrics['final_balance']:.2f}")
    
    return bt, metrics
