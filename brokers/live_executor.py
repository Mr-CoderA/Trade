"""Live Trading Executor - Converts AI signals to broker trades"""

import logging
from typing import Tuple, Optional
from datetime import datetime
from brokers.pocket_option import PocketOptionBroker, TradeType, AccountType
from utils.logger import setup_logger

logger = setup_logger(__name__)


class LiveTradingExecutor:
    """
    Orchestrates live trading execution
    - Converts AI signals to broker trades
    - Manages position sizing and risk
    - Monitors active trades
    - Implements safety mechanisms
    """
    
    def __init__(
        self,
        broker: PocketOptionBroker,
        account_balance: float,
        risk_per_trade: float = 0.01,
        max_position_size: float = 0.05,
        max_daily_trades: int = 20,
        demo_mode: bool = True
    ):
        """
        Initialize live trading executor
        
        Args:
            broker: PocketOptionBroker instance
            account_balance: Starting account balance
            risk_per_trade: Risk per trade (0.01 = 1%)
            max_position_size: Max position size (0.05 = 5%)
            max_daily_trades: Max trades per day
            demo_mode: If True, only logs trades without executing
        """
        self.broker = broker
        self.account_balance = account_balance
        self.risk_per_trade = risk_per_trade
        self.max_position_size = max_position_size
        self.max_daily_trades = max_daily_trades
        self.demo_mode = demo_mode
        
        # Trading state
        self.trades_executed = []
        self.last_signal = None
        self.active_positions = {}
        
        logger.info(
            f"LiveTradingExecutor initialized\n"
            f"├─ Account Balance: ${account_balance:.2f}\n"
            f"├─ Risk Per Trade: {risk_per_trade*100:.1f}%\n"
            f"├─ Max Position Size: {max_position_size*100:.1f}%\n"
            f"├─ Max Daily Trades: {max_daily_trades}\n"
            f"└─ Mode: {'DEMO (simulation)' if demo_mode else 'LIVE (real money)'}"
        )
    
    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss_price: float,
        current_balance: float
    ) -> float:
        """
        Calculate position size based on risk management
        
        Args:
            entry_price: Entry price
            stop_loss_price: Stop loss price
            current_balance: Current account balance
            
        Returns:
            Position size in USD
        """
        # Calculate price distance to stop loss
        price_distance = abs(entry_price - stop_loss_price)
        
        if price_distance <= 0:
            logger.warning("Invalid stop loss - price distance is 0")
            return 0
        
        # Calculate max position size
        max_position = current_balance * self.max_position_size
        
        # Calculate position based on risk
        risk_amount = current_balance * self.risk_per_trade
        position_size = risk_amount / price_distance
        
        # Use whichever is smaller (risk or max position)
        position_size = min(position_size, max_position)
        
        # Round to nearest dollar
        position_size = round(position_size)
        
        logger.debug(
            f"Position Size Calculation:\n"
            f"├─ Entry: ${entry_price:.4f}\n"
            f"├─ Stop Loss: ${stop_loss_price:.4f}\n"
            f"├─ Distance: ${price_distance:.4f}\n"
            f"├─ Risk Amount: ${risk_amount:.2f}\n"
            f"├─ Max Position: ${max_position:.2f}\n"
            f"└─ Recommended Size: ${position_size:.2f}"
        )
        
        return position_size
    
    def execute_signal(
        self,
        signal: str,  # "BUY", "SELL", "HOLD"
        confidence: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        asset: str = "AED/CNY",
        expiration_minutes: int = 5,
        current_balance: float = None
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Execute a trading signal
        
        Args:
            signal: "BUY", "SELL", or "HOLD"
            confidence: Signal confidence (0-1)
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
            asset: Asset symbol
            expiration_minutes: Expiration time in minutes
            current_balance: Current account balance
            
        Returns:
            (success, message, trade_id)
        """
        current_balance = current_balance or self.account_balance
        
        # Validate signal
        if signal not in ["BUY", "SELL", "HOLD"]:
            return False, f"Invalid signal: {signal}", None
        
        if signal == "HOLD":
            logger.info("Signal is HOLD - no trade executed")
            return True, "HOLD signal - no trade", None
        
        if confidence < 0.75:
            return False, f"Confidence too low: {confidence:.2f} (min 0.75)", None
        
        # Calculate position size
        position_size = self.calculate_position_size(
            entry_price,
            stop_loss,
            current_balance
        )
        
        if position_size < 1:
            return False, f"Position size too small: ${position_size:.2f}", None
        
        # Determine trade direction
        trade_direction = TradeType.BUY if signal == "BUY" else TradeType.SELL
        
        # Execute trade
        if self.demo_mode:
            # Demo mode: simulate trade
            success, message = self._simulate_trade(
                signal, asset, position_size, entry_price, stop_loss, take_profit
            )
            return success, message, None
        else:
            # Live mode: execute on broker
            success, message, trade_id = self.broker.execute_trade(
                asset=asset,
                direction=trade_direction,
                amount=position_size,
                expiration_minutes=expiration_minutes,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            if success:
                self.active_positions[trade_id] = {
                    "asset": asset,
                    "direction": signal,
                    "entry_price": entry_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "position_size": position_size,
                    "confidence": confidence,
                    "execution_time": datetime.now()
                }
            
            return success, message, trade_id
    
    def _simulate_trade(
        self,
        signal: str,
        asset: str,
        position_size: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float
    ) -> Tuple[bool, str]:
        """Simulate a trade (demo mode)"""
        trade_log = (
            f"\n{'='*70}\n"
            f"DEMO TRADE SIMULATION\n"
            f"{'='*70}\n"
            f"Time:           {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Asset:          {asset}\n"
            f"Signal:         {signal}\n"
            f"Entry Price:    ${entry_price:.4f}\n"
            f"Position Size:  ${position_size:.2f}\n"
            f"Stop Loss:      ${stop_loss:.4f} (Risk: ${abs(entry_price - stop_loss) * position_size:.2f})\n"
            f"Take Profit:    ${take_profit:.4f} (Reward: ${abs(take_profit - entry_price) * position_size:.2f})\n"
            f"Risk/Reward:    1:{abs(take_profit - entry_price) / abs(entry_price - stop_loss):.2f}\n"
            f"{'='*70}\n"
        )
        
        logger.info(trade_log)
        
        self.trades_executed.append({
            "signal": signal,
            "asset": asset,
            "entry_price": entry_price,
            "position_size": position_size,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "time": datetime.now()
        })
        
        return True, f"[DEMO MODE] {signal} trade simulated for ${position_size:.2f}"
    
    def close_position(self, trade_id: str) -> Tuple[bool, str]:
        """Close an active position"""
        if not self.demo_mode:
            return self.broker.close_trade(trade_id)
        else:
            if trade_id in self.active_positions:
                del self.active_positions[trade_id]
                return True, f"[DEMO] Position {trade_id} closed"
            return False, f"Position {trade_id} not found"
    
    def get_active_positions(self) -> dict:
        """Get all active positions"""
        return self.active_positions.copy()
    
    def get_trade_summary(self) -> dict:
        """Get trading summary"""
        return {
            "total_trades": len(self.trades_executed),
            "active_positions": len(self.active_positions),
            "current_balance": self.account_balance,
            "mode": "DEMO" if self.demo_mode else "LIVE",
            "last_signal": self.last_signal
        }
    
    def set_demo_mode(self, enabled: bool):
        """Toggle demo mode (safety switch)"""
        self.demo_mode = enabled
        mode_str = "ON (simulation)" if enabled else "OFF (live trading)"
        logger.warning(f"Demo mode set to: {mode_str}")
    
    def __repr__(self):
        return (
            f"LiveTradingExecutor("
            f"Mode: {'DEMO' if self.demo_mode else 'LIVE'}, "
            f"Active Positions: {len(self.active_positions)}, "
            f"Trades Executed: {len(self.trades_executed)})"
        )
