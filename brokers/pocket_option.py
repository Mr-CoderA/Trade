"""Pocket Option Broker Integration Module - Live Trading Support"""

import requests
import json
import time
import logging
from typing import Dict, Tuple, Optional
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class AccountType(Enum):
    """Account type enumeration"""
    DEMO = "demo"
    REAL = "real"


class TradeType(Enum):
    """Trade type enumeration"""
    BUY = "call"  # Pocket Option uses "call" for up
    SELL = "put"  # Pocket Option uses "put" for down


class PocketOptionBroker:
    """
    Pocket Option Broker Integration
    Handles authentication, trade execution, and position management
    
    IMPORTANT: This module trades REAL MONEY when real_account=True
    Always test with demo account first!
    """
    
    def __init__(self, email: str, password: str, account_type: AccountType = AccountType.DEMO):
        """
        Initialize Pocket Option connection
        
        Args:
            email: Pocket Option account email
            password: Pocket Option account password
            account_type: AccountType.DEMO or AccountType.REAL
            
        WARNING: Keep credentials secure! Never commit to git!
        """
        self.email = email
        self.password = password
        self.account_type = account_type
        self.ssid = None
        self.user_id = None
        self.balance = 0
        self.currency = "USD"
        self.is_connected = False
        
        # API settings
        self.api_url = "https://api.pocketoption.com"
        self.websocket_url = "wss://ws.pocketoption.com/echo"
        
        # Trade tracking
        self.active_trades = {}  # {trade_id: trade_info}
        self.trade_history = []
        self.max_trades_per_day = 20
        self.trades_today = 0
        self.last_trade_time = None
        
        # Safety limits
        self.max_loss_per_day = 100  # USD
        self.daily_loss = 0
        self.equity_stop_out = 0.5  # Stop if account balance drops 50%
        self.initial_balance = 0
        
        logger.info(f"Pocket Option Broker initialized (Account: {account_type.value})")
    
    def connect(self) -> bool:
        """
        Authenticate and connect to Pocket Option
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Connecting to Pocket Option ({self.account_type.value} account)...")
            
            # Login API call
            login_url = f"{self.api_url}/api/v1/login"
            payload = {
                "email": self.email,
                "password": self.password,
                "application_id": 47
            }
            
            response = requests.post(login_url, json=payload, timeout=10)
            data = response.json()
            
            if data.get("result") != "ok":
                logger.error(f"Login failed: {data.get('message', 'Unknown error')}")
                return False
            
            # Extract credentials
            self.ssid = data.get("ssid")
            self.user_id = data.get("user_id")
            
            # Get account info
            if not self._fetch_account_info():
                return False
            
            self.is_connected = True
            self.initial_balance = self.balance
            logger.info(f"✓ Connected successfully. Balance: ${self.balance:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Connection failed: {str(e)}")
            return False
    
    def _fetch_account_info(self) -> bool:
        """Fetch current account information"""
        try:
            headers = {"ssid": self.ssid}
            response = requests.get(
                f"{self.api_url}/api/v1/profile",
                headers=headers,
                timeout=10
            )
            data = response.json()
            
            if data.get("result") == "ok":
                profile = data.get("profile", {})
                self.balance = float(profile.get("balance", 0))
                self.currency = profile.get("currency", "USD")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to fetch account info: {str(e)}")
            return False
    
    def check_safety_limits(self) -> Tuple[bool, str]:
        """
        Check if trading is allowed based on safety limits
        
        Returns:
            (is_safe, reason_if_not_safe)
        """
        # Check max trades per day
        if self.trades_today >= self.max_trades_per_day:
            return False, f"Max trades per day ({self.max_trades_per_day}) reached"
        
        # Check daily loss limit
        if self.daily_loss >= self.max_loss_per_day:
            return False, f"Daily loss limit (${self.max_loss_per_day}) reached"
        
        # Check equity stop-out
        loss_percentage = (self.initial_balance - self.balance) / self.initial_balance
        if loss_percentage >= self.equity_stop_out:
            return False, f"Account down {loss_percentage*100:.1f}% - equity stop-out triggered"
        
        # Check cooldown between trades (prevent spam)
        if self.last_trade_time:
            time_since_last = (datetime.now() - self.last_trade_time).total_seconds()
            if time_since_last < 5:  # Minimum 5 seconds between trades
                return False, "Trade cooldown active (5 sec minimum between trades)"
        
        return True, "OK"
    
    def get_balance(self) -> float:
        """Get current account balance"""
        self._fetch_account_info()
        return self.balance
    
    def execute_trade(
        self,
        asset: str,
        direction: TradeType,
        amount: float,
        expiration_minutes: int = 5,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Execute a trade on Pocket Option
        
        Args:
            asset: Asset symbol (e.g., 'AUD/USD', 'AED/CNY')
            direction: TradeType.BUY or TradeType.SELL
            amount: Trade amount in USD
            expiration_minutes: Minutes until expiration (1-60)
            stop_loss: Stop loss price (optional)
            take_profit: Take profit price (optional)
            
        Returns:
            (success, message, trade_id)
            
        WARNING: This executes REAL trades on real account!
        """
        # Safety checks
        is_safe, safety_msg = self.check_safety_limits()
        if not is_safe:
            logger.warning(f"Trade blocked: {safety_msg}")
            return False, f"Trade blocked: {safety_msg}", None
        
        if not self.is_connected:
            return False, "Not connected to Pocket Option", None
        
        try:
            logger.info(f"Executing {direction.value.upper()} {asset} for ${amount} ({expiration_minutes} min)")
            
            # Prepare trade payload
            trade_url = f"{self.api_url}/api/v1/trade"
            headers = {"ssid": self.ssid}
            
            payload = {
                "user_id": self.user_id,
                "pair": asset,
                "direction": direction.value,  # "call" or "put"
                "amount": amount,
                "expiration": expiration_minutes * 60,  # Convert to seconds
                "type": "forex"  # Asset type
            }
            
            # Add optional stops if provided
            if stop_loss:
                payload["stop_loss"] = stop_loss
            if take_profit:
                payload["take_profit"] = take_profit
            
            response = requests.post(trade_url, json=payload, headers=headers, timeout=10)
            data = response.json()
            
            if data.get("result") == "ok":
                trade_id = data.get("id")
                self.active_trades[trade_id] = {
                    "asset": asset,
                    "direction": direction.value,
                    "amount": amount,
                    "entry_time": datetime.now(),
                    "expiration_minutes": expiration_minutes,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit
                }
                
                self.last_trade_time = datetime.now()
                self.trades_today += 1
                
                logger.info(f"✓ Trade executed! ID: {trade_id}")
                return True, f"Trade {trade_id} executed", trade_id
            else:
                error_msg = data.get("message", "Unknown error")
                logger.error(f"Trade execution failed: {error_msg}")
                return False, f"Trade failed: {error_msg}", None
                
        except Exception as e:
            logger.error(f"Trade execution exception: {str(e)}")
            return False, f"Trade exception: {str(e)}", None
    
    def get_trade_status(self, trade_id: str) -> Dict:
        """
        Get status of a specific trade
        
        Returns:
            Trade info dictionary
        """
        try:
            headers = {"ssid": self.ssid}
            response = requests.get(
                f"{self.api_url}/api/v1/trades/{trade_id}",
                headers=headers,
                timeout=10
            )
            data = response.json()
            
            if data.get("result") == "ok":
                return data.get("trade", {})
            return {}
            
        except Exception as e:
            logger.error(f"Failed to get trade status: {str(e)}")
            return {}
    
    def close_trade(self, trade_id: str) -> Tuple[bool, str]:
        """
        Manually close a trade before expiration
        
        Returns:
            (success, message)
        """
        try:
            if trade_id not in self.active_trades:
                return False, f"Trade {trade_id} not found"
            
            headers = {"ssid": self.ssid}
            response = requests.post(
                f"{self.api_url}/api/v1/trades/{trade_id}/close",
                headers=headers,
                timeout=10
            )
            data = response.json()
            
            if data.get("result") == "ok":
                # Update trade history
                trade_info = self.active_trades.pop(trade_id)
                trade_info["close_time"] = datetime.now()
                self.trade_history.append(trade_info)
                
                # Update balance
                self._fetch_account_info()
                
                logger.info(f"✓ Trade {trade_id} closed")
                return True, f"Trade {trade_id} closed"
            else:
                return False, data.get("message", "Unknown error")
                
        except Exception as e:
            logger.error(f"Failed to close trade: {str(e)}")
            return False, str(e)
    
    def get_active_trades(self) -> Dict:
        """Get all active trades"""
        return self.active_trades.copy()
    
    def get_trade_history(self) -> list:
        """Get trade history"""
        return self.trade_history.copy()
    
    def disconnect(self):
        """Disconnect from Pocket Option"""
        try:
            # Close all open trades if needed
            for trade_id in list(self.active_trades.keys()):
                self.close_trade(trade_id)
            
            self.is_connected = False
            logger.info("Disconnected from Pocket Option")
            
        except Exception as e:
            logger.error(f"Disconnect error: {str(e)}")
    
    def __repr__(self):
        return (
            f"PocketOptionBroker("
            f"Account: {self.account_type.value}, "
            f"Balance: ${self.balance:.2f}, "
            f"Connected: {self.is_connected})"
        )
