"""Live Trading Configuration and Account Setup"""

import os
import json
import getpass
from pathlib import Path
from brokers.pocket_option import AccountType

# Path for secure credentials storage
CREDENTIALS_FILE = Path("config/.credentials")


class LiveTradingConfig:
    """
    Manages live trading configuration
    - Account credentials
    - Trading parameters
    - Safety limits
    """
    
    def __init__(self):
        """Initialize live trading config"""
        self.broker_email = None
        self.broker_password = None
        self.account_type = AccountType.DEMO
        self.initial_balance = 10000
        self.risk_per_trade = 0.01
        self.max_position_size = 0.05
        self.max_daily_trades = 20
        self.max_daily_loss = 100
        self.equity_stop_out = 0.5
        self.demo_mode = True  # Safety default
        self.asset = "AED/CNY"
        self.expiration_minutes = 5
    
    def setup_account_interactive(self):
        """
        Interactive setup for live trading account
        
        IMPORTANT: This is a critical security step!
        """
        print("\n" + "="*70)
        print("LIVE TRADING ACCOUNT SETUP")
        print("="*70)
        print("\n⚠️  SECURITY WARNING:")
        print("   - Your credentials will be stored locally (encrypted if possible)")
        print("   - Never share these credentials")
        print("   - Always use DEMO account first to test")
        print("   - Real money trading can result in total loss")
        print("\n" + "="*70 + "\n")
        
        # Account selection
        print("1. Account Type Selection")
        print("   [1] DEMO Account (Recommended for testing)")
        print("   [2] REAL Account (WARNING: Real money)")
        choice = input("\n   Choose account type [1 or 2]: ").strip()
        
        if choice == "2":
            confirm = input("\n   ⚠️  You selected REAL ACCOUNT. This will trade REAL MONEY.\n"
                          "   Type 'YES I UNDERSTAND' to confirm: ")
            if confirm != "YES I UNDERSTAND":
                print("   ✗ Real account setup cancelled")
                self.account_type = AccountType.DEMO
                self.demo_mode = True
                return False
            else:
                self.account_type = AccountType.REAL
                self.demo_mode = False
                print("   ✓ Real account mode ENABLED (Trading real money)")
        else:
            self.account_type = AccountType.DEMO
            self.demo_mode = True
            print("   ✓ Demo account mode ENABLED (Safe testing)")
        
        # Credentials
        print("\n2. Pocket Option Credentials")
        self.broker_email = input("   Email: ").strip()
        self.broker_password = getpass.getpass("   Password (hidden): ")
        
        # Account balance
        print("\n3. Account Balance")
        balance_input = input("   Starting balance (default $10,000): ").strip()
        try:
            self.initial_balance = float(balance_input) if balance_input else 10000
        except ValueError:
            self.initial_balance = 10000
        print(f"   ✓ Balance set to: ${self.initial_balance:.2f}")
        
        # Risk settings
        print("\n4. Risk Management")
        
        risk_input = input("   Risk per trade as % (default 1%): ").strip()
        try:
            self.risk_per_trade = float(risk_input) / 100 if risk_input else 0.01
        except ValueError:
            self.risk_per_trade = 0.01
        print(f"   ✓ Risk per trade: {self.risk_per_trade*100:.1f}%")
        
        max_pos_input = input("   Max position size as % (default 5%): ").strip()
        try:
            self.max_position_size = float(max_pos_input) / 100 if max_pos_input else 0.05
        except ValueError:
            self.max_position_size = 0.05
        print(f"   ✓ Max position size: {self.max_position_size*100:.1f}%")
        
        # Safety limits
        print("\n5. Safety Limits")
        
        max_trades_input = input("   Max trades per day (default 20): ").strip()
        try:
            self.max_daily_trades = int(max_trades_input) if max_trades_input else 20
        except ValueError:
            self.max_daily_trades = 20
        print(f"   ✓ Max daily trades: {self.max_daily_trades}")
        
        max_loss_input = input(f"   Max daily loss in $ (default $100): ").strip()
        try:
            self.max_daily_loss = float(max_loss_input) if max_loss_input else 100
        except ValueError:
            self.max_daily_loss = 100
        print(f"   ✓ Max daily loss: ${self.max_daily_loss:.2f}")
        
        stop_out_input = input("   Equity stop-out % (default 50%): ").strip()
        try:
            self.equity_stop_out = float(stop_out_input) / 100 if stop_out_input else 0.5
        except ValueError:
            self.equity_stop_out = 0.5
        print(f"   ✓ Equity stop-out: {self.equity_stop_out*100:.1f}%")
        
        # Asset and expiration
        print("\n6. Trade Parameters")
        
        asset_input = input("   Asset symbol (default AED/CNY): ").strip()
        self.asset = asset_input if asset_input else "AED/CNY"
        print(f"   ✓ Asset: {self.asset}")
        
        expir_input = input("   Expiration time in minutes (default 5): ").strip()
        try:
            self.expiration_minutes = int(expir_input) if expir_input else 5
        except ValueError:
            self.expiration_minutes = 5
        print(f"   ✓ Expiration: {self.expiration_minutes} minutes")
        
        # Summary
        print("\n" + "="*70)
        print("CONFIGURATION SUMMARY")
        print("="*70)
        print(f"Account Type:       {self.account_type.value.upper()}")
        print(f"Initial Balance:    ${self.initial_balance:.2f}")
        print(f"Risk Per Trade:     {self.risk_per_trade*100:.1f}%")
        print(f"Max Position Size:  {self.max_position_size*100:.1f}%")
        print(f"Max Daily Trades:   {self.max_daily_trades}")
        print(f"Max Daily Loss:     ${self.max_daily_loss:.2f}")
        print(f"Equity Stop-Out:    {self.equity_stop_out*100:.1f}%")
        print(f"Asset:              {self.asset}")
        print(f"Expiration:         {self.expiration_minutes} min")
        print(f"Demo Mode:          {'YES (Safe)' if self.demo_mode else 'NO (LIVE MONEY)'}")
        print("="*70)
        
        confirm = input("\nSave this configuration? [yes/no]: ").strip().lower()
        if confirm == "yes":
            self.save()
            print("✓ Configuration saved!")
            return True
        else:
            print("✗ Configuration not saved")
            return False
    
    def save(self):
        """Save configuration to file"""
        try:
            # Create config directory if it doesn't exist
            os.makedirs("config", exist_ok=True)
            
            config_data = {
                "account_type": self.account_type.value,
                "initial_balance": self.initial_balance,
                "risk_per_trade": self.risk_per_trade,
                "max_position_size": self.max_position_size,
                "max_daily_trades": self.max_daily_trades,
                "max_daily_loss": self.max_daily_loss,
                "equity_stop_out": self.equity_stop_out,
                "demo_mode": self.demo_mode,
                "asset": self.asset,
                "expiration_minutes": self.expiration_minutes
            }
            
            # Save to JSON (credentials stored separately)
            with open("config/live_trading.json", "w") as f:
                json.dump(config_data, f, indent=2)
            
            # IMPORTANT: Store credentials separately and securely
            if self.broker_email and self.broker_password:
                credentials = {
                    "email": self.broker_email,
                    "password": self.broker_password  # TODO: Encrypt this!
                }
                
                # Create secure credentials file
                cred_path = "config/.credentials"
                with open(cred_path, "w") as f:
                    json.dump(credentials, f)
                
                # Make file read-only
                os.chmod(cred_path, 0o600)
                
                print(f"Credentials saved to: {cred_path}")
        
        except Exception as e:
            print(f"Error saving configuration: {e}")
    
    def load(self):
        """Load configuration from file"""
        try:
            if os.path.exists("config/live_trading.json"):
                with open("config/live_trading.json", "r") as f:
                    data = json.load(f)
                    self.account_type = AccountType(data.get("account_type", "demo"))
                    self.initial_balance = data.get("initial_balance", 10000)
                    self.risk_per_trade = data.get("risk_per_trade", 0.01)
                    self.max_position_size = data.get("max_position_size", 0.05)
                    self.max_daily_trades = data.get("max_daily_trades", 20)
                    self.max_daily_loss = data.get("max_daily_loss", 100)
                    self.equity_stop_out = data.get("equity_stop_out", 0.5)
                    self.demo_mode = data.get("demo_mode", True)
                    self.asset = data.get("asset", "AED/CNY")
                    self.expiration_minutes = data.get("expiration_minutes", 5)
                    
                    return True
        except Exception as e:
            print(f"Error loading configuration: {e}")
        
        return False
    
    def load_credentials(self):
        """Load credentials from secure file"""
        try:
            cred_path = "config/.credentials"
            if os.path.exists(cred_path):
                with open(cred_path, "r") as f:
                    data = json.load(f)
                    self.broker_email = data.get("email")
                    self.broker_password = data.get("password")
                    return True
        except Exception as e:
            print(f"Error loading credentials: {e}")
        
        return False
    
    def __repr__(self):
        return (
            f"LiveTradingConfig("
            f"Account: {self.account_type.value}, "
            f"Balance: ${self.initial_balance:.2f}, "
            f"Mode: {'DEMO' if self.demo_mode else 'LIVE'})"
        )
