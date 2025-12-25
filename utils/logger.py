"""Logging module for trading system."""

import logging
import logging.handlers
from pathlib import Path
from config import config

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

LOG_CONFIG = config.get('logging', {})
LOG_LEVEL = getattr(logging, LOG_CONFIG.get('level', 'INFO'))
LOG_FORMAT = LOG_CONFIG.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
LOG_FILE = LOG_DIR / LOG_CONFIG.get('file', 'trading_system.log').split('/')[-1]


def setup_logger(name: str) -> logging.Logger:
    """Setup logger with file and console handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)
    
    # File handler
    file_handler = logging.handlers.RotatingFileHandler(
        LOG_FILE,
        maxBytes=LOG_CONFIG.get('max_file_size', 10485760),
        backupCount=5
    )
    file_handler.setLevel(LOG_LEVEL)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(LOG_LEVEL)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    
    # Add handlers if not already present
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger
