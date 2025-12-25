"""Screen capture and OCR module for real-time data extraction."""

import numpy as np
from typing import Tuple, Optional, Dict
import mss
import cv2
import pytesseract
from PIL import Image
import re
from utils.logger import setup_logger
from config import config

logger = setup_logger(__name__)


class ScreenCapture:
    """Capture screen content."""
    
    def __init__(self, region: Optional[Tuple[int, int, int, int]] = None):
        """Initialize screen capture.
        
        Args:
            region: Tuple of (x1, y1, x2, y2) for region of interest, None for full screen
        """
        self.region = region
        self.sct = mss.mss()
    
    def capture(self) -> np.ndarray:
        """Capture screen content.
        
        Returns:
            Screen capture as numpy array
        """
        if self.region:
            x1, y1, x2, y2 = self.region
            monitor = {'top': y1, 'left': x1, 'width': x2 - x1, 'height': y2 - y1}
        else:
            monitor = self.sct.monitors[1]  # Full screen
        
        screenshot = self.sct.grab(monitor)
        img = np.array(screenshot)
        
        # Convert from BGRA to BGR
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)


class PriceOCR:
    """Extract price data from screen captures using OCR."""
    
    def __init__(self):
        """Initialize price OCR."""
        self.screen_capture = ScreenCapture()
    
    def extract_prices(self) -> Dict[str, float]:
        """Extract price data from screen capture.
        
        Returns:
            Dictionary with extracted prices
        """
        try:
            img = self.screen_capture.capture()
            
            # Preprocess image for OCR
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply thresholding
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
            
            # Use Tesseract to extract text
            text = pytesseract.image_to_string(thresh)
            
            # Extract price patterns
            prices = self._parse_prices(text)
            
            logger.info(f"Extracted prices: {prices}")
            return prices
            
        except Exception as e:
            logger.error(f"Error in OCR extraction: {e}")
            return {}
    
    def _parse_prices(self, text: str) -> Dict[str, float]:
        """Parse price data from OCR text.
        
        Args:
            text: OCR extracted text
            
        Returns:
            Dictionary with extracted values
        """
        prices = {}
        
        # Look for price patterns (e.g., "3.45", "0.12")
        price_pattern = r'(\d+\.?\d*)'
        matches = re.findall(price_pattern, text)
        
        if matches:
            try:
                prices['bid'] = float(matches[0]) if len(matches) > 0 else None
                prices['ask'] = float(matches[1]) if len(matches) > 1 else None
                prices['mid'] = (prices['bid'] + prices['ask']) / 2 if prices['bid'] and prices['ask'] else None
            except (ValueError, IndexError):
                logger.warning("Could not parse prices from OCR text")
        
        return prices
    
    def extract_chart_candles(self, sensitivity: float = 0.5) -> Dict:
        """Attempt to extract candle data from chart visualization.
        
        Args:
            sensitivity: Detection sensitivity (0-1)
            
        Returns:
            Dictionary with detected candle information
        """
        try:
            img = self.screen_capture.capture()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect edges
            edges = cv2.Canny(gray, 50, 150)
            
            # Find contours (potential candles)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            candle_info = {
                'detected_candles': len(contours),
                'contours': contours
            }
            
            logger.info(f"Detected {len(contours)} potential candles in chart")
            return candle_info
            
        except Exception as e:
            logger.error(f"Error detecting chart candles: {e}")
            return {'detected_candles': 0}


class MonitoringService:
    """Monitor screen for real-time price updates."""
    
    def __init__(self, update_interval: int = 5):
        """Initialize monitoring service.
        
        Args:
            update_interval: Update interval in seconds
        """
        self.update_interval = update_interval
        self.price_ocr = PriceOCR()
        self.last_prices = {}
    
    def get_latest_prices(self) -> Dict[str, float]:
        """Get latest prices from screen monitoring.
        
        Returns:
            Dictionary with latest prices
        """
        prices = self.price_ocr.extract_prices()
        self.last_prices = prices
        return prices
    
    def detect_price_change(self) -> Optional[Dict]:
        """Detect significant price changes.
        
        Returns:
            Dictionary with change information if detected, None otherwise
        """
        prices = self.get_latest_prices()
        
        if not prices or not self.last_prices:
            return None
        
        changes = {}
        for key in prices:
            if key in self.last_prices and self.last_prices[key]:
                change_pct = (prices[key] - self.last_prices[key]) / self.last_prices[key]
                if abs(change_pct) > 0.001:  # 0.1% change threshold
                    changes[key] = change_pct
        
        return changes if changes else None
