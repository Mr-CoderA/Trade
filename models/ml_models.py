"""Machine learning models for time series prediction with GPU acceleration."""

import numpy as np
import pandas as pd
from typing import Tuple, List
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras.optimizers import Adam
from utils.logger import setup_logger
from config import config

# Enable GPU memory growth to avoid OOM errors
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            setup_logger(__name__).info(f"GPU enabled: {gpu}")
    except RuntimeError as e:
        setup_logger(__name__).error(f"GPU configuration error: {e}")

logger = setup_logger(__name__)


class LSTMModel:
    """LSTM neural network for time series prediction."""
    
    def __init__(self, lookback_window: int = 60, forecast_horizon: int = 5):
        """Initialize LSTM model.
        
        Args:
            lookback_window: Number of time steps to look back
            forecast_horizon: Number of steps to forecast
        """
        self.lookback_window = lookback_window
        self.forecast_horizon = forecast_horizon
        self.model = None
        self.scaler = StandardScaler()
    
    def create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training.
        
        Args:
            data: Time series data
            
        Returns:
            Tuple of (X, y) sequences
        """
        X, y = [], []
        
        for i in range(len(data) - self.lookback_window - self.forecast_horizon + 1):
            X.append(data[i:i + self.lookback_window])
            y.append(data[i + self.lookback_window:i + self.lookback_window + self.forecast_horizon, -1])  # Use close price
        
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape: Tuple):
        """Build LSTM model architecture with GPU acceleration.
        
        Args:
            input_shape: Shape of input data (lookback_window, n_features)
        """
        with tf.device('/GPU:0') if gpus else tf.device('/CPU:0'):
            self.model = Sequential([
                layers.LSTM(64, activation='relu', input_shape=input_shape, return_sequences=True),
                layers.Dropout(0.2),
                layers.LSTM(32, activation='relu', return_sequences=False),
                layers.Dropout(0.2),
                layers.Dense(16, activation='relu'),
                layers.Dense(self.forecast_horizon)
            ])
            
            # Use GPU-accelerated optimizer
            self.model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )
        
        logger.info("LSTM model created with GPU acceleration")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int = 50, batch_size: int = 32, validation_split: float = 0.2):
        """Train LSTM model with GPU acceleration.
        
        Args:
            X_train: Training features
            y_train: Training targets
            epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Validation split ratio
        """
        if self.model is None:
            self.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        
        logger.info("Training LSTM model on GPU...")
        
        # Convert to GPU tensors if available
        if gpus:
            X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
            y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
        
        with tf.device('/GPU:0') if gpus else tf.device('/CPU:0'):
            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                verbose=0
            )
        
        logger.info(f"LSTM training complete on GPU. Final loss: {history.history['loss'][-1]:.6f}")
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions.
        
        Args:
            X: Input sequences
            
        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        return self.model.predict(X, verbose=0)


class EnsembleModel:
    """Ensemble model combining multiple algorithms."""
    
    def __init__(self, lookback_window: int = 60, forecast_horizon: int = 5):
        """Initialize ensemble model."""
        self.lookback_window = lookback_window
        self.forecast_horizon = forecast_horizon
        self.models = {
            'lstm': None,
            'xgboost': None,
            'lightgbm': None,
        }
        self.weights = config.get_nested('model.ensemble_weights', {
            'lstm': 0.4,
            'xgboost': 0.3,
            'lightgbm': 0.3
        })
        self.scaler = StandardScaler()
    
    def prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for ML models.
        
        Args:
            data: Dataframe with indicators and prices
            
        Returns:
            Feature array
        """
        # Select relevant features (exclude price columns)
        exclude_cols = ['close', 'open', 'high', 'low', 'volume']
        feature_cols = [col for col in data.columns if col not in exclude_cols]
        
        X = data[feature_cols].values
        X = self.scaler.fit_transform(X)
        
        return X
    
    def create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series models.
        
        Args:
            X: Features
            y: Targets
            
        Returns:
            Tuple of (sequences_X, sequences_y)
        """
        seq_X, seq_y = [], []
        
        for i in range(len(X) - self.lookback_window - self.forecast_horizon + 1):
            seq_X.append(X[i:i + self.lookback_window])
            seq_y.append(y[i + self.lookback_window:i + self.lookback_window + self.forecast_horizon])
        
        return np.array(seq_X), np.array(seq_y)
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 50):
        """Train ensemble models.
        
        Args:
            X: Feature sequences for LSTM
            y: Target sequences
        """
        logger.info("Training ensemble models...")
        
        # Flatten X for tree-based models
        n_samples, n_timesteps, n_features = X.shape
        X_flat = X.reshape(n_samples, n_timesteps * n_features)
        y_flat = y[:, 0]  # Use first prediction step
        
        # LSTM
        logger.info("Training LSTM...")
        self.models['lstm'] = LSTMModel(self.lookback_window, self.forecast_horizon)
        self.models['lstm'].build_model(input_shape=(X.shape[1], X.shape[2]))
        self.models['lstm'].train(X, y, epochs=epochs)
        
        # XGBoost with GPU acceleration
        logger.info("Training XGBoost on GPU...")
        self.models['xgboost'] = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            tree_method='gpu_hist',  # GPU-accelerated histogram
            gpu_id=0,  # Use first GPU
            predictor='gpu_predictor'  # GPU prediction
        )
        self.models['xgboost'].fit(X_flat, y_flat)
        
        # LightGBM with GPU acceleration
        logger.info("Training LightGBM on GPU...")
        self.models['lightgbm'] = lgb.LGBMRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            device_type='gpu',  # Use GPU
            gpu_device_id=0  # Use first GPU
        )
        self.models['lightgbm'].fit(X_flat, y_flat)
        
        logger.info("Ensemble training complete")
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, float]:
        """Make ensemble predictions.
        
        Args:
            X: Feature sequences
            
        Returns:
            Tuple of (predictions, confidence_score)
        """
        predictions = {}
        
        # LSTM prediction
        if self.models['lstm'] is not None:
            lstm_pred = self.models['lstm'].predict(X)
            predictions['lstm'] = lstm_pred[:, 0]  # First step prediction
        
        # Tree models - flatten input
        n_samples, n_timesteps, n_features = X.shape
        X_flat = X.reshape(n_samples, n_timesteps * n_features)
        
        if self.models['xgboost'] is not None:
            predictions['xgboost'] = self.models['xgboost'].predict(X_flat)
        
        if self.models['lightgbm'] is not None:
            predictions['lightgbm'] = self.models['lightgbm'].predict(X_flat)
        
        # Weighted ensemble
        ensemble_pred = np.zeros(len(next(iter(predictions.values()))))
        total_weight = 0
        
        for model_name, pred in predictions.items():
            weight = self.weights.get(model_name, 1/3)
            ensemble_pred += pred * weight
            total_weight += weight
        
        ensemble_pred /= total_weight
        
        # Calculate confidence (agreement between models)
        if len(predictions) > 1:
            pred_values = np.array(list(predictions.values()))
            std_dev = np.std(pred_values, axis=0)
            confidence = 1 / (1 + std_dev)  # Higher std = lower confidence
        else:
            confidence = 0.5 * np.ones(len(ensemble_pred))
        
        return ensemble_pred, confidence


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Evaluate model performance.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary with performance metrics
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Direction accuracy (for classification of up/down)
    true_direction = np.sign(y_true - y_true[0])
    pred_direction = np.sign(y_pred - y_pred[0])
    direction_accuracy = np.sum(true_direction == pred_direction) / len(y_true)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'direction_accuracy': direction_accuracy
    }
