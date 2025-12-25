"""GPU utilities and diagnostics for accelerated training."""

import os
import sys
import tensorflow as tf
import numpy as np
from typing import Dict, List
from utils.logger import setup_logger

logger = setup_logger(__name__)


def check_gpu_availability() -> bool:
    """Check if GPU is available for TensorFlow.
    
    Returns:
        True if GPU is available, False otherwise
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        logger.info(f"✓ {len(gpus)} GPU(s) detected")
        return True
    else:
        logger.warning("⚠ No GPU detected, will use CPU")
        return False


def get_gpu_details() -> Dict:
    """Get detailed information about available GPUs.
    
    Returns:
        Dictionary with GPU details
    """
    gpus = tf.config.list_physical_devices('GPU')
    cpu_devices = tf.config.list_physical_devices('CPU')
    
    details = {
        'gpus_available': len(gpus),
        'gpu_devices': [str(gpu) for gpu in gpus],
        'cpu_count': len(cpu_devices),
        'tensorflow_version': tf.__version__,
        'cuda_available': tf.test.is_built_with_cuda(),
        'gpu_support': True if gpus else False
    }
    
    return details


def setup_gpu_memory_growth():
    """Configure GPU memory growth to prevent OOM errors."""
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        logger.warning("No GPUs found, skipping memory growth configuration")
        return False
    
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info("✓ GPU memory growth configured")
        return True
    except Exception as e:
        logger.error(f"Error configuring GPU memory growth: {e}")
        return False


def set_gpu_compute_capability(capability: str = 'fp32'):
    """Set GPU compute capability for mixed precision training.
    
    Args:
        capability: 'fp32', 'fp16', or 'mixed'
    """
    if capability == 'mixed':
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        logger.info("✓ Mixed precision (FP16) training enabled")
    elif capability == 'fp16':
        policy = tf.keras.mixed_precision.Policy('float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        logger.info("✓ FP16 training enabled")
    else:
        logger.info("Using FP32 (standard) precision")


def benchmark_gpu_vs_cpu(data_size: int = 1000) -> Dict:
    """Benchmark GPU vs CPU performance.
    
    Args:
        data_size: Size of data for benchmark
        
    Returns:
        Dictionary with timing results
    """
    import time
    
    # Create test data
    data = tf.random.normal((data_size, 100))
    
    # GPU benchmark
    results = {'gpu': None, 'cpu': None}
    
    if check_gpu_availability():
        with tf.device('/GPU:0'):
            start = time.time()
            for _ in range(10):
                _ = tf.matmul(data, tf.transpose(data))
            results['gpu'] = time.time() - start
        logger.info(f"GPU benchmark: {results['gpu']:.4f} seconds")
    
    # CPU benchmark
    with tf.device('/CPU:0'):
        start = time.time()
        for _ in range(10):
            _ = tf.matmul(data, tf.transpose(data))
        results['cpu'] = time.time() - start
    logger.info(f"CPU benchmark: {results['cpu']:.4f} seconds")
    
    if results['gpu']:
        speedup = results['cpu'] / results['gpu']
        logger.info(f"GPU Speedup: {speedup:.2f}x")
        results['speedup'] = speedup
    
    return results


def print_gpu_info():
    """Print comprehensive GPU information."""
    logger.info("="*80)
    logger.info("GPU CONFIGURATION SUMMARY")
    logger.info("="*80)
    
    details = get_gpu_details()
    
    logger.info(f"TensorFlow Version: {details['tensorflow_version']}")
    logger.info(f"CUDA Support: {details['cuda_available']}")
    logger.info(f"GPUs Available: {details['gpus_available']}")
    
    if details['gpus_available'] > 0:
        logger.info("GPU Devices:")
        for i, gpu in enumerate(details['gpu_devices']):
            logger.info(f"  {i+1}. {gpu}")
    
    logger.info(f"CPU Devices: {details['cpu_count']}")
    logger.info(f"GPU Support Enabled: {details['gpu_support']}")
    logger.info("="*80)


def check_xgboost_gpu_support() -> bool:
    """Check if XGBoost has GPU support.
    
    Returns:
        True if XGBoost GPU support is available
    """
    try:
        import xgboost as xgb
        # Try to create a model with GPU
        model = xgb.XGBRegressor(tree_method='gpu_hist', gpu_id=0)
        logger.info("✓ XGBoost GPU support available")
        return True
    except Exception as e:
        logger.warning(f"XGBoost GPU support not available: {e}")
        return False


def check_lightgbm_gpu_support() -> bool:
    """Check if LightGBM has GPU support.
    
    Returns:
        True if LightGBM GPU support is available
    """
    try:
        import lightgbm as lgb
        # Try to create a model with GPU
        model = lgb.LGBMRegressor(device_type='gpu', gpu_device_id=0)
        logger.info("✓ LightGBM GPU support available")
        return True
    except Exception as e:
        logger.warning(f"LightGBM GPU support not available: {e}")
        return False


def diagnose_gpu_setup():
    """Run comprehensive GPU diagnostics."""
    logger.info("\n" + "="*80)
    logger.info("GPU DIAGNOSTICS")
    logger.info("="*80 + "\n")
    
    # TensorFlow GPU
    logger.info("TensorFlow GPU Support:")
    tf_gpu = check_gpu_availability()
    if tf_gpu:
        print_gpu_info()
    
    # Memory growth
    logger.info("\nGPU Memory Configuration:")
    setup_gpu_memory_growth()
    
    # XGBoost GPU
    logger.info("\nXGBoost GPU Support:")
    xgb_gpu = check_xgboost_gpu_support()
    
    # LightGBM GPU
    logger.info("\nLightGBM GPU Support:")
    lgb_gpu = check_lightgbm_gpu_support()
    
    # Benchmark
    if tf_gpu:
        logger.info("\nPerformance Benchmark (GPU vs CPU):")
        benchmark_results = benchmark_gpu_vs_cpu()
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("DIAGNOSTICS SUMMARY")
    logger.info("="*80)
    logger.info(f"TensorFlow GPU: {'✓ ENABLED' if tf_gpu else '✗ NOT AVAILABLE'}")
    logger.info(f"XGBoost GPU: {'✓ ENABLED' if xgb_gpu else '✗ NOT AVAILABLE'}")
    logger.info(f"LightGBM GPU: {'✓ ENABLED' if lgb_gpu else '✗ NOT AVAILABLE'}")
    logger.info("="*80 + "\n")
    
    return {
        'tensorflow_gpu': tf_gpu,
        'xgboost_gpu': xgb_gpu,
        'lightgbm_gpu': lgb_gpu
    }
