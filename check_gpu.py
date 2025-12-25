#!/usr/bin/env python3
"""Quick GPU verification script."""

import sys

def check_gpu():
    """Check GPU configuration."""
    print("\n" + "="*80)
    print("GPU ACCELERATION CHECK")
    print("="*80 + "\n")
    
    # TensorFlow
    print("1. TensorFlow GPU Support:")
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"   ✓ {len(gpus)} GPU(s) found")
            for i, gpu in enumerate(gpus):
                print(f"     {i+1}. {gpu}")
        else:
            print("   ✗ No GPUs detected")
            print("   → Install CUDA 12.x and cuDNN 8.x")
        print(f"   TensorFlow Version: {tf.__version__}")
        print(f"   CUDA Available: {tf.test.is_built_with_cuda()}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # XGBoost
    print("\n2. XGBoost GPU Support:")
    try:
        import xgboost as xgb
        model = xgb.XGBRegressor(tree_method='gpu_hist', gpu_id=0)
        print("   ✓ XGBoost GPU support available")
    except Exception as e:
        print(f"   ✗ XGBoost GPU not available: {str(e)[:50]}...")
    
    # LightGBM
    print("\n3. LightGBM GPU Support:")
    try:
        import lightgbm as lgb
        model = lgb.LGBMRegressor(device_type='gpu', gpu_device_id=0)
        print("   ✓ LightGBM GPU support available")
    except Exception as e:
        print(f"   ✗ LightGBM GPU not available: {str(e)[:50]}...")
    
    # NVIDIA Tools
    print("\n4. NVIDIA Tools:")
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            gpus_info = result.stdout.strip().split('\n')
            for i, info in enumerate(gpus_info):
                print(f"   ✓ GPU {i}: {info}")
        else:
            print("   ✗ nvidia-smi not available")
            print("   → Update NVIDIA GPU drivers")
    except Exception as e:
        print(f"   ✗ Error checking nvidia-smi: {e}")
    
    # Summary
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("""
If GPU not detected:
1. Download NVIDIA CUDA from: https://developer.nvidia.com/cuda-downloads
2. Download cuDNN from: https://developer.nvidia.com/cudnn
3. Update GPU drivers: https://www.nvidia.com/Download/driverDetails.aspx
4. Run: pip install -r requirements.txt
5. Restart Python interpreter

To run full diagnostics:
   python -c "from utils.gpu_utils import diagnose_gpu_setup; diagnose_gpu_setup()"

To run training with GPU:
   python main.py

To monitor GPU during training:
   nvidia-smi -l 1
""")
    print("="*80 + "\n")

if __name__ == '__main__':
    check_gpu()
