#!/usr/bin/env python3
"""
Test script for MLflow integration.
This script tests the MLflow functionality without running the full training pipeline.
"""

import os
import sys
import numpy as np
import pandas as pd

# Add src to path
sys.path.append('src')

from src.modelling.mlflow_utils import *
from src.modelling.model_train import setup_mlflow

def test_mlflow_setup():
    """Test MLflow setup and configuration."""
    print("Testing MLflow setup...")
    
    try:
        import mlflow
        print(f"✓ MLflow version: {mlflow.__version__}")
        
        # Test experiment setup
        client = setup_mlflow()
        print("✓ MLflow client initialized")
        
        # Test experiment creation
        experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
        if experiment:
            print(f"✓ Experiment '{MLFLOW_EXPERIMENT_NAME}' exists")
        else:
            print(f"✗ Experiment '{MLFLOW_EXPERIMENT_NAME}' not found")
        
        return True
        
    except Exception as e:
        print(f"✗ MLflow setup failed: {e}")
        return False

def test_model_registry():
    """Test model registry functionality."""
    print("\nTesting Model Registry...")
    
    try:
        client = MlflowClient()
        
        # List available models
        print("Available models in registry:")
        models = list_available_models()
        
        if models:
            print(f"✓ Found {len(models)} model versions")
            
            # Test loading latest model
            model, version = load_model_from_registry()
            if model is not None:
                print(f"✓ Successfully loaded model version {version}")
                
                # Test model info
                model_info = get_model_info(version)
                if model_info:
                    print("✓ Model info retrieved successfully")
                
                return True
            else:
                print("✗ Failed to load model from registry")
                return False
        else:
            print("ℹ No models found in registry (this is expected for first run)")
            return True
            
    except Exception as e:
        print(f"✗ Model registry test failed: {e}")
        return False

def test_inference():
    """Test model inference with dummy data."""
    print("\nTesting Model Inference...")
    
    try:
        # Create dummy test data with correct number of features (30, not 31)
        # Features: V1-V28 (28) + Time (1) + Amount (1) = 30 total
        dummy_data = np.random.randn(10, 30)  # 30 features (V1-V28, Time, Amount)
        
        # Try to load model
        model, version = load_model_from_registry()
        
        if model is not None:
            # Test predictions
            predictions, probabilities = predict_fraud(model, dummy_data)
            
            if predictions is not None:
                print(f"✓ Made predictions on {len(predictions)} samples")
                print(f"✓ Predictions shape: {predictions.shape}")
                print(f"✓ Probabilities shape: {probabilities.shape}")
                return True
            else:
                print("✗ Prediction failed")
                return False
        else:
            print("ℹ No model available for inference test")
            return True
            
    except Exception as e:
        print(f"✗ Inference test failed: {e}")
        return False

def main():
    """Run all MLflow tests."""
    print("="*60)
    print("MLflow Integration Test Suite")
    print("="*60)
    
    tests = [
        ("MLflow Setup", test_mlflow_setup),
        ("Model Registry", test_model_registry),
        ("Model Inference", test_inference)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "="*60)
    print("Test Results Summary")
    print("="*60)
    
    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("✓ All tests passed! MLflow integration is working correctly.")
    else:
        print("✗ Some tests failed. Check the error messages above.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
