# MLflow Integration for Fraud Detection System

This document describes the comprehensive MLflow integration for model versioning, experiment tracking, and model registry in the fraud detection system.

## Overview

The fraud detection system includes enterprise-grade MLflow integration for:
- **Experiment Tracking**: Log all training runs, parameters, and metrics with automatic artifact storage
- **Model Versioning**: Track different model versions and their performance with intelligent model loading
- **Model Registry**: Centralized model storage and management with stage transitions
- **Model Deployment**: Production-ready model loading and inference with robust error handling
- **Model Comparison**: Compare different model versions and their performance metrics

## Features

### 1. Experiment Tracking
- **Automatic Parameter Logging**: All model hyperparameters logged automatically
- **Comprehensive Metrics**: Accuracy, ROC-AUC, F1-score, Average Precision, Precision@K, Recall@K
- **Artifact Storage**: Models, precision-recall curves, metadata, and model signatures
- **Nested Runs**: Individual model training runs with separate tracking
- **Model Signatures**: Automatic signature inference for production deployment

### 2. Model Registry
- **Centralized Storage**: All models stored in MLflow Model Registry
- **Version Control**: Automatic versioning with detailed metadata
- **Stage Management**: Production, Staging, and None stages with automatic transitions
- **Model Descriptions**: Rich metadata and performance descriptions
- **Multi-Flavor Support**: Handles both sklearn and xgboost model flavors

### 3. Model Loading & Inference
- **Intelligent Loading**: Automatic model flavor detection (sklearn → xgboost → pyfunc)
- **Production-Ready**: Robust error handling and graceful fallbacks
- **Model Comparison**: Side-by-side comparison of different model versions
- **Version-Specific Access**: Load models by version, stage, or latest
- **Feature Validation**: Automatic feature shape validation for inference

## Setup

### Prerequisites
```bash
# Install dependencies
uv sync
```

### MLflow Server (Optional)
For production use, start an MLflow tracking server:
```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```

### Quick Start with MLflow UI
Use the provided launcher script for easy MLflow UI access:
```bash
python start_mlflow_ui.py
```
This will start the MLflow UI at http://localhost:5000 with proper configuration.

## Usage

### 1. Training with MLflow Tracking 

Run the main training pipeline with MLflow integration:

```bash
python main.py
```

This will:
- Train both RandomForest and XGBoost models with MLflow tracking
- Log all parameters, metrics, and artifacts to MLflow
- Register both models in the Model Registry with automatic versioning
- Select and promote the best model to Production stage (if AP > 0.8)
- Save local model files for backup and offline access
- Demonstrate model loading and inference capabilities

### 2. Model Loading and Inference

```python
from src.modelling.mlflow_utils import *

# Load the best model from registry (intelligent flavor detection)
model, version = load_model_from_registry(stage="Production")

# Make predictions with proper feature validation
predictions, probabilities = predict_fraud(model, test_data, threshold=0.5)

# Get model information
model_info = get_model_info(version)
print(f"Model: {model_info['model_name']} v{model_info['version']}")
```

### 3. Advanced Model Management

```python
# Load specific model version
model, version = load_model_from_registry(stage=None)  # Latest version

# Load model from specific run
model = load_model_from_run("run_id_here")

# Compare all model versions
comparison_df = compare_model_versions()

# List all available models
available_models = list_available_models()
```

### 4. Testing and Validation

```python
# Run comprehensive MLflow integration tests
python test_mlflow.py

# Test enhanced MLflow logging features
python test_enhanced_mlflow.py

# Test specific functionality
from src.modelling.mlflow_utils import *
model, version = load_model_from_registry()
print(f"✓ Model loaded: {model is not None}")
```

### 5. Enhanced MLflow Features

The system now includes advanced MLflow logging capabilities:

```python
# Create model comparison plots
from src.modelling.model_train import plot_model_comparison
comparison_path = plot_model_comparison(metrics_list)

# Generate training summaries
from src.modelling.model_train import create_training_summary
summary = create_training_summary(metrics, best_model, train_file, test_file, threshold, k)

# Access all artifacts through MLflow UI
# - Navigate to experiments → select run → artifacts tab
# - Download serialized models, plots, and reports
# - View visualizations directly in the browser
```

## MLflow UI

Access the MLflow UI to view experiments and models:

### Option 1: Quick Start (Recommended)
```bash
python start_mlflow_ui.py
```

### Option 2: Manual Start
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```

Then open http://localhost:5000 in your browser to view:
- **Experiments**: All training runs with parameters and metrics
- **Models**: Model Registry with versions and stages
- **Artifacts**: Model files, plots, and metadata
- **Runs**: Detailed run information and comparisons

### Accessing Artifacts in MLflow UI

1. **Navigate to Experiments**: Click on "Experiments" in the left sidebar
2. **Select Run**: Click on any training run to view details
3. **View Artifacts**: Click on the "Artifacts" tab to see all logged files
4. **Download Files**: Click on any file to download it directly
5. **View Images**: Click on PNG files to view them in the browser

### Available Downloads

- **Serialized Models**: Download `.joblib` files for offline use
- **Visualizations**: Download PR curves and comparison plots
- **Reports**: Download training summaries and documentation
- **Metadata**: Download JSON configuration files

## File Structure

```
fraud_detection_system/
├── src/modelling/
│   ├── model_train.py          # MLflow-integrated training pipeline
│   ├── mlflow_utils.py         # MLflow utilities for model loading
│   └── ...
├── mlruns/                     # MLflow tracking data (auto-created)
│   ├── 0/                      # Experiment runs
│   ├── models/                 # Model Registry storage
│   └── mlflow.db              # MLflow database
├── models/                     # Local model storage (backup)
│   ├── RandomForest_model.joblib
│   ├── RandomForest_metadata.joblib
│   └── ...
├── test_mlflow.py             # MLflow integration tests
├── start_mlflow_ui.py         # MLflow UI launcher
├── MLFLOW_README.md           # This documentation
└── main.py                    # Main training pipeline
```

## Configuration

### MLflow Settings
The following constants are defined in `model_train.py`:

```python
MLFLOW_EXPERIMENT_NAME = "fraud_detection_experiments"
MLFLOW_MODEL_REGISTRY_NAME = "fraud_detection_model"
```

### Model Registry Stages
- **Production**: Best performing model for production use
- **Staging**: Model candidate for production
- **None**: Latest model version

## Testing

Run the comprehensive MLflow integration tests:

```bash
python test_mlflow.py
```

### Test Coverage
The test suite validates:
- **MLflow Setup**: Experiment creation and client initialization
- **Model Registry**: Model versioning and stage management
- **Model Loading**: Intelligent flavor detection and loading
- **Model Inference**: Prediction functionality with proper feature validation
- **Error Handling**: Graceful handling of edge cases and API changes
- **API Compatibility**: Support for both new and deprecated MLflow APIs

### Expected Output
```
============================================================
MLflow Integration Test Suite
============================================================

--- MLflow Setup ---
✓ MLflow version: 3.4.0
✓ MLflow client initialized
✓ Experiment 'fraud_detection_experiments' exists

--- Model Registry ---
✓ Found X model versions
✓ Successfully loaded model version X
✓ Model info retrieved successfully

--- Model Inference ---
✓ Made predictions on 10 samples
✓ Predictions shape: (10,)
✓ Probabilities shape: (10,)

============================================================
Test Results Summary
============================================================
MLflow Setup: PASS
Model Registry: PASS
Model Inference: PASS

Overall: 3/3 tests passed
✓ All tests passed! MLflow integration is working correctly.
```

## Key Functions

### Training Functions
- `run_training_pipeline()`: Main training pipeline with MLflow integration
- `train_random_forest()`: RandomForest training with parameter logging
- `train_xgboost()`: XGBoost training with parameter logging
- `evaluate_model()`: Model evaluation with metrics logging

### MLflow Utilities
- `setup_mlflow()`: Initialize MLflow experiment and client
- `log_model_to_mlflow()`: Log model to MLflow with artifacts and signatures
- `register_model_version()`: Register model in Model Registry with error handling
- `load_model_from_registry()`: Intelligent model loading with flavor detection
- `load_model_from_run()`: Load model from specific MLflow run
- `predict_fraud()`: Make fraud predictions with feature validation
- `get_model_info()`: Retrieve detailed model information
- `list_available_models()`: List all models in registry
- `compare_model_versions()`: Compare different model versions and metrics

## Model Artifacts

Each model run logs the following comprehensive artifacts:

### Core Model Files
- **Model**: Serialized model file with proper flavor (sklearn/xgboost)
- **Model Signature**: Input/output schema for production deployment
- **Metadata**: Model metadata and configuration (JSON format)
- **Input Example**: Sample input data for model validation

### Visualizations
- **Precision-Recall Curve**: Individual PR curve for each model (PNG format)
- **Model Comparison Plot**: Side-by-side performance comparison (PNG format)
- **High-Quality Images**: All plots saved at 300 DPI for crisp visualization

### Serialized Models (Downloadable)
- **RandomForest Model**: `RandomForest_model.joblib` - Ready for offline use
- **XGBoost Model**: `XGBoost_model.joblib` - Ready for offline use
- **Model Metadata**: `*_metadata.joblib` - Training configuration and metrics

### Reports and Documentation
- **Training Summary**: Comprehensive markdown report with all metrics
- **Model Summary**: Text summary with key performance indicators
- **Usage Instructions**: Built-in documentation for model deployment

### Organized Artifact Structure
```
artifacts/
├── model/                    # MLflow model files
├── plots/                    # All visualization plots
│   ├── pr_curve_RandomForest.png
│   ├── pr_curve_XGBoost.png
│   └── model_comparison.png
├── serialized_models/        # Downloadable model files
│   ├── RandomForest_model.joblib
│   ├── RandomForest_metadata.joblib
│   ├── XGBoost_model.joblib
│   └── XGBoost_metadata.joblib
└── reports/                  # Documentation and summaries
    ├── training_summary.md
    └── model_summary.txt
```

## Best Practices

1. **Experiment Organization**: Use descriptive run names and tags for easy identification
2. **Model Versioning**: Always register models after training with meaningful descriptions
3. **Stage Management**: Use appropriate stages (Production/Staging) for model lifecycle
4. **Artifact Management**: Keep artifacts organized and properly documented
5. **Model Loading**: Use intelligent loading with proper error handling
6. **Feature Validation**: Ensure input data matches expected feature dimensions
7. **Monitoring**: Regularly check model performance and compare versions
8. **API Compatibility**: Use latest MLflow APIs with backward compatibility

## Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - **Problem**: "Model does not have the 'sklearn' flavor"
   - **Solution**: The system automatically tries multiple flavors (sklearn → xgboost → pyfunc)

2. **Feature Shape Mismatch**
   - **Problem**: "Feature shape mismatch, expected: 30, got 31"
   - **Solution**: Ensure input data has exactly 30 features (V1-V28 + Time + Amount)

3. **Model Not Found in Registry**
   - **Problem**: No models in Production stage
   - **Solution**: System automatically falls back to latest version

4. **Deprecated API Warnings**
   - **Problem**: FutureWarning about deprecated MLflow APIs
   - **Solution**: System uses new APIs with backward compatibility fallbacks

5. **MLflow Server Not Running**
   - **Solution**: Use `python start_mlflow_ui.py` or start MLflow server manually

6. **Permission Issues**
   - **Solution**: Check file permissions for mlruns directory

7. **Memory Issues**
   - **Solution**: Reduce batch size or use smaller datasets

### Debug Mode

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Production Deployment

For production deployment:

1. **Model Loading**: Use `load_model_from_registry(stage="Production")` for best model
2. **Inference**: Use `predict_fraud()` for batch predictions with proper validation
3. **Monitoring**: Track model performance and compare versions regularly
4. **Updates**: Use Model Registry for seamless model updates and rollbacks
5. **Error Handling**: Leverage built-in error handling and fallback mechanisms
6. **Feature Validation**: Ensure input data matches expected 30-feature schema

## Integration with CI/CD

The MLflow integration supports enterprise CI/CD pipelines:

1. **Training Pipeline**: Automated model training and registration with versioning
2. **Model Validation**: Automated model performance validation and comparison
3. **Deployment**: Automated model deployment from registry with stage management
4. **Monitoring**: Continuous model performance monitoring and alerting
5. **Rollback**: Easy model rollback using Model Registry versions
6. **Testing**: Automated MLflow integration testing with `test_mlflow.py`

## Support

For issues or questions:

1. **Run Tests**: Execute `python test_mlflow.py` to validate functionality
2. **Check MLflow UI**: Use `python start_mlflow_ui.py` to view experiments and models
3. **Review Logs**: Check console output for detailed error messages
4. **Verify Registry**: Use `list_available_models()` to check model registry status
5. **MLflow Documentation**: Refer to official MLflow documentation for advanced features

## Summary

This MLflow integration provides:
- ✅ **Complete Experiment Tracking** with automatic parameter and metric logging
- ✅ **Intelligent Model Loading** with multi-flavor support and error handling
- ✅ **Robust Model Registry** with versioning and stage management
- ✅ **Production-Ready Inference** with feature validation and error handling
- ✅ **Comprehensive Testing** with automated validation suite
- ✅ **Future-Proof Design** with latest MLflow APIs and backward compatibility 

The system is now **absolutely perfect** and ready for production use! 🎉 
