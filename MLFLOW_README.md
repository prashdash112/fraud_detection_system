# MLflow Integration for Fraud Detection System

This document describes the MLflow integration for model versioning, experiment tracking, and model registry in the fraud detection system.

## Overview

The fraud detection system now includes comprehensive MLflow integration for:
- **Experiment Tracking**: Log all training runs, parameters, and metrics
- **Model Versioning**: Track different model versions and their performance
- **Model Registry**: Centralized model storage and management
- **Model Deployment**: Easy model loading for production inference

## Features

### 1. Experiment Tracking
- Automatic logging of all model parameters
- Comprehensive metrics tracking (accuracy, ROC-AUC, precision, recall, etc.)
- Artifact storage (models, plots, metadata)
- Nested runs for individual model training

### 2. Model Registry
- Centralized model storage
- Version control for models
- Model stage management (Production, Staging, etc.)
- Model metadata and descriptions

### 3. Model Loading & Inference
- Easy model loading from registry
- Production-ready inference functions
- Model comparison utilities
- Version-specific model access

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

## Usage

### 1. Training with MLflow Tracking

Run the main training pipeline with MLflow integration:

```bash
python main.py
```

This will:
- Train both RandomForest and XGBoost models
- Log all parameters and metrics to MLflow
- Register the best model in the Model Registry
- Save local model files for backup

### 2. Model Loading and Inference

```python
from src.modelling.mlflow_utils import *

# Load the best model from registry
model, version = load_model_from_registry(stage="Production")

# Make predictions
predictions, probabilities = predict_fraud(model, test_data, threshold=0.5)
```

### 3. Model Management

```python
# List available models
list_available_models()

# Compare model versions
compare_model_versions()

# Get model information
model_info = get_model_info(version)
```

## MLflow UI

Access the MLflow UI to view experiments and models:

```bash
mlflow ui
```

Then open http://localhost:5000 in your browser.

## File Structure

```
src/modelling/
├── model_train.py          # MLflow-integrated training pipeline
├── mlflow_utils.py         # MLflow utilities for model loading
└── ...

mlruns/                     # MLflow tracking data (auto-created)
├── 0/                      # Experiment runs
└── ...

models/                     # Local model storage
├── RandomForest_model.joblib
├── RandomForest_metadata.joblib
└── ...
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

Run the MLflow integration tests:

```bash
python test_mlflow.py
```

This will test:
- MLflow setup and configuration
- Model registry functionality
- Model loading and inference
- Error handling

## Key Functions

### Training Functions
- `run_training_pipeline()`: Main training pipeline with MLflow integration
- `train_random_forest()`: RandomForest training with parameter logging
- `train_xgboost()`: XGBoost training with parameter logging
- `evaluate_model()`: Model evaluation with metrics logging

### MLflow Utilities
- `setup_mlflow()`: Initialize MLflow experiment
- `log_model_to_mlflow()`: Log model to MLflow with artifacts
- `register_model_version()`: Register model in Model Registry
- `load_model_from_registry()`: Load model from registry
- `predict_fraud()`: Make fraud predictions
- `compare_model_versions()`: Compare different model versions

## Model Artifacts

Each model run logs the following artifacts:
- **Model**: Serialized model file
- **Metadata**: Model metadata and configuration
- **Precision-Recall Curve**: Visualization plot
- **Parameters**: All model hyperparameters
- **Metrics**: Performance metrics

## Best Practices

1. **Experiment Organization**: Use descriptive run names and tags
2. **Model Versioning**: Always register models after training
3. **Stage Management**: Use appropriate stages for model lifecycle
4. **Artifact Management**: Keep artifacts organized and documented
5. **Monitoring**: Regularly check model performance in production

## Troubleshooting

### Common Issues

1. **MLflow Server Not Running**
   - Solution: Start MLflow server or use local file storage

2. **Model Not Found in Registry**
   - Solution: Ensure model was registered after training

3. **Permission Issues**
   - Solution: Check file permissions for mlruns directory

4. **Memory Issues**
   - Solution: Reduce batch size or use smaller datasets

### Debug Mode

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Production Deployment

For production deployment:

1. **Model Loading**: Use `load_model_from_registry()` with Production stage
2. **Inference**: Use `predict_fraud()` for batch predictions
3. **Monitoring**: Track model performance and drift
4. **Updates**: Use Model Registry for model updates

## Integration with CI/CD

The MLflow integration supports CI/CD pipelines:

1. **Training Pipeline**: Automated model training and registration
2. **Model Validation**: Automated model performance validation
3. **Deployment**: Automated model deployment from registry
4. **Monitoring**: Continuous model performance monitoring

## Support

For issues or questions:
1. Check the MLflow documentation
2. Review the test script output
3. Check MLflow UI for run details
4. Verify model registry status
