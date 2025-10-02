"""
MLflow utilities for model loading and inference.
"""
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np

# MLflow configuration
MLFLOW_EXPERIMENT_NAME = "fraud_detection_experiments"
MLFLOW_MODEL_REGISTRY_NAME = "fraud_detection_model"

def load_model_from_registry(stage="Production"):
    """
    Load the best model from MLflow Model Registry.
    
    Args:
        stage (str): Model stage to load (Production, Staging, None for latest)
    
    Returns:
        model: Loaded MLflow model
        model_version: Version of the loaded model
    """
    try:
        client = MlflowClient()
        
        if stage:
            # Get model by stage
            model_versions = client.get_latest_versions(
                MLFLOW_MODEL_REGISTRY_NAME, 
                stages=[stage]
            )
            if not model_versions:
                print(f"No model found in {stage} stage. Trying latest version...")
                stage = None
        
        if not stage:
            # Get latest version
            model_versions = client.get_latest_versions(MLFLOW_MODEL_REGISTRY_NAME)
        
        if not model_versions:
            raise ValueError(f"No model found in registry: {MLFLOW_MODEL_REGISTRY_NAME}")
        
        latest_version = model_versions[0]
        model_uri = f"models:/{MLFLOW_MODEL_REGISTRY_NAME}/{latest_version.version}"
        
        print(f"Loading model version {latest_version.version} from stage: {latest_version.current_stage}")
        
        # Load model
        model = mlflow.sklearn.load_model(model_uri)
        
        return model, latest_version.version
        
    except Exception as e:
        print(f"Error loading model from registry: {e}")
        return None, None

def load_model_from_run(run_id):
    """
    Load model from a specific MLflow run.
    
    Args:
        run_id (str): MLflow run ID
    
    Returns:
        model: Loaded MLflow model
    """
    try:
        model_uri = f"runs:/{run_id}/model"
        model = mlflow.sklearn.load_model(model_uri)
        return model
    except Exception as e:
        print(f"Error loading model from run {run_id}: {e}")
        return None

def predict_fraud(model, data, threshold=0.5):
    """
    Make fraud predictions using the loaded model.
    
    Args:
        model: Loaded MLflow model
        data: Input data (numpy array or pandas DataFrame)
        threshold (float): Probability threshold for fraud classification
    
    Returns:
        predictions: Binary predictions (0 or 1)
        probabilities: Fraud probabilities
    """
    try:
        # Get probabilities
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(data)[:, 1]
        else:
            probabilities = model.decision_function(data)
            # Normalize to 0-1 range
            probabilities = (probabilities - probabilities.min()) / (probabilities.max() - probabilities.min() + 1e-8)
        
        # Make binary predictions
        predictions = (probabilities >= threshold).astype(int)
        
        return predictions, probabilities
        
    except Exception as e:
        print(f"Error making predictions: {e}")
        return None, None

def get_model_info(model_version):
    """
    Get information about the loaded model.
    
    Args:
        model_version: Model version object from MLflow
    
    Returns:
        dict: Model information
    """
    try:
        client = MlflowClient()
        
        # Get model details
        model_info = {
            "model_name": MLFLOW_MODEL_REGISTRY_NAME,
            "version": model_version.version,
            "stage": model_version.current_stage,
            "description": model_version.description,
            "creation_timestamp": model_version.creation_timestamp,
            "last_updated_timestamp": model_version.last_updated_timestamp
        }
        
        return model_info
        
    except Exception as e:
        print(f"Error getting model info: {e}")
        return None

def list_available_models():
    """
    List all available models in the registry.
    
    Returns:
        list: List of model versions
    """
    try:
        client = MlflowClient()
        model_versions = client.get_latest_versions(MLFLOW_MODEL_REGISTRY_NAME)
        
        print(f"Available models in {MLFLOW_MODEL_REGISTRY_NAME}:")
        for version in model_versions:
            print(f"  Version {version.version}: {version.current_stage}")
            print(f"    Description: {version.description}")
            print(f"    Created: {version.creation_timestamp}")
            print()
        
        return model_versions
        
    except Exception as e:
        print(f"Error listing models: {e}")
        return []

def compare_model_versions():
    """
    Compare different model versions and their metrics.
    
    Returns:
        pandas.DataFrame: Comparison of model versions
    """
    try:
        client = MlflowClient()
        model_versions = client.get_latest_versions(MLFLOW_MODEL_REGISTRY_NAME)
        
        comparison_data = []
        for version in model_versions:
            try:
                # Get run details
                run = client.get_run(version.run_id)
                
                metrics = run.data.metrics
                comparison_data.append({
                    "version": version.version,
                    "stage": version.current_stage,
                    "accuracy": metrics.get("accuracy", "N/A"),
                    "roc_auc": metrics.get("roc_auc", "N/A"),
                    "average_precision": metrics.get("average_precision", "N/A"),
                    "f1_score": metrics.get("f1_score", "N/A"),
                    "run_id": version.run_id
                })
            except Exception as e:
                print(f"Error getting metrics for version {version.version}: {e}")
        
        if comparison_data:
            df = pd.DataFrame(comparison_data)
            print("Model Version Comparison:")
            print(df.to_string(index=False))
            return df
        else:
            print("No model versions found for comparison")
            return None
            
    except Exception as e:
        print(f"Error comparing model versions: {e}")
        return None
