import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, roc_auc_score,
    precision_recall_curve, f1_score, average_precision_score
)
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import joblib
import os
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

# MLflow configuration
MLFLOW_EXPERIMENT_NAME = "fraud_detection_experiments"
MLFLOW_MODEL_REGISTRY_NAME = "fraud_detection_model"

def setup_mlflow():
    """Setup MLflow experiment and return client."""
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    client = MlflowClient()
    return client

def load_data_for_model(filepath):
    """Load features and target from a CSV file."""
    df = pd.read_csv(filepath)
    X = df.drop('Class', axis=1).values
    y = df['Class'].values
    return X, y

def train_random_forest(X_train, y_train, **kwargs):
    """Train a RandomForestClassifier and return the fitted model."""
    # Set default parameters
    params = {
        'n_estimators': 100,
        'random_state': 42,
        'n_jobs': -1,
        **kwargs
    }
    
    # Log parameters to MLflow
    mlflow.log_params(params)
    
    rf = RandomForestClassifier(**params)
    rf.fit(X_train, y_train)
    return rf

def train_xgboost(X_train, y_train, **kwargs):
    """Train an XGBClassifier and return the fitted model."""
    # Set default parameters
    params = {
        'n_estimators': 100,
        'use_label_encoder': False,
        'eval_metric': 'logloss',
        'random_state': 42,
        'n_jobs': -1,
        **kwargs
    }
    
    # Log parameters to MLflow
    mlflow.log_params(params)
    
    xgb = XGBClassifier(**params)
    xgb.fit(X_train, y_train)
    return xgb

def precision_recall_at_k(y_true, y_scores, k=100):
    """Compute Precision@K and Recall@K for the top K highest probability predictions."""
    # Sort by predicted probability
    idx = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[idx]
    top_k = y_true_sorted[:k]
    precision_at_k = np.sum(top_k) / k
    recall_at_k = np.sum(top_k) / np.sum(y_true)
    return precision_at_k, recall_at_k

def plot_precision_recall_curve(y_true, y_scores, model_name="Model"):
    """Plot and save the precision-recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)
    plt.figure()
    plt.plot(recall, precision, label=f'AP={ap:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve: {model_name}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'pr_curve_{model_name}.png')
    plt.close()

def evaluate_model(
    model, X_test, y_test, model_name="Model", threshold=0.5, k=100, plot_pr_curve=True
):
    """
    Evaluate a classifier with fraud detection metrics.
    - threshold: probability threshold for positive class
    - k: for Precision@K and Recall@K
    """
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = model.decision_function(X_test)
        y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min() + 1e-8)

    # Documented threshold choice
    print(f"\n{model_name} - Using probability threshold = {threshold:.2f}")
    y_pred = (y_proba >= threshold).astype(int)

    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    ap = average_precision_score(y_test, y_proba)
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    precision_at_k, recall_at_k = precision_recall_at_k(y_test, y_proba, k=k)

    print(f"\n{model_name} Results:")
    print("Accuracy:", f"{acc:.4f}")
    print("ROC AUC:", f"{roc_auc:.4f}")
    print("Average Precision (AP):", f"{ap:.4f}")
    print("F1-score:", f"{f1:.4f}")
    print(f"Precision@{k}: {precision_at_k:.4f}")
    print(f"Recall@{k}: {recall_at_k:.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred, digits=4))

    if plot_pr_curve:
        plot_precision_recall_curve(y_test, y_proba, model_name=model_name)

    # Log metrics to MLflow
    mlflow.log_metrics({
        "accuracy": acc,
        "roc_auc": roc_auc,
        "average_precision": ap,
        "f1_score": f1,
        f"precision_at_{k}": precision_at_k,
        f"recall_at_{k}": recall_at_k,
        "threshold": threshold
    })

    # Return metrics for comparison
    return {
        "model_name": model_name,
        "accuracy": acc,
        "roc_auc": roc_auc,
        "average_precision": ap,
        "f1": f1,
        f"precision_at_{k}": precision_at_k,
        f"recall_at_{k}": recall_at_k,
        "threshold": threshold
    }

def log_model_to_mlflow(model, model_name, X_test, y_test, metrics, metadata):
    """Log model to MLflow with signature and metadata."""
    # Infer model signature
    signature = infer_signature(X_test, model.predict(X_test))
    
    # Log model based on type
    if model_name == "RandomForest":
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=X_test[:5],
            registered_model_name=MLFLOW_MODEL_REGISTRY_NAME
        )
    elif model_name == "XGBoost":
        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="model",
            signature=signature,
            input_example=X_test[:5],
            registered_model_name=MLFLOW_MODEL_REGISTRY_NAME
        )
    
    # Log additional artifacts
    mlflow.log_dict(metadata, "model_metadata.json")
    
    # Log precision-recall curve
    if os.path.exists(f'pr_curve_{model_name}.png'):
        mlflow.log_artifact(f'pr_curve_{model_name}.png')

def register_model_version(client, model_name, metrics):
    """Register model version in MLflow Model Registry."""
    try:
        # Get the latest version of the model
        latest_version = client.get_latest_versions(MLFLOW_MODEL_REGISTRY_NAME)[0]
        
        # Add model description
        client.update_model_version(
            name=MLFLOW_MODEL_REGISTRY_NAME,
            version=latest_version.version,
            description=f"Best {model_name} model with AP: {metrics['average_precision']:.4f}"
        )
        
        # Transition to Production if it's the best model
        if metrics['average_precision'] > 0.8:  # Adjust threshold as needed
            client.transition_model_version_stage(
                name=MLFLOW_MODEL_REGISTRY_NAME,
                version=latest_version.version,
                stage="Production"
            )
            print(f"Model {model_name} transitioned to Production stage")
        
        return latest_version.version
    except Exception as e:
        print(f"Error registering model: {e}")
        return None

def serialize_model(model, model_name, metadata, output_dir="models"):
    """Serialize the model, metadata, and save to disk."""
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"{model_name}_model.joblib")
    meta_path = os.path.join(output_dir, f"{model_name}_metadata.joblib")
    joblib.dump(model, model_path)
    joblib.dump(metadata, meta_path)
    print(f"Serialized {model_name} model to {model_path}")
    print(f"Serialized {model_name} metadata to {meta_path}")

def run_training_pipeline(
    train_file, 
    test_file,
    train_rf=True,
    train_xgb=True,
    threshold=0.5, 
    k=100,
    serialize_best=True,
    output_dir="models"
):
    """Load data, train models, evaluate, and serialize the best one with MLflow tracking."""
    # Setup MLflow
    client = setup_mlflow()
    
    # Start MLflow run
    with mlflow.start_run():
        # Log run parameters
        mlflow.log_params({
            "train_file": train_file,
            "test_file": test_file,
            "threshold": threshold,
            "k": k,
            "train_rf": train_rf,
            "train_xgb": train_xgb
        })
        
        X_train, y_train = load_data_for_model(train_file)
        X_test, y_test = load_data_for_model(test_file)
        print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

        models = {}
        metrics = []

        if train_rf:
            print("\n=== Training RandomForestClassifier ===")
            with mlflow.start_run(nested=True, run_name="RandomForest"):
                rf = train_random_forest(X_train, y_train)
                rf_metrics = evaluate_model(
                    rf, X_test, y_test, model_name="RandomForest", threshold=threshold, k=k
                )
                models['random_forest'] = rf
                metrics.append(rf_metrics)
                
                # Log model to MLflow
                metadata = {
                    "metrics": rf_metrics,
                    "feature_names": [f'V{i}' for i in range(1, 29)] + ['Time', 'Amount'],
                    "threshold": threshold,
                    "k": k,
                    "train_file": train_file,
                    "test_file": test_file
                }
                log_model_to_mlflow(rf, "RandomForest", X_test, y_test, rf_metrics, metadata)

        if train_xgb:
            print("\n=== Training XGBClassifier ===")
            with mlflow.start_run(nested=True, run_name="XGBoost"):
                xgb = train_xgboost(X_train, y_train)
                xgb_metrics = evaluate_model(
                    xgb, X_test, y_test, model_name="XGBoost", threshold=threshold, k=k
                )
                models['xgboost'] = xgb
                metrics.append(xgb_metrics)
                
                # Log model to MLflow
                metadata = {
                    "metrics": xgb_metrics,
                    "feature_names": [f'V{i}' for i in range(1, 29)] + ['Time', 'Amount'],
                    "threshold": threshold,
                    "k": k,
                    "train_file": train_file,
                    "test_file": test_file
                }
                log_model_to_mlflow(xgb, "XGBoost", X_test, y_test, xgb_metrics, metadata)

        # Select best model by average precision (AP)
        best = max(metrics, key=lambda m: m['average_precision'])
        best_name = best['model_name']
        print(f"\nBest model by Average Precision: {best_name}")

        # Log best model metrics to main run
        mlflow.log_metrics({
            "best_model_accuracy": best['accuracy'],
            "best_model_roc_auc": best['roc_auc'],
            "best_model_average_precision": best['average_precision'],
            "best_model_f1": best['f1'],
            f"best_model_precision_at_{k}": best[f'precision_at_{k}'],
            f"best_model_recall_at_{k}": best[f'recall_at_{k}']
        })
        
        # Register best model version
        model_key_map = {
            "RandomForest": "random_forest",
            "XGBoost": "xgboost"
        }
        model_key = model_key_map[best_name]
        version = register_model_version(client, best_name, best)
        
        # Serialize best model locally
        if serialize_best:
            metadata = {
                "metrics": best,
                "feature_names": [f'V{i}' for i in range(1, 29)] + ['Time', 'Amount'],
                "threshold": threshold,
                "k": k,
                "train_file": train_file,
                "test_file": test_file,
                "mlflow_version": version
            }
            serialize_model(models[model_key], best_name, metadata, output_dir=output_dir)

        return models, metrics




