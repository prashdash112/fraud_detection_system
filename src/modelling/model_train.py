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
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'AP={ap:.3f}', linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve: {model_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'pr_curve_{model_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_model_comparison(metrics_list, output_dir="models"):
    """Create a comprehensive model comparison plot."""
    if len(metrics_list) < 2:
        return
    
    model_names = [m['model_name'] for m in metrics_list]
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    
    # Metrics to compare
    metric_names = ['accuracy', 'roc_auc', 'average_precision', 'f1']
    metric_labels = ['Accuracy', 'ROC AUC', 'Average Precision', 'F1 Score']
    
    for i, (metric, label) in enumerate(zip(metric_names, metric_labels)):
        ax = axes[i//2, i%2]
        values = [m[metric] for m in metrics_list]
        
        bars = ax.bar(model_names, values, alpha=0.7, color=['skyblue', 'lightcoral'])
        ax.set_title(f'{label} Comparison')
        ax.set_ylabel(label)
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Rotate x-axis labels
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    comparison_path = os.path.join(output_dir, 'model_comparison.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return comparison_path

def create_training_summary(metrics, best_model, train_file, test_file, threshold, k):
    """Create a comprehensive training summary report."""
    summary = f"""
# Fraud Detection Model Training Summary

## Dataset Information
- Training File: {train_file}
- Test File: {test_file}
- Threshold: {threshold}
- K (for Precision@K): {k}

## Model Performance Comparison

"""
    
    for i, metric in enumerate(metrics, 1):
        summary += f"""
### {metric['model_name']} Results
- **Accuracy**: {metric['accuracy']:.4f}
- **ROC AUC**: {metric['roc_auc']:.4f}
- **Average Precision**: {metric['average_precision']:.4f}
- **F1 Score**: {metric['f1']:.4f}
- **Precision@{k}**: {metric.get(f'precision_at_{k}', 'N/A')}
- **Recall@{k}**: {metric.get(f'recall_at_{k}', 'N/A')}
- **Threshold**: {metric.get('threshold', 'N/A')}

"""
    
    summary += f"""
## Best Model Selection
- **Selected Model**: {best_model['model_name']}
- **Selection Criteria**: Average Precision (AP)
- **Best AP Score**: {best_model['average_precision']:.4f}

## Model Artifacts Available
1. **Serialized Models**: `.joblib` files for both RandomForest and XGBoost
2. **Model Metadata**: JSON files with training configuration
3. **Precision-Recall Curves**: PNG plots for each model
4. **Model Comparison**: Side-by-side performance comparison plot
5. **Model Summary**: Text summary with key metrics

## Usage Instructions
- Load models using MLflow Model Registry
- Use `load_model_from_registry()` for production inference
- Access all artifacts through MLflow UI
- Download serialized models for offline use

Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    return summary

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

def log_model_to_mlflow(model, model_name, X_test, y_test, metrics, metadata, output_dir="models"):
    """Log model to MLflow with signature, metadata, and artifacts."""
    # Infer model signature
    signature = infer_signature(X_test, model.predict(X_test))
    
    # Log model based on type
    if model_name == "RandomForest":
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            input_example=X_test[:5],
            registered_model_name=MLFLOW_MODEL_REGISTRY_NAME
        )
    elif model_name == "XGBoost":
        mlflow.xgboost.log_model(
            xgb_model=model,
            name="model",
            signature=signature,
            input_example=X_test[:5],
            registered_model_name=MLFLOW_MODEL_REGISTRY_NAME
        )
    
    # Log additional artifacts
    mlflow.log_dict(metadata, "model_metadata.json")
    
    # Log precision-recall curve
    pr_curve_path = f'pr_curve_{model_name}.png'
    if os.path.exists(pr_curve_path):
        mlflow.log_artifact(pr_curve_path, "plots")
        print(f"✓ Logged PR curve: {pr_curve_path}")
    
    # Log serialized model files (joblib format)
    model_path = os.path.join(output_dir, f"{model_name}_model.joblib")
    meta_path = os.path.join(output_dir, f"{model_name}_metadata.joblib")
    
    if os.path.exists(model_path):
        mlflow.log_artifact(model_path, "serialized_models")
        print(f"✓ Logged serialized model: {model_path}")
    
    if os.path.exists(meta_path):
        mlflow.log_artifact(meta_path, "serialized_models")
        print(f"✓ Logged model metadata: {meta_path}")
    
    # Log model summary as text
    model_summary = f"""
Model: {model_name}
Accuracy: {metrics['accuracy']:.4f}
ROC AUC: {metrics['roc_auc']:.4f}
Average Precision: {metrics['average_precision']:.4f}
F1 Score: {metrics['f1']:.4f}
Precision@100: {metrics.get('precision_at_100', 'N/A')}
Recall@100: {metrics.get('recall_at_100', 'N/A')}
Threshold: {metrics.get('threshold', 'N/A')}
    """
    mlflow.log_text(model_summary, "model_summary.txt")

def register_model_version(client, model_name, metrics):
    """Register model version in MLflow Model Registry."""
    try:
        # Get the latest version of the model
        try:
            # Use new API with fallback
            model_versions = client.search_model_versions(f"name='{MLFLOW_MODEL_REGISTRY_NAME}'")
            if not model_versions:
                print("No model versions found in registry")
                return None
            latest_version = model_versions[0]
        except:
            # Fallback to deprecated method
            latest_version = client.get_latest_versions(MLFLOW_MODEL_REGISTRY_NAME)[0]
        
        # Add model description
        try:
            client.update_model_version(
                name=MLFLOW_MODEL_REGISTRY_NAME,
                version=latest_version.version,
                description=f"Best {model_name} model with AP: {metrics['average_precision']:.4f}"
            )
        except Exception as e:
            print(f"Warning: Could not update model description: {e}")
        
        # Transition to Production if it's the best model
        if metrics['average_precision'] > 0.8:  # Adjust threshold as needed
            try:
                client.transition_model_version_stage(
                    name=MLFLOW_MODEL_REGISTRY_NAME,
                    version=latest_version.version,
                    stage="Production"
                )
                print(f"Model {model_name} transitioned to Production stage")
            except Exception as e:
                print(f"Warning: Could not transition model to Production: {e}")
        
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
                
                # Serialize model locally first
                rf_metadata = {
                    "metrics": rf_metrics,
                    "feature_names": [f'V{i}' for i in range(1, 29)] + ['Time', 'Amount'],
                    "threshold": threshold,
                    "k": k,
                    "train_file": train_file,
                    "test_file": test_file
                }
                serialize_model(rf, "RandomForest", rf_metadata, output_dir=output_dir)
                
                # Log model to MLflow with all artifacts
                log_model_to_mlflow(rf, "RandomForest", X_test, y_test, rf_metrics, rf_metadata, output_dir)

        if train_xgb:
            print("\n=== Training XGBClassifier ===")
            with mlflow.start_run(nested=True, run_name="XGBoost"):
                xgb = train_xgboost(X_train, y_train)
                xgb_metrics = evaluate_model(
                    xgb, X_test, y_test, model_name="XGBoost", threshold=threshold, k=k
                )
                models['xgboost'] = xgb
                metrics.append(xgb_metrics)
                
                # Serialize model locally first
                xgb_metadata = {
                    "metrics": xgb_metrics,
                    "feature_names": [f'V{i}' for i in range(1, 29)] + ['Time', 'Amount'],
                    "threshold": threshold,
                    "k": k,
                    "train_file": train_file,
                    "test_file": test_file
                }
                serialize_model(xgb, "XGBoost", xgb_metadata, output_dir=output_dir)
                
                # Log model to MLflow with all artifacts
                log_model_to_mlflow(xgb, "XGBoost", X_test, y_test, xgb_metrics, xgb_metadata, output_dir)

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
        
        # Create and log model comparison plot
        if len(metrics) > 1:
            comparison_path = plot_model_comparison(metrics, output_dir)
            if comparison_path and os.path.exists(comparison_path):
                mlflow.log_artifact(comparison_path, "plots")
                print(f"✓ Logged model comparison plot: {comparison_path}")
        
        # Create and log training summary
        training_summary = create_training_summary(metrics, best, train_file, test_file, threshold, k)
        summary_path = os.path.join(output_dir, 'training_summary.md')
        with open(summary_path, 'w') as f:
            f.write(training_summary)
        mlflow.log_artifact(summary_path, "reports")
        print(f"✓ Logged training summary: {summary_path}")
        
        # Register best model version
        model_key_map = {
            "RandomForest": "random_forest",
            "XGBoost": "xgboost"
        }
        model_key = model_key_map[best_name]
        version = register_model_version(client, best_name, best)
        
        # Note: Models are already serialized and logged to MLflow above
        print(f"✓ All models serialized and logged to MLflow")
        print(f"✓ Best model: {best_name} (version {version})")

        return models, metrics




