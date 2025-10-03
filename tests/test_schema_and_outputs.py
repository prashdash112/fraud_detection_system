import numpy as np
import pandas as pd

from src.modelling.model_train import load_data_for_model
from src.modelling.mlflow_utils import load_model_from_registry, predict_fraud


def test_preprocessed_schema_columns_present():
    """Ensure preprocessed CSV has all expected 30 feature columns plus Class."""
    # Use the balanced train file shipped with the repo
    csv_path = "./src/data/creditcard_preprocessed_balanced_train.csv"
    df = pd.read_csv(csv_path)

    # Expected feature columns: V1..V28 + Time + Amount
    expected_features = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount"]
    missing = [c for c in expected_features if c not in df.columns]

    assert len(missing) == 0, f"Missing expected feature columns: {missing}"
    assert "Class" in df.columns, "Target column 'Class' is missing"


def test_model_inference_shapes():
    """Loaded model should produce probability scores and predictions of expected shapes."""
    model, version = load_model_from_registry(stage=None)  # pick latest available

    # If no model in registry yet, skip test gracefully
    if model is None:
        import pytest
        pytest.skip("No model available in MLflow registry yet")

    # Create dummy input with correct number of features
    X = np.random.randn(8, 30)
    preds, probs = predict_fraud(model, X, threshold=0.5)

    assert preds is not None and probs is not None, "Model did not return outputs"
    assert preds.shape == (8,), f"Predictions shape mismatch: {preds.shape}"
    assert probs.shape == (8,), f"Probabilities shape mismatch: {probs.shape}"


