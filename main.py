import numpy as np
from src.preprocessing.preprocessing import *
from src.preprocessing.handle_imbalance import *
from src.modelling.model_train import *
from src.modelling.mlflow_utils import * 

def run(steps=None):
    """
    steps: list of str, e.g. ["step1", "step2", "step3", "step4"]
    step1: data loading and preprocessing
    step2: balancing data and train/test split
    step3: model training and MLflow logging
    step4: MLflow model registry demo and inference
    If steps is None, all steps are run.
    """
    if steps is None:
        steps = ["step1", "step2", "step3", "step4"]

    # --- Step 1: Data loading and preprocessing ---
    if "step1" in steps:
        INPUT_FILE = "./src/data/creditcard.csv" 
        OUTPUT_FILE = "./src/data/creditcard_preprocessed.csv"
            
        processed_df = preprocess_pipeline(
            input_path=INPUT_FILE,
            output_path=OUTPUT_FILE,
            missing_strategy='mean',
            scaling_method='standard'
        )

        print(f"\nProcessed data shape: {processed_df.shape}")
        print(f"\nFirst few rows:\n{processed_df.head()}")
        print(f"\nData statistics:\n{processed_df.describe()}")

    # --- Step 2: Balancing data and train/test split ---
    if "step2" in steps:
        INPUT_preprocessed_FILE = "./src/data/creditcard_preprocessed.csv"
        TRAIN_balanced_OUTPUT = "./src/data/creditcard_preprocessed_balanced_train.csv"
        TEST_balanced_OUTPUT = "./src/data/creditcard_preprocessed_balanced_test.csv"
        
        print("\n" + "="*60)
        print("FRAUD DETECTION: CLASS IMBALANCE HANDLING")
        print("="*60)
        
        # Load data
        X, y = load_imbalanced_data(INPUT_preprocessed_FILE)
        show_distribution(y, "Original Data")
        
        # Split and balance
        X_train, X_test, y_train, y_test = split_and_balance(X, y, test_size=0.2)
        
        # Save processed data
        feature_cols = [f'V{i}' for i in range(1, 29)] + ['Time', 'Amount']
        save_balanced_data(X_train, y_train, TRAIN_balanced_OUTPUT, feature_cols)
        save_balanced_data(X_test, y_test, TEST_balanced_OUTPUT, feature_cols)
        
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Training samples: {len(y_train):,}")
        print(f"Test samples: {len(y_test):,}")
        print(f"\n✓ Ready for model training!")
        print("="*60)

    # --- Step 3: Modelling and MLflow logging ---
    if "step3" in steps:
        # File paths (can be overridden)
        MODEL_TRAIN_FILE = "./src/data/creditcard_preprocessed_balanced_train.csv"
        MODEL_TEST_FILE = "./src/data/creditcard_preprocessed_balanced_test.csv"
        
        print("\n" + "="*60)
        print("FRAUD DETECTION: MLflow Model Training & Registry")
        print("="*60)
        
        # Train models with MLflow tracking
        models, metrics = run_training_pipeline(
            MODEL_TRAIN_FILE, 
            MODEL_TEST_FILE,
            train_rf=True,
            train_xgb=True,
            threshold=0.5, 
            k=100,
            serialize_best=True,
            output_dir="models"
        )

    # --- Step 4: MLflow Model Registry Demo and Inference ---
    if "step4" in steps:
        MODEL_TEST_FILE = "./src/data/creditcard_preprocessed_balanced_test.csv"
        print("\n" + "="*60)
        print("MLflow Model Registry Demo")
        print("="*60)
        
        # Demonstrate MLflow model loading
        print("\n1. Loading best model from MLflow Model Registry...")
        model, version = load_model_from_registry(stage="Production")
        
        if model is not None:
            print(f"✓ Successfully loaded model version {version}")
            
            # Get model information
            model_info = get_model_info(version)
            if model_info:
                print(f"Model Info: {model_info}")
            
            # Load test data for inference demo
            print("\n2. Loading test data for inference demo...")
            X_test, y_test = load_data_for_model(MODEL_TEST_FILE)
            
            # Make predictions
            print("\n3. Making predictions on test data...")
            predictions, probabilities = predict_fraud(model, X_test[:100], threshold=0.5)
            
            if predictions is not None:
                print(f"✓ Made predictions on {len(predictions)} samples")
                print(f"Fraud predictions: {np.sum(predictions)} out of {len(predictions)}")
                print(f"Average fraud probability: {np.mean(probabilities):.4f}")
            
            # Compare model versions
            print("\n4. Comparing available model versions...")
            compare_model_versions()
            
        else:
            print("✗ Failed to load model from registry")
        
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print("✓ Models trained and logged to MLflow")
        print("✓ Best model registered in Model Registry")
        print("✓ Model versioning and tracking enabled")
        print("✓ Ready for production inference!")
        print("="*60)


if __name__ == "__main__":
    # Example usage:
    # run(["step1"])  # Only preprocessing
    # run(["step1", "step2"])  # Preprocessing + balancing
    # run(["step1", "step2", "step3"])  # Up to model training
    # run(["step1", "step2", "step3", "step4"])  # Full pipeline
    run(["step1", "step2", "step3", "step4"])
