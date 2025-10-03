
# Fraud Detection Model Training Summary

## Dataset Information
- Training File: ./src/data/creditcard_preprocessed_balanced_train.csv
- Test File: ./src/data/creditcard_preprocessed_balanced_test.csv
- Threshold: 0.5
- K (for Precision@K): 100

## Model Performance Comparison


### RandomForest Results
- **Accuracy**: 0.9995
- **ROC AUC**: 0.9609
- **Average Precision**: 0.8183
- **F1 Score**: 0.8208
- **Precision@100**: 0.77
- **Recall@100**: 0.8105263157894737
- **Threshold**: 0.5


### XGBoost Results
- **Accuracy**: 0.9992
- **ROC AUC**: 0.9687
- **Average Precision**: 0.8080
- **F1 Score**: 0.7600
- **Precision@100**: 0.76
- **Recall@100**: 0.8
- **Threshold**: 0.5


## Best Model Selection
- **Selected Model**: RandomForest
- **Selection Criteria**: Average Precision (AP)
- **Best AP Score**: 0.8183

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

Generated on: 2025-10-03 11:33:46
