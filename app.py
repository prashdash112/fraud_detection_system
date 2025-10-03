"""
FastAPI application for fraud detection model serving.
"""
import os
import joblib
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Fraud Detection API",
    description="A REST API for fraud detection using machine learning models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global variables for model and metadata
model = None
model_metadata = None
model_name = None

class TransactionData(BaseModel):
    """Pydantic model for transaction data input validation."""
    V1: float = Field(..., description="PCA component V1")
    V2: float = Field(..., description="PCA component V2")
    V3: float = Field(..., description="PCA component V3")
    V4: float = Field(..., description="PCA component V4")
    V5: float = Field(..., description="PCA component V5")
    V6: float = Field(..., description="PCA component V6")
    V7: float = Field(..., description="PCA component V7")
    V8: float = Field(..., description="PCA component V8")
    V9: float = Field(..., description="PCA component V9")
    V10: float = Field(..., description="PCA component V10")
    V11: float = Field(..., description="PCA component V11")
    V12: float = Field(..., description="PCA component V12")
    V13: float = Field(..., description="PCA component V13")
    V14: float = Field(..., description="PCA component V14")
    V15: float = Field(..., description="PCA component V15")
    V16: float = Field(..., description="PCA component V16")
    V17: float = Field(..., description="PCA component V17")
    V18: float = Field(..., description="PCA component V18")
    V19: float = Field(..., description="PCA component V19")
    V20: float = Field(..., description="PCA component V20")
    V21: float = Field(..., description="PCA component V21")
    V22: float = Field(..., description="PCA component V22")
    V23: float = Field(..., description="PCA component V23")
    V24: float = Field(..., description="PCA component V24")
    V25: float = Field(..., description="PCA component V25")
    V26: float = Field(..., description="PCA component V26")
    V27: float = Field(..., description="PCA component V27")
    V28: float = Field(..., description="PCA component V28")
    Time: float = Field(..., description="Time in seconds between current and first transaction")
    Amount: float = Field(..., description="Transaction amount")

    class Config:
        schema_extra = {
            "example": {
                "V1": -0.7059898246110177,
                "V2": 0.6277668093643811,
                "V3": -0.035994995232166,
                "V4": 0.1806427850874308,
                "V5": 0.4599348239833234,
                "V6": -0.036283158251373,
                "V7": 0.2802046719288935,
                "V8": -0.1841152764576969,
                "V9": 0.0685241005919484,
                "V10": 0.5863629005107058,
                "V11": -0.25233334795008,
                "V12": -1.2299078418984513,
                "V13": 0.4682882741114543,
                "V14": 0.4017355215141967,
                "V15": -0.3078030347127327,
                "V16": -0.1123814085906342,
                "V17": -0.4589679556521681,
                "V18": 0.0405522364190535,
                "V19": -0.9375302972907276,
                "V20": 0.1741002550832633,
                "V21": -0.1256561406066695,
                "V22": -0.1784533927889745,
                "V23": -0.1156088530642112,
                "V24": -0.2434813742463694,
                "V25": -1.156796820313679,
                "V26": 1.148810949147973,
                "V27": 1.0191007119749338,
                "V28": 0.0030985451139533,
                "Time": 0.0037648659393779,
                "Amount": -0.307400143722893
            }
        }

class PredictionResponse(BaseModel):
    """Pydantic model for prediction response."""
    fraud_probability: float = Field(..., description="Probability of fraud (0-1)")
    is_fraud: bool = Field(..., description="Binary fraud prediction")
    threshold: float = Field(..., description="Threshold used for binary prediction")
    model_name: str = Field(..., description="Name of the model used for prediction")

class HealthResponse(BaseModel):
    """Pydantic model for health check response."""
    status: str
    model_loaded: bool
    model_name: Optional[str] = None
    model_accuracy: Optional[float] = None

def load_model():
    """Load the best model and metadata from the models directory."""
    global model, model_metadata, model_name

    try:
        # Try to load RandomForest model first (now preferred)
        rf_model_path = "models/RandomForest_model.joblib"
        rf_metadata_path = "models/RandomForest_metadata.joblib"

        if os.path.exists(rf_model_path) and os.path.exists(rf_metadata_path):
            model = joblib.load(rf_model_path)
            model_metadata = joblib.load(rf_metadata_path)
            model_name = "RandomForest"
            logger.info("✓ Loaded RandomForest model")
        else:
            # Fallback to XGBoost
            xgb_model_path = "models/XGBoost_model.joblib"
            xgb_metadata_path = "models/XGBoost_metadata.joblib"

            if os.path.exists(xgb_model_path) and os.path.exists(xgb_metadata_path):
                model = joblib.load(xgb_model_path)
                model_metadata = joblib.load(xgb_metadata_path)
                model_name = "XGBoost"
                logger.info("✓ Loaded XGBoost model")
            else:
                raise FileNotFoundError("No model files found in models/ directory")

        logger.info(f"Model loaded successfully: {model_name}")
        logger.info(f"Model metadata: {model_metadata}")

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise e

@app.on_event("startup")
async def startup_event():
    """Load model on application startup."""
    logger.info("Starting up Fraud Detection API...")
    load_model()
    logger.info("API startup complete!")

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Fraud Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    if model is None:
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            model_name=None,
            model_accuracy=None
        )
    
    # Get model accuracy from metadata if available
    accuracy = None
    if model_metadata and "metrics" in model_metadata:
        accuracy = model_metadata["metrics"].get("accuracy")
    
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        model_name=model_name,
        model_accuracy=accuracy
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_fraud(
    transaction: TransactionData,
    threshold: float = 0.5
):
    """
    Predict fraud probability for a single transaction.
    
    Args:
        transaction: Transaction data
        threshold: Probability threshold for fraud classification (default: 0.5)
    
    Returns:
        PredictionResponse with fraud probability and binary prediction
    """
    if model is None:
        raise HTTPException(
            status_code=500,
            detail="Model not loaded. Please check the health endpoint."
        )
    
    try:
        # Convert transaction data to numpy array
        transaction_dict = transaction.dict()
        feature_values = [transaction_dict[f"V{i}"] for i in range(1, 29)]
        feature_values.extend([transaction_dict["Time"], transaction_dict["Amount"]])
        
        # Reshape for single prediction
        X = np.array(feature_values).reshape(1, -1)
        
        # Get fraud probability
        if hasattr(model, "predict_proba"):
            fraud_probability = model.predict_proba(X)[0, 1]
        else:
            # For models without predict_proba, use decision_function
            decision_score = model.decision_function(X)[0]
            # Normalize to 0-1 range (sigmoid transformation)
            fraud_probability = 1 / (1 + np.exp(-decision_score))
        
        # Make binary prediction
        is_fraud = fraud_probability >= threshold
        
        return PredictionResponse(
            fraud_probability=float(fraud_probability),
            is_fraud=bool(is_fraud),
            threshold=threshold,
            model_name=model_name
        )
        
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error making prediction: {str(e)}"
        )

@app.post("/predict_batch", response_model=List[PredictionResponse])
async def predict_fraud_batch(
    transactions: List[TransactionData],
    threshold: float = 0.5
):
    """
    Predict fraud probability for multiple transactions.
    
    Args:
        transactions: List of transaction data
        threshold: Probability threshold for fraud classification (default: 0.5)
    
    Returns:
        List of PredictionResponse objects
    """
    if model is None:
        raise HTTPException(
            status_code=500,
            detail="Model not loaded. Please check the health endpoint."
        )
    
    if len(transactions) > 1000:  # Limit batch size
        raise HTTPException(
            status_code=400,
            detail="Batch size too large. Maximum 1000 transactions per batch."
        )
    
    try:
        # Convert transactions to numpy array
        X_list = []
        for transaction in transactions:
            transaction_dict = transaction.dict()
            feature_values = [transaction_dict[f"V{i}"] for i in range(1, 29)]
            feature_values.extend([transaction_dict["Time"], transaction_dict["Amount"]])
            X_list.append(feature_values)
        
        X = np.array(X_list)
        
        # Get fraud probabilities
        if hasattr(model, "predict_proba"):
            fraud_probabilities = model.predict_proba(X)[:, 1]
        else:
            # For models without predict_proba, use decision_function
            decision_scores = model.decision_function(X)
            # Normalize to 0-1 range (sigmoid transformation)
            fraud_probabilities = 1 / (1 + np.exp(-decision_scores))
        
        # Create responses
        responses = []
        for prob in fraud_probabilities:
            is_fraud = prob >= threshold
            responses.append(PredictionResponse(
                fraud_probability=float(prob),
                is_fraud=bool(is_fraud),
                threshold=threshold,
                model_name=model_name
            ))
        
        return responses
        
    except Exception as e:
        logger.error(f"Error making batch predictions: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error making batch predictions: {str(e)}"
        )

@app.get("/model_info")
async def get_model_info():
    """Get information about the loaded model."""
    if model is None:
        raise HTTPException(
            status_code=500,
            detail="Model not loaded. Please check the health endpoint."
        )
    
    info = {
        "model_name": model_name,
        "model_type": type(model).__name__,
        "features": 30,  # V1-V28 + Time + Amount
        "feature_names": [f"V{i}" for i in range(1, 29)] + ["Time", "Amount"]
    }
    
    if model_metadata:
        info.update({
            "metadata": model_metadata,
            "training_info": {
                "threshold": model_metadata.get("threshold", "Unknown"),
                "k": model_metadata.get("k", "Unknown"),
                "train_file": model_metadata.get("train_file", "Unknown"),
                "test_file": model_metadata.get("test_file", "Unknown")
            }
        })
    
    return info

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 
