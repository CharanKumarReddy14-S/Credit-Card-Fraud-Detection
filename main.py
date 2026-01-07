"""
FastAPI Backend for Credit Card Fraud Detection
Provides REST API endpoints for predictions
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
import joblib
import os
import io
from datetime import datetime

# Load models and preprocessor
try:
    # Load best model (try both formats)
    if os.path.exists('models/best_model.pkl'):
        model = joblib.load('models/best_model.pkl')
        model_type = 'sklearn'
    elif os.path.exists('models/best_model.keras'):
        from tensorflow import keras
        model = keras.models.load_model('models/best_model.keras')
        model_type = 'keras'
    else:
        # Fallback to XGBoost
        model = joblib.load('models/xgboost.pkl')
        model_type = 'sklearn'
    
    # Load preprocessor
    preprocessor_data = joblib.load('models/preprocessor.pkl')
    scaler = preprocessor_data['scaler']
    feature_columns = preprocessor_data['feature_columns']
    
    print("✅ Models loaded successfully")
    
except Exception as e:
    print(f"❌ Error loading models: {e}")
    print("Please run train_model.py first to train and save models")
    model = None
    scaler = None
    feature_columns = None
    model_type = None


# Initialize FastAPI app
app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="API for detecting fraudulent credit card transactions using ML",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for request/response
class Transaction(BaseModel):
    """Single transaction data"""
    Time: float = Field(..., description="Time in seconds from first transaction")
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float = Field(..., description="Transaction amount")
    
    class Config:
        json_schema_extra = {
            "example": {
                "Time": 0,
                "V1": -1.359807,
                "V2": -0.072781,
                "V3": 2.536347,
                "V4": 1.378155,
                "V5": -0.338321,
                "V6": 0.462388,
                "V7": 0.239599,
                "V8": 0.098698,
                "V9": 0.363787,
                "V10": 0.090794,
                "V11": -0.551600,
                "V12": -0.617801,
                "V13": -0.991390,
                "V14": -0.311169,
                "V15": 1.468177,
                "V16": -0.470401,
                "V17": 0.207971,
                "V18": 0.025791,
                "V19": 0.403993,
                "V20": 0.251412,
                "V21": -0.018307,
                "V22": 0.277838,
                "V23": -0.110474,
                "V24": 0.066928,
                "V25": 0.128539,
                "V26": -0.189115,
                "V27": 0.133558,
                "V28": -0.021053,
                "Amount": 149.62
            }
        }


class PredictionResponse(BaseModel):
    """Response for single prediction"""
    is_fraud: bool
    fraud_probability: float
    confidence: str
    risk_level: str
    timestamp: str


class BatchPredictionResponse(BaseModel):
    """Response for batch predictions"""
    total_transactions: int
    fraud_count: int
    legitimate_count: int
    fraud_percentage: float
    predictions: List[Dict]
    timestamp: str


def preprocess_transaction(transaction_dict: Dict) -> pd.DataFrame:
    """
    Preprocess a single transaction for prediction
    """
    df = pd.DataFrame([transaction_dict])
    
    # Feature engineering (same as in training)
    if 'Time' in df.columns:
        df['Hour'] = (df['Time'] / 3600) % 24
        df['Day'] = (df['Time'] / 86400).astype(int)
        df['Time_of_Day'] = pd.cut(df['Hour'], 
                                   bins=[0, 6, 12, 18, 24],
                                   labels=[0, 1, 2, 3])
        df['Time_of_Day'] = df['Time_of_Day'].astype(int)
    
    if 'Amount' in df.columns:
        df['Amount_log'] = np.log1p(df['Amount'])
        df['Amount_Category'] = pd.cut(df['Amount'],
                                      bins=[-np.inf, 10, 100, 500, np.inf],
                                      labels=[0, 1, 2, 3])
        df['Amount_Category'] = df['Amount_Category'].astype(int)
    
    # Ensure all feature columns are present
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    
    # Select and order columns
    df = df[feature_columns]
    
    # Scale features
    df_scaled = scaler.transform(df)
    df_scaled = pd.DataFrame(df_scaled, columns=feature_columns)
    
    return df_scaled


def get_risk_level(probability: float) -> str:
    """Determine risk level based on fraud probability"""
    if probability < 0.3:
        return "LOW"
    elif probability < 0.6:
        return "MEDIUM"
    elif probability < 0.8:
        return "HIGH"
    else:
        return "CRITICAL"


def get_confidence(probability: float) -> str:
    """Determine confidence level"""
    if probability < 0.4 or probability > 0.6:
        return "HIGH"
    else:
        return "MEDIUM"


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "message": "Credit Card Fraud Detection API",
        "version": "1.0.0",
        "model_loaded": model is not None
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_type": model_type,
        "features_count": len(feature_columns) if feature_columns else 0,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_transaction(transaction: Transaction):
    """
    Predict if a single transaction is fraudulent
    
    Args:
        transaction: Transaction data
        
    Returns:
        Prediction result with fraud probability
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert to dict and preprocess
        transaction_dict = transaction.dict()
        processed_data = preprocess_transaction(transaction_dict)
        
        # Make prediction
        if model_type == 'keras':
            fraud_prob = float(model.predict(processed_data)[0][0])
        else:
            fraud_prob = float(model.predict_proba(processed_data)[0][1])
        
        is_fraud = fraud_prob > 0.5
        
        return PredictionResponse(
            is_fraud=is_fraud,
            fraud_probability=round(fraud_prob, 4),
            confidence=get_confidence(fraud_prob),
            risk_level=get_risk_level(fraud_prob),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/upload", response_model=BatchPredictionResponse)
async def upload_csv(file: UploadFile = File(...)):
    """
    Upload CSV file and get fraud predictions for all transactions
    
    Args:
        file: CSV file with transactions
        
    Returns:
        Batch prediction results
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted")
    
    try:
        # Read CSV
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Check if required columns exist
        required_cols = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required columns: {missing_cols}"
            )
        
        # Process each transaction
        predictions = []
        
        for idx, row in df.iterrows():
            transaction_dict = row.to_dict()
            processed_data = preprocess_transaction(transaction_dict)
            
            # Make prediction
            if model_type == 'keras':
                fraud_prob = float(model.predict(processed_data, verbose=0)[0][0])
            else:
                fraud_prob = float(model.predict_proba(processed_data)[0][1])
            
            is_fraud = fraud_prob > 0.5
            
            predictions.append({
                "transaction_id": int(idx),
                "amount": float(row['Amount']),
                "is_fraud": is_fraud,
                "fraud_probability": round(fraud_prob, 4),
                "risk_level": get_risk_level(fraud_prob)
            })
        
        # Calculate statistics
        fraud_count = sum(1 for p in predictions if p['is_fraud'])
        total = len(predictions)
        
        return BatchPredictionResponse(
            total_transactions=total,
            fraud_count=fraud_count,
            legitimate_count=total - fraud_count,
            fraud_percentage=round((fraud_count / total) * 100, 2),
            predictions=predictions,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.get("/model/info")
async def model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": model_type,
        "features_count": len(feature_columns),
        "feature_names": feature_columns[:10],  # First 10 features
        "model_loaded": True
    }


if __name__ == "__main__":
    import uvicorn
    
    print("Starting Credit Card Fraud Detection API...")
    print("API Documentation: http://localhost:8000/docs")
    print("Alternative docs: http://localhost:8000/redoc")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)