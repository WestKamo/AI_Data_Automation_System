from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from datetime import datetime, timedelta
import joblib
from pathlib import Path
import json
from transformers import pipeline
import nltk
from prophet import Prophet

# Download required NLTK data
nltk.download('vader_lexicon')
nltk.download('punkt')

app = FastAPI(title="AI-Powered Analytics System")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data models
class TimeSeriesData(BaseModel):
    dates: list
    values: list

class TextData(BaseModel):
    text: str

class TransactionData(BaseModel):
    amount: float
    timestamp: str
    category: str

# Initialize AI models
sentiment_analyzer = pipeline("sentiment-analysis")

# Endpoints
@app.get("/")
async def root():
    return {
        "message": "Welcome to AI-Powered Analytics System",
        "version": "1.0.0",
        "endpoints": [
            "/predict/sales",
            "/analyze/sentiment",
            "/detect/anomalies",
            "/analyze/trends"
        ]
    }

@app.post("/predict/sales")
async def predict_sales(data: TimeSeriesData):
    try:
        # Create and fit Prophet model
        df = pd.DataFrame({
            'ds': pd.to_datetime(data.dates),
            'y': data.values
        })
        model = Prophet()
        model.fit(df)
        
        # Make future predictions
        future_dates = model.make_future_dataframe(periods=30)
        forecast = model.predict(future_dates)
        
        return {
            "predictions": forecast['yhat'].tail(30).tolist(),
            "dates": forecast['ds'].tail(30).astype(str).tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/sentiment")
async def analyze_sentiment(data: TextData):
    try:
        result = sentiment_analyzer(data.text)[0]
        return {
            "sentiment": result['label'],
            "confidence": result['score']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect/anomalies")
async def detect_anomalies(transactions: list[TransactionData]):
    try:
        # Convert transactions to DataFrame
        df = pd.DataFrame([t.dict() for t in transactions])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Train isolation forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        anomalies = iso_forest.fit_predict(df[['amount']])
        
        # Return anomalous transactions
        anomaly_indices = np.where(anomalies == -1)[0]
        return {
            "anomalies": df.iloc[anomaly_indices].to_dict('records'),
            "total_anomalies": len(anomaly_indices)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analyze/trends")
async def analyze_trends():
    try:
        # Simulate trend analysis (replace with actual data in production)
        trends = {
            "rising_categories": ["Electronics", "Healthcare", "Remote Work Tools"],
            "declining_categories": ["Travel", "Entertainment"],
            "stable_categories": ["Food", "Utilities"],
            "timestamp": datetime.now().isoformat()
        }
        return trends
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
