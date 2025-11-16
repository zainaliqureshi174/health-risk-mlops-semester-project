"""
FastAPI Inference Server for Health Risk Prediction
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import uvicorn

app = FastAPI(title="Health Risk Prediction API", version="1.0")

# Load model and scaler
try:
    with open('models/federated_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/federated_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("✅ Model and scaler loaded successfully")
except Exception as e:
    print(f"⚠️  Error loading model: {e}")
    model = None
    scaler = None

# Request schema
class HealthData(BaseModel):
    pm25: float
    pm10: float
    aqi: float
    temperature: float
    humidity: float
    precipitation: float
    heart_rate_avg: float
    sleep_hours: float
    spo2: float
    stress_level: int
    
    class Config:
        json_schema_extra = {
            "example": {
                "pm25": 50.0,
                "pm10": 80.0,
                "aqi": 150.0,
                "temperature": 25.0,
                "humidity": 60.0,
                "precipitation": 2.0,
                "heart_rate_avg": 75.0,
                "sleep_hours": 7.0,
                "spo2": 98.0,
                "stress_level": 5
            }
        }

# Response schema
class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    risk_level: str

@app.get("/")
def root():
    return {
        "message": "Health Risk Prediction API",
        "version": "1.0",
        "endpoints": {
            "/predict": "POST - Make prediction",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation"
        }
    }

@app.get("/health")
def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict", response_model=PredictionResponse)
def predict(data: HealthData):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Prepare input
        features = np.array([[
            data.pm25, data.pm10, data.aqi,
            data.temperature, data.humidity, data.precipitation,
            data.heart_rate_avg, data.sleep_hours,
            data.spo2, data.stress_level
        ]])
        
        # Scale
        features_scaled = scaler.transform(features)
        
        # Predict
        prediction = int(model.predict(features_scaled)[0])
        probability = float(model.predict_proba(features_scaled)[0][1])
        
        # Risk level
        if probability < 0.3:
            risk_level = "Low"
        elif probability < 0.6:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        return PredictionResponse(
            prediction=prediction,
            probability=probability,
            risk_level=risk_level
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("🚀 Starting Health Risk Prediction API...")
    print("📊 API docs: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
