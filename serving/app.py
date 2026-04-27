"""
Car Price Prediction — Serving API
====================================
Load model từ MLflow Model Registry (stage=Production).
Expose endpoint POST /predict nhận các features và trả về giá dự đoán.
"""

import os
import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from loguru import logger

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
REGISTERED_MODEL_NAME = os.environ.get("REGISTERED_MODEL_NAME", "car_price_model")
MODEL_STAGE = os.environ.get("MODEL_STAGE", "Production")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

logger.info(f"Loading model '{REGISTERED_MODEL_NAME}' at stage='{MODEL_STAGE}' from MLflow...")
try:
    model = mlflow.sklearn.load_model(
        model_uri=f"models:/{REGISTERED_MODEL_NAME}/{MODEL_STAGE}"
    )
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None

app = FastAPI(
    title="Car Price Prediction API",
    description="Dự đoán giá xe dựa trên các features đầu vào. Model được load từ MLflow Registry.",
    version="1.0.0",
)


class CarFeatures(BaseModel):
    brand: str
    model: str
    year: int
    mileage: float
    fuel_type: str
    transmission: str
    color: str | None = None
    num_owners: int | None = None
    # Thêm các field khác nếu bảng CarInfo có thêm cột


class PredictionResponse(BaseModel):
    predicted_price: float
    model_stage: str
    model_name: str


@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/health", tags=["Health"])
def health():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check MLflow connection.")
    return {"status": "healthy", "model_stage": MODEL_STAGE}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(features: CarFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not available.")

    try:
        input_df = pd.DataFrame([features.model_dump()])
        predicted_price = model.predict(input_df)[0]

        logger.info(f"Prediction: {predicted_price:,.0f} | Input: {features.model_dump()}")

        return PredictionResponse(
            predicted_price=float(predicted_price),
            model_stage=MODEL_STAGE,
            model_name=REGISTERED_MODEL_NAME,
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/reload-model", tags=["Model Management"])
def reload_model():
    """
    Endpoint để hot-reload model mới nhất từ MLflow Registry.
    Gọi endpoint này sau khi DAG promote_best_model chạy xong
    để serving app tự động dùng model Production mới nhất.
    """
    global model
    try:
        logger.info("Reloading model from MLflow Registry...")
        model = mlflow.sklearn.load_model(
            model_uri=f"models:/{REGISTERED_MODEL_NAME}/{MODEL_STAGE}"
        )
        logger.info("Model reloaded successfully.")
        return {"status": "reloaded", "model_stage": MODEL_STAGE}
    except Exception as e:
        logger.error(f"Reload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reload model: {str(e)}")
