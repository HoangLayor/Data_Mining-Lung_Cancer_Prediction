"""
FastAPI Application for the Lung Cancer Prediction project.

Provides REST API endpoints for predictions and data management.

Run with:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger
from src.utils.config import NEW_DATA_FILE, RAW_DATA_DIR, DATABASE_URL
from src.utils.database import get_db
from src.utils.models import Patient
from src.models.predict import predict, reload_model

logger = get_logger("api")

# ──────────────────────────────────────────────
# FastAPI App Setup
# ──────────────────────────────────────────────
app = FastAPI(
    title="Lung Cancer Risk Prediction API",
    description=(
        "Machine Learning API for predicting lung cancer risk "
        "based on patient survey data. Uses trained models with "
        "SHAP explainability support."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# Pydantic Schemas
# ──────────────────────────────────────────────
class PatientData(BaseModel):
    """Schema for patient input data (29 features)."""

    age: int = Field(..., ge=1, le=120, description="Patient age (1-120)")
    gender: int = Field(..., description="Gender: 0=Female, 1=Male")
    education_years: int = Field(..., ge=0, le=30, description="Total years of formal education")
    income_level: int = Field(..., ge=1, le=5, description="Income level (1-5)")
    smoker: int = Field(..., ge=0, le=1, description="Smoker: 0=No, 1=Yes")
    smoking_years: float = Field(..., ge=0, description="Total number of years smoked")
    cigarettes_per_day: float = Field(..., ge=0, description="Average cigarettes per day")
    pack_years: float = Field(..., ge=0, description="Cumulative smoking exposure (pack-years)")
    passive_smoking: int = Field(..., ge=0, le=1, description="Exposure to secondhand smoke (0/1)")
    air_pollution_index: float = Field(..., ge=0, le=100, description="Air quality exposure index (0-100)")
    occupational_exposure: int = Field(..., ge=0, le=1, description="Hazardous substance exposure at work (0/1)")
    radon_exposure: int = Field(..., ge=0, le=1, description="History of radon exposure (0/1)")
    family_history_cancer: int = Field(..., ge=0, le=1, description="Family history of cancer (0/1)")
    copd: int = Field(..., ge=0, le=1, description="Diagnosis of COPD (0/1)")
    asthma: int = Field(..., ge=0, le=1, description="History of asthma (0/1)")
    previous_tb: int = Field(..., ge=0, le=1, description="History of tuberculosis (0/1)")
    chronic_cough: int = Field(..., ge=0, le=1, description="Long-term cough symptoms (0/1)")
    chest_pain: int = Field(..., ge=0, le=1, description="Reports of chest pain (0/1)")
    shortness_of_breath: int = Field(..., ge=0, le=1, description="Breathing difficulty (0/1)")
    fatigue: int = Field(..., ge=0, le=1, description="Persistent fatigue symptoms (0/1)")
    bmi: float = Field(..., ge=10, le=60, description="Body mass index")
    oxygen_saturation: float = Field(..., ge=70, le=100, description="Blood oxygen saturation level (%)")
    fev1_x10: float = Field(..., description="Lung function measure (FEV1)")
    crp_level: float = Field(..., description="C-reactive protein level (inflammation)")
    xray_abnormal: int = Field(..., ge=0, le=1, description="Abnormal imaging findings (0/1)")
    exercise_hours_per_week: float = Field(..., ge=0, description="Weekly physical activity duration")
    diet_quality: int = Field(..., ge=1, le=5, description="Overall dietary quality (1-5)")
    alcohol_units_per_week: float = Field(..., ge=0, description="Average alcohol consumption per week")
    healthcare_access: int = Field(..., ge=1, le=5, description="Access to healthcare services (1-5)")

    class Config:
        json_schema_extra = {
            "example": {
                "age": 62,
                "gender": 1,
                "education_years": 16,
                "income_level": 3,
                "smoker": 1,
                "smoking_years": 25.0,
                "cigarettes_per_day": 20.0,
                "pack_years": 25.0,
                "passive_smoking": 0,
                "air_pollution_index": 45.0,
                "occupational_exposure": 1,
                "radon_exposure": 0,
                "family_history_cancer": 1,
                "copd": 0,
                "asthma": 0,
                "previous_tb": 0,
                "chronic_cough": 1,
                "chest_pain": 1,
                "shortness_of_breath": 0,
                "fatigue": 1,
                "bmi": 26.5,
                "oxygen_saturation": 98.0,
                "fev1_x10": 38.0,
                "crp_level": 1.5,
                "xray_abnormal": 0,
                "exercise_hours_per_week": 3.0,
                "diet_quality": 4,
                "alcohol_units_per_week": 5.0,
                "healthcare_access": 4
            }
        }


class PredictionResponse(BaseModel):
    """Schema for prediction response."""

    prediction: int = Field(..., description="0 = No Cancer, 1 = Cancer")
    probability: float = Field(..., description="Probability of cancer (0-1)")
    risk_level: str = Field(..., description="Risk level: Low / Medium / High")
    label: str = Field(..., description="Human-readable prediction label")


class UpdateDataRequest(BaseModel):
    """Schema for data update request."""

    records: list = Field(..., description="List of patient data records")


class StatusResponse(BaseModel):
    """Schema for status response."""

    status: str
    message: str
    timestamp: str


# ──────────────────────────────────────────────
# Middleware
# ──────────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests."""
    logger.info(f"Request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Response: {response.status_code}")
    return response


# ──────────────────────────────────────────────
# Global Exception Handler
# ──────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all unhandled exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat(),
        },
    )


# ──────────────────────────────────────────────
# API Endpoints
# ──────────────────────────────────────────────
@app.get("/", response_model=StatusResponse)
async def root():
    """Health check endpoint."""
    return StatusResponse(
        status="healthy",
        message="Lung Cancer Risk Prediction API is running",
        timestamp=datetime.now().isoformat(),
    )


@app.get("/health", response_model=StatusResponse)
async def health_check():
    """Detailed health check."""
    return StatusResponse(
        status="healthy",
        message="All systems operational",
        timestamp=datetime.now().isoformat(),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(patient: PatientData):
    """
    Predict lung cancer risk for a patient.

    Takes patient survey data and returns a prediction with
    probability and risk level.
    """
    logger.info(f"Prediction request received for patient: age={patient.age}, gender={patient.gender}")

    try:
        # Convert Pydantic model to dict
        data_dict = patient.model_dump()
        result = predict(data_dict)

        return PredictionResponse(
            prediction=result["prediction"],
            probability=result["probability"],
            risk_level=result["risk_level"],
            label=result["label"],
        )
    except FileNotFoundError as e:
        logger.error(f"Model files not found: {e}")
        raise HTTPException(
            status_code=503,
            detail="Model not available. Please train the model first.",
        )
    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/update-data", response_model=StatusResponse)
async def update_data(request: UpdateDataRequest):
    """
    Submit new patient data for future retraining.

    Appends records to the new data CSV file.
    """
    logger.info(f"Received {len(request.records)} new records for retraining")

    try:
        with get_db() as db:
            for rec in request.records:
                # Ensure record has correct source
                patient = Patient(**rec, source="api")
                db.add(patient)
            db.commit()

        logger.info(f"Successfully saved {len(request.records)} records to database")

        return StatusResponse(
            status="success",
            message=f"Saved {len(request.records)} records to PostgreSQL database",
            timestamp=datetime.now().isoformat(),
        )
    except Exception as e:
        logger.error(f"Data update error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to save data: {str(e)}")


@app.post("/reload-model", response_model=StatusResponse)
async def reload_model_endpoint():
    """Reload the model from disk (e.g., after retraining)."""
    try:
        reload_model()
        return StatusResponse(
            status="success",
            message="Model reloaded successfully",
            timestamp=datetime.now().isoformat(),
        )
    except Exception as e:
        logger.error(f"Model reload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    logger.info("Starting Lung Cancer Prediction API...")
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
