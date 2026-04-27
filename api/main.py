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
from src.utils.config import NEW_DATA_FILE, RAW_DATA_DIR
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
    """Schema for patient input data."""

    GENDER: str = Field(..., description="Patient gender: M or F")
    AGE: int = Field(..., ge=1, le=120, description="Patient age (1-120)")
    SMOKING: int = Field(..., ge=1, le=2, description="Smoking: 1=No, 2=Yes")
    YELLOW_FINGERS: int = Field(..., ge=1, le=2, description="Yellow fingers: 1=No, 2=Yes")
    ANXIETY: int = Field(..., ge=1, le=2, description="Anxiety: 1=No, 2=Yes")
    PEER_PRESSURE: int = Field(..., ge=1, le=2, description="Peer pressure: 1=No, 2=Yes")
    CHRONIC_DISEASE: int = Field(..., ge=1, le=2, description="Chronic disease: 1=No, 2=Yes")
    FATIGUE: int = Field(..., ge=1, le=2, description="Fatigue: 1=No, 2=Yes")
    ALLERGY: int = Field(..., ge=1, le=2, description="Allergy: 1=No, 2=Yes")
    WHEEZING: int = Field(..., ge=1, le=2, description="Wheezing: 1=No, 2=Yes")
    ALCOHOL_CONSUMING: int = Field(..., ge=1, le=2, description="Alcohol consuming: 1=No, 2=Yes")
    COUGHING: int = Field(..., ge=1, le=2, description="Coughing: 1=No, 2=Yes")
    SHORTNESS_OF_BREATH: int = Field(..., ge=1, le=2, description="Shortness of breath: 1=No, 2=Yes")
    SWALLOWING_DIFFICULTY: int = Field(..., ge=1, le=2, description="Swallowing difficulty: 1=No, 2=Yes")
    CHEST_PAIN: int = Field(..., ge=1, le=2, description="Chest pain: 1=No, 2=Yes")

    @validator("GENDER")
    def validate_gender(cls, v):
        if v.upper() not in ("M", "F"):
            raise ValueError("GENDER must be 'M' or 'F'")
        return v.upper()

    class Config:
        json_schema_extra = {
            "example": {
                "GENDER": "M",
                "AGE": 65,
                "SMOKING": 2,
                "YELLOW_FINGERS": 2,
                "ANXIETY": 1,
                "PEER_PRESSURE": 1,
                "CHRONIC_DISEASE": 2,
                "FATIGUE": 2,
                "ALLERGY": 1,
                "WHEEZING": 2,
                "ALCOHOL_CONSUMING": 2,
                "COUGHING": 2,
                "SHORTNESS_OF_BREATH": 2,
                "SWALLOWING_DIFFICULTY": 1,
                "CHEST_PAIN": 2,
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
    logger.info(f"Prediction request received for patient: AGE={patient.AGE}, GENDER={patient.GENDER}")

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
        new_df = pd.DataFrame(request.records)

        # Append to existing file or create new
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        if NEW_DATA_FILE.exists():
            existing = pd.read_csv(NEW_DATA_FILE)
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined.to_csv(NEW_DATA_FILE, index=False)
            total = len(combined)
        else:
            new_df.to_csv(NEW_DATA_FILE, index=False)
            total = len(new_df)

        logger.info(f"Data saved to {NEW_DATA_FILE}. Total records: {total}")

        return StatusResponse(
            status="success",
            message=f"Saved {len(request.records)} records. Total: {total}",
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
