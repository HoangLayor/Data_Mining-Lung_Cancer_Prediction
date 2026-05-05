from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from typing import Optional
import mlflow.sklearn
import pandas as pd
import os
import sys

# Ensure src is in python path
sys.path.append("/app")
from src.data.preprocess import preprocess_data
from src.features.build_features import build_features
from src.utils.config import NEW_DATA_FILE
from src.models.explain import get_local_explanation_dict

app = FastAPI(title="Lung Cancer Prediction API")

# Setup templates and static files
import os
base_dir = os.path.dirname(os.path.abspath(__file__))
frontend_dir = os.path.abspath(os.path.join(base_dir, "..", "frontend"))

app.mount("/static", StaticFiles(directory=os.path.join(frontend_dir, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(frontend_dir, "templates"))

@app.get("/")
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

MODEL_URI = f"models:/{os.environ.get('REGISTERED_MODEL_NAME', 'lung_cancer_model')}/{os.environ.get('MODEL_STAGE', 'Production')}"
model = None
background_data = None

def load_background():
    global background_data
    try:
        from src.utils.config import PROCESSED_DATA_FILE
        if os.path.exists(PROCESSED_DATA_FILE):
            df = pd.read_csv(PROCESSED_DATA_FILE)
            if "lung_cancer_risk" in df.columns:
                df = df.drop(columns=["lung_cancer_risk"])
            
            # Use build_features to match the model's expected input
            from src.features.build_features import build_features
            df = build_features(df)
            
            # Select 100 random samples for background
            background_data = df.sample(min(100, len(df)), random_state=42)
            print(f"Background data loaded: {background_data.shape}")
    except Exception as e:
        print(f"Could not load background data: {e}")

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
    alcohol_units_per_week: float = Field(..., ge=0, description="Alcohol units consumed per week")
    healthcare_access: int = Field(..., ge=1, le=5, description="Access to healthcare (1-5)")
    lung_cancer_risk: Optional[int] = Field(None, ge=0, le=1, description="Target label (0=No, 1=Yes) - Only for ingestion")

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

@app.on_event("startup")
def startup_event():
    load_model()
    load_background()

def load_model():
    global model
    try:
        print(f"Loading model from {MODEL_URI}")
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
        model = mlflow.sklearn.load_model(MODEL_URI)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Warning: Failed to load model on startup: {e}")

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
def predict(data: PatientData):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    
    df = pd.DataFrame([data.model_dump()])
    
    try:
        # Preprocess the incoming JSON data using the saved preprocessor
        X, _, _ = preprocess_data(df, fit_preprocessor=False, save_output=False)
        X = build_features(X)

        # Filter features to match exactly what the model expects
        if hasattr(model, "feature_names_in_"):
            X = X[model.feature_names_in_]

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[0]
            risk_prob = float(probs[1]) if len(probs) > 1 else float(probs[0])
            # Assuming models output 0 or 1, or string "YES"/"NO"
            prediction = model.predict(X)[0]
        else:
            prediction = model.predict(X)[0]
            risk_prob = 1.0 if prediction in [1, "YES"] else 0.0
            
        pred_label = "YES" if prediction in [1, "YES", "1"] else "NO"
            
        return {
            "prediction": pred_label,
            "risk_probability": risk_prob,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/explain")
def explain(data: PatientData):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    
    df = pd.DataFrame([data.model_dump()])
    
    try:
        # Preprocess the incoming JSON data using the saved preprocessor
        X, _, _ = preprocess_data(df, fit_preprocessor=False, save_output=False)
        X = build_features(X)

        # Filter features to match exactly what the model expects
        if hasattr(model, "feature_names_in_"):
            X = X[model.feature_names_in_]
            if background_data is not None:
                # Ensure background columns match exactly
                bg = background_data[model.feature_names_in_]
            else:
                bg = None

        explanation = get_local_explanation_dict(model, X, X_background=bg)
            
        return {
            "status": "success",
            "explanation": explanation
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/reload-model")
def reload_model_endpoint():
    load_model()
    if model is None:
        raise HTTPException(status_code=500, detail="Failed to reload model")
    return {"status": "success", "message": "Model reloaded successfully"}

class UpdateDataRequest(BaseModel):
    records: list[PatientData] = Field(..., description="List of patient data records")

    model_config = {
        "json_schema_extra": {
            "example": {
                "records": [
                    {
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
                        "healthcare_access": 4,
                        "lung_cancer_risk": 1
                    }
                ]
            }
        }
    }

@app.post("/ingest-data")
def ingest_data(request: UpdateDataRequest):
    try:
        if not request.records:
            return {"status": "success", "message": "No records provided."}
            
        df = pd.DataFrame([r.model_dump() for r in request.records])
        
        # Save to NEW_DATA_FILE
        # If it doesn't exist, write with header. Otherwise, append without header.
        if not os.path.exists(NEW_DATA_FILE):
            df.to_csv(NEW_DATA_FILE, index=False)
        else:
            df.to_csv(NEW_DATA_FILE, mode='a', header=False, index=False)
            
        return {
            "status": "success", 
            "message": f"Successfully ingested {len(df)} records.",
            "file_path": str(NEW_DATA_FILE)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    try:
        # Read the uploaded CSV file
        df = pd.read_csv(file.file)
        
        # 1. Validate Columns
        expected_cols = list(PatientData.model_fields.keys())
        missing_cols = [col for col in expected_cols if col not in df.columns]
        if missing_cols:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required columns: {missing_cols}"
            )
        
        # 2. Validate Data Types and Constraints (using Pydantic)
        # We validate a sample or all rows depending on file size
        try:
            for i, row in df[expected_cols].head(100).iterrows():
                PatientData(**row.to_dict())
        except Exception as e:
            raise HTTPException(
                status_code=400, 
                detail=f"Data validation failed at row {i}: {str(e)}"
            )

        # Save to NEW_DATA_FILE
        # If it doesn't exist, write with header. Otherwise, append without header.
        if not os.path.exists(NEW_DATA_FILE):
            df[expected_cols].to_csv(NEW_DATA_FILE, index=False)
        else:
            df[expected_cols].to_csv(NEW_DATA_FILE, mode='a', header=False, index=False)
            
        return {
            "status": "success", 
            "message": f"Successfully validated and saved {len(df)} records.",
            "file_path": str(NEW_DATA_FILE)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing CSV: {str(e)}")
