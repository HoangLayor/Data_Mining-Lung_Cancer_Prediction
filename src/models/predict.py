"""
Prediction Module for the Lung Cancer Prediction project.

Handles loading the trained model and preprocessor,
validating input data, and returning predictions.
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from src.utils.logger import get_logger
from src.utils.config import (
    BEST_MODEL_FILE,
    PREPROCESSOR_FILE,
    REQUIRED_COLUMNS,
    TARGET_COLUMN,
    BINARY_COLUMNS,
    NUMERICAL_COLUMNS,
    CATEGORICAL_COLUMNS,
)
from src.features.build_features import build_features

logger = get_logger(__name__)

# ──────────────────────────────────────────────
# Module-level cache for model and preprocessor
# ──────────────────────────────────────────────
_cached_model = None
_cached_preprocessor = None


def _load_model(model_path: Optional[str] = None):
    """Load and cache the trained model."""
    global _cached_model
    if _cached_model is not None:
        return _cached_model

    path = Path(model_path) if model_path else BEST_MODEL_FILE
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    _cached_model = joblib.load(path)
    logger.info(f"Model loaded from: {path}")
    return _cached_model


def _load_preprocessor(preprocessor_path: Optional[str] = None):
    """Load and cache the preprocessor."""
    global _cached_preprocessor
    if _cached_preprocessor is not None:
        return _cached_preprocessor

    path = Path(preprocessor_path) if preprocessor_path else PREPROCESSOR_FILE
    if not path.exists():
        raise FileNotFoundError(f"Preprocessor file not found: {path}")

    _cached_preprocessor = joblib.load(path)
    logger.info(f"Preprocessor loaded from: {path}")
    return _cached_preprocessor


def _validate_input(data_dict: Dict[str, Any]) -> None:
    """
    Validate input data dictionary.

    Args:
        data_dict: Patient data dictionary.

    Raises:
        ValueError: If required fields are missing or invalid.
    """
    # Required feature columns (excluding target)
    required_features = [col for col in REQUIRED_COLUMNS if col != TARGET_COLUMN]
    missing = [col for col in required_features if col not in data_dict]

    if missing:
        # Check if they are provided in uppercase
        for col in missing:
            if col.upper() in data_dict:
                # Silently fix case or warn? Let's just warn and fix in preprocess.
                pass
            else:
                raise ValueError(f"Missing required fields: {missing}")

    # Validate age
    age_col = "age" if "age" in data_dict else "AGE"
    if age_col in data_dict:
        age = data_dict[age_col]
        if not isinstance(age, (int, float)) or age < 0 or age > 120:
            raise ValueError(f"Invalid age value: {age}. Must be 0-120")

    # Validate gender
    gender_col = "gender" if "gender" in data_dict else "GENDER"
    if gender_col in data_dict:
        val = data_dict[gender_col]
        if isinstance(val, str):
            if val.upper() not in ("M", "F", "MALE", "FEMALE"):
                 raise ValueError(f"Invalid gender value: {val}")
        elif val not in (0, 1):
             raise ValueError(f"Invalid gender value: {val}. Expected 0 or 1")

    logger.info("Input validation passed [OK]")


def _preprocess_input(data_dict: Dict[str, Any]) -> pd.DataFrame:
    """
    Preprocess a single input record for prediction.

    Applies the same transformations as the training pipeline.

    Args:
        data_dict: Raw patient data.

    Returns:
        Preprocessed DataFrame ready for prediction.
    """
    # Normalize keys to lowercase
    data_dict = {k.lower(): v for k, v in data_dict.items()}
    
    # Create DataFrame from input
    df = pd.DataFrame([data_dict])

    # Import preprocessing functions locally to avoid circular imports
    from src.data.preprocess import _encode_gender, _remap_binary_columns

    # Apply standard preprocessing
    df = _encode_gender(df)
    df = _remap_binary_columns(df)

    # Ensure all required features are present
    required_features = [col for col in REQUIRED_COLUMNS if col != TARGET_COLUMN]
    for col in required_features:
        if col not in df.columns:
            df[col] = 0 # Default value
            
    # Remove target column if present
    if TARGET_COLUMN in df.columns:
        df = df.drop(columns=[TARGET_COLUMN])

    # Scale numerical features
    scaler = _load_preprocessor()
    df[NUMERICAL_COLUMNS] = scaler.transform(df[NUMERICAL_COLUMNS])

    # Build features
    df = build_features(df)

    # Align with training features from metadata
    try:
        import json
        from src.utils.config import MODEL_METADATA_FILE
        if MODEL_METADATA_FILE.exists():
            with open(MODEL_METADATA_FILE, "r") as f:
                metadata = json.load(f)
                # Get last version's features
                if metadata:
                    feature_names = metadata[-1].get("feature_names", [])
                    if feature_names:
                        # Add missing columns (as 0) and filter/reorder
                        for col in feature_names:
                            if col not in df.columns:
                                df[col] = 0
                        df = df[feature_names]
                        logger.info(f"Aligned features with metadata (count: {len(feature_names)})")
    except Exception as e:
        logger.warning(f"Could not align features with metadata: {e}")

    return df


def predict(
    data_dict: Dict[str, Any],
    model_path: Optional[str] = None,
    preprocessor_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Make a lung cancer risk prediction for a single patient.

    Args:
        data_dict: Patient data dictionary with all required fields.
        model_path: Optional path to model file.
        preprocessor_path: Optional path to preprocessor file.

    Returns:
        Dictionary with:
            - prediction (int): 0 (no cancer) or 1 (cancer)
            - probability (float): Probability of cancer
            - risk_level (str): Low / Medium / High

    Raises:
        ValueError: If input validation fails.
        FileNotFoundError: If model or preprocessor files are missing.
    """
    logger.info("Starting prediction...")
    logger.info(f"Input data: {data_dict}")

    # Step 1: Validate input
    _validate_input(data_dict)

    # Step 2: Load model
    model = _load_model(model_path)

    # Step 3: Preprocess input
    X = _preprocess_input(data_dict)
    logger.info(f"Preprocessed features shape: {X.shape}")

    # Step 4: Predict
    prediction = int(model.predict(X)[0])
    probability = float(model.predict_proba(X)[0][1])

    # Determine risk level
    if probability < 0.3:
        risk_level = "Low"
    elif probability < 0.7:
        risk_level = "Medium"
    else:
        risk_level = "High"

    result = {
        "prediction": prediction,
        "probability": round(probability, 4),
        "risk_level": risk_level,
        "label": "Lung Cancer Detected" if prediction == 1 else "No Lung Cancer",
    }

    logger.info(f"Prediction result: {result}")
    return result


def reload_model() -> None:
    """Force reload of model and preprocessor (e.g., after retraining)."""
    global _cached_model, _cached_preprocessor
    _cached_model = None
    _cached_preprocessor = None
    logger.info("Model cache cleared. Will reload on next prediction.")


if __name__ == "__main__":
    # Example prediction with new schema
    sample_patient = {
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

    result = predict(sample_patient)
    print(f"\nPrediction Result:")
    print(f"  Label:       {result['label']}")
    print(f"  Prediction:  {result['prediction']}")
    print(f"  Probability: {result['probability']}")
    print(f"  Risk Level:  {result['risk_level']}")
