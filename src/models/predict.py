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
        raise ValueError(f"Missing required fields: {missing}")

    # Validate GENDER
    if "GENDER" in data_dict:
        valid_genders = {"M", "F", 0, 1}
        if data_dict["GENDER"] not in valid_genders:
            raise ValueError(
                f"Invalid GENDER value: {data_dict['GENDER']}. "
                f"Expected one of {valid_genders}"
            )

    # Validate AGE
    if "AGE" in data_dict:
        age = data_dict["AGE"]
        if not isinstance(age, (int, float)) or age < 0 or age > 120:
            raise ValueError(f"Invalid AGE value: {age}. Must be 0-120")

    # Validate binary columns
    valid_binary = {0, 1, 2}
    for col in BINARY_COLUMNS:
        if col in data_dict:
            if data_dict[col] not in valid_binary:
                raise ValueError(
                    f"Invalid value for {col}: {data_dict[col]}. "
                    f"Expected one of {valid_binary}"
                )

    logger.info("Input validation passed [OK]")


def _preprocess_input(data_dict: Dict[str, Any]) -> pd.DataFrame:
    """
    Preprocess a single input record for prediction.

    Applies the same transformations as the training pipeline:
        1. Encode GENDER
        2. Remap binary columns (1/2 → 0/1)
        3. Scale numerical features

    Args:
        data_dict: Raw patient data.

    Returns:
        Preprocessed DataFrame ready for prediction.
    """
    # Create DataFrame from input
    df = pd.DataFrame([data_dict])

    # Encode GENDER
    if "GENDER" in df.columns and pd.api.types.is_string_dtype(df["GENDER"]):
        df["GENDER"] = df["GENDER"].map({"M": 1, "F": 0}).astype(int)

    # Remap binary columns (1/2 → 0/1)
    for col in BINARY_COLUMNS:
        if col in df.columns:
            unique_vals = set(df[col].dropna().unique())
            if unique_vals.issubset({1, 2}):
                df[col] = df[col].map({1: 0, 2: 1})

    # Remove target column if present
    if TARGET_COLUMN in df.columns:
        df = df.drop(columns=[TARGET_COLUMN])

    # Scale numerical features
    scaler = _load_preprocessor()
    df[NUMERICAL_COLUMNS] = scaler.transform(df[NUMERICAL_COLUMNS])

    # Build features
    df = build_features(df)

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
    # Example prediction
    sample_patient = {
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

    result = predict(sample_patient)
    print(f"\nPrediction Result:")
    print(f"  Label:       {result['label']}")
    print(f"  Prediction:  {result['prediction']}")
    print(f"  Probability: {result['probability']}")
    print(f"  Risk Level:  {result['risk_level']}")
