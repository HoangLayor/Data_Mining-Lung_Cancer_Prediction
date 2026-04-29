"""
Data Ingestion Module for the Lung Cancer Prediction project.

Handles loading data from CSV files and optional API sources,
validates schema, and saves raw data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

from src.utils.logger import get_logger
from src.utils.config import (
    REQUIRED_COLUMNS,
    RAW_DATA_DIR,
    RAW_DATA_FILE,
    BINARY_COLUMNS,
    CATEGORICAL_COLUMNS,
    NUMERICAL_COLUMNS,
    TARGET_COLUMN,
    DATABASE_URL,
)
from src.utils.database import get_db
from src.utils.models import Patient

logger = get_logger(__name__)


def load_csv(path: Optional[str] = None) -> pd.DataFrame:
    """
    Load dataset from a CSV file.

    Args:
        path: Path to the CSV file. Defaults to configured RAW_DATA_FILE.

    Returns:
        DataFrame with the loaded data.

    Raises:
        FileNotFoundError: If the file does not exist.
        pd.errors.EmptyDataError: If the file is empty.
    """
    file_path = Path(path) if path else RAW_DATA_FILE
    logger.info(f"Starting data ingestion from: {file_path}")

    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"Data file not found: {file_path}")

    try:
        df = pd.read_csv(file_path)
        logger.info(f"Successfully loaded {len(df)} records with {len(df.columns)} columns")
        logger.info(f"Columns: {list(df.columns)}")
        return df
    except pd.errors.EmptyDataError:
        logger.error(f"Empty file: {file_path}")
    except Exception as e:
        logger.error(f"Error loading CSV: {e}")
        raise


def load_from_db() -> pd.DataFrame:
    """
    Load dataset from the PostgreSQL 'patients' table.

    Returns:
        DataFrame with the loaded data.
    """
    logger.info("Starting data ingestion from PostgreSQL database...")
    
    try:
        with get_db() as db:
            query = db.query(Patient)
            df = pd.read_sql(query.statement, db.bind)
        
        # Remove metadata columns not needed for training
        if "id" in df.columns:
            df = df.drop(columns=["id"])
        if "source" in df.columns:
            df = df.drop(columns=["source"])
        if "created_at" in df.columns:
            df = df.drop(columns=["created_at"])
            
        logger.info(f"Successfully loaded {len(df)} records from database")
        return df
    except Exception as e:
        logger.error(f"Error loading from database: {e}")
        raise


def load_from_api() -> pd.DataFrame:
    """
    Mock function to simulate loading data from an external API.

    Returns:
        DataFrame with synthetic sample data.
    """
    logger.info("Loading data from API (mock)...")

    np.random.seed(42)
    n_samples = 100

    data = {
        "age": np.random.randint(20, 85, n_samples),
        "gender": np.random.randint(0, 2, n_samples),
        "education_years": np.random.randint(8, 22, n_samples),
        "income_level": np.random.randint(1, 6, n_samples),
        "smoking_years": np.random.randint(0, 40, n_samples),
        "cigarettes_per_day": np.random.randint(0, 40, n_samples),
        "pack_years": np.random.uniform(0, 60, n_samples),
        "air_pollution_index": np.random.uniform(10, 100, n_samples),
        "bmi": np.random.uniform(18, 40, n_samples),
        "oxygen_saturation": np.random.uniform(90, 100, n_samples),
        "fev1_x10": np.random.uniform(20, 50, n_samples),
        "crp_level": np.random.uniform(0, 10, n_samples),
        "exercise_hours_per_week": np.random.uniform(0, 15, n_samples),
        "diet_quality": np.random.randint(1, 6, n_samples),
        "alcohol_units_per_week": np.random.uniform(0, 30, n_samples),
        "healthcare_access": np.random.randint(1, 6, n_samples),
    }

    for col in BINARY_COLUMNS:
        data[col] = np.random.choice([0, 1], n_samples)

    data[TARGET_COLUMN] = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])

    df = pd.DataFrame(data)
    # Ensure all required columns are present
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = 0
            
    # Reorder columns to match standard dataset
    df = df[REQUIRED_COLUMNS]

    logger.info(f"Mock API returned {len(df)} records")
    return df


def validate_data(df: pd.DataFrame) -> bool:
    """
    Validate the DataFrame schema against expected columns and types.

    Args:
        df: DataFrame to validate.

    Returns:
        True if validation passes.

    Raises:
        ValueError: If required columns are missing or data types are wrong.
    """
    logger.info("Validating data schema...")

    # Check required columns
    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        error_msg = f"Missing required columns: {missing_cols}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Check data types
    errors = []

    # age should be numeric
    if "age" in df.columns and not pd.api.types.is_numeric_dtype(df["age"]):
        errors.append("age column must be numeric")

    # gender should be numeric (0/1)
    if "gender" in df.columns and not pd.api.types.is_numeric_dtype(df["gender"]):
        errors.append("gender column must be numeric (0 or 1)")

    # Binary columns should contain only 0 or 1
    for col in BINARY_COLUMNS:
        if col in df.columns:
            unique_vals = set(df[col].dropna().unique())
            valid_vals = {0, 1}
            if not unique_vals.issubset(valid_vals):
                errors.append(f"{col} contains invalid values: {unique_vals}. Expected {valid_vals}")

    # Target should be 0 or 1
    if TARGET_COLUMN in df.columns:
        unique_target = set(df[TARGET_COLUMN].dropna().unique())
        valid_target = {0, 1}
        if not unique_target.issubset(valid_target):
            errors.append(f"{TARGET_COLUMN} contains invalid values: {unique_target}. Expected {valid_target}")

    if errors:
        for err in errors:
            logger.error(f"Validation error: {err}")
        raise ValueError(f"Data validation failed: {errors}")

    # Log summary statistics
    logger.info("Data validation passed [OK]")
    logger.info(f"  Total records: {len(df)}")
    logger.info(f"  Null values:\n{df.isnull().sum().to_string()}")

    if TARGET_COLUMN in df.columns:
        target_dist = df[TARGET_COLUMN].value_counts()
        logger.info(f"  Target distribution:\n{target_dist.to_string()}")

    return True


def save_raw_data(df: pd.DataFrame, path: Optional[str] = None) -> Path:
    """
    Save DataFrame to the raw data directory.

    Args:
        df: DataFrame to save.
        path: Optional output path. Defaults to RAW_DATA_FILE.

    Returns:
        Path where the file was saved.
    """
    output_path = Path(path) if path else RAW_DATA_FILE
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)
    logger.info(f"Raw data saved to: {output_path} ({len(df)} records)")

    return output_path


def generate_sample_dataset(n_samples: int = 1000) -> pd.DataFrame:
    """
    Generate a realistic synthetic lung cancer survey dataset.

    This mimics the structure of the Kaggle Lung Cancer Prediction Dataset (30 variables).
    Used for development and testing when the real dataset is unavailable.

    Args:
        n_samples: Number of samples to generate.

    Returns:
        DataFrame with synthetic data.
    """
    logger.info(f"Generating synthetic dataset with {n_samples} samples...")
    np.random.seed(42)

    data = {}
    
    # Demographic
    data["age"] = np.random.randint(18, 90, n_samples)
    data["gender"] = np.random.randint(0, 2, n_samples)
    data["education_years"] = np.random.randint(8, 22, n_samples)
    data["income_level"] = np.random.randint(1, 6, n_samples)

    # Smoking
    data["smoker"] = np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])
    data["smoking_years"] = np.zeros(n_samples)
    data["cigarettes_per_day"] = np.zeros(n_samples)
    for i in range(n_samples):
        if data["smoker"][i] == 1:
            max_smoke = max(1, data["age"][i] - 15)
            data["smoking_years"][i] = np.random.randint(1, max_smoke)
            data["cigarettes_per_day"][i] = np.random.randint(1, 40)
    data["pack_years"] = (data["cigarettes_per_day"] / 20) * data["smoking_years"]

    # Environmental/Lifestyle
    data["passive_smoking"] = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
    data["air_pollution_index"] = np.random.uniform(10, 100, n_samples)
    data["occupational_exposure"] = np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])
    data["radon_exposure"] = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])
    data["family_history_cancer"] = np.random.choice([0, 1], size=n_samples, p=[0.85, 0.15])
    
    # Medical
    data["copd"] = np.array([np.random.choice([0, 1], p=[0.9, 0.1]) if p < 10 else np.random.choice([0, 1], p=[0.7, 0.3]) for p in data["pack_years"]])
    data["asthma"] = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])
    data["previous_tb"] = np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05])
    data["chronic_cough"] = np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])
    data["chest_pain"] = np.random.choice([0, 1], size=n_samples, p=[0.85, 0.15])
    data["shortness_of_breath"] = np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])
    data["fatigue"] = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
    
    # Measurements
    data["bmi"] = np.random.uniform(18, 35, n_samples)
    data["oxygen_saturation"] = np.random.uniform(92, 100, n_samples)
    data["fev1_x10"] = np.random.uniform(20, 50, n_samples)
    data["crp_level"] = np.random.uniform(0, 10, n_samples)
    data["xray_abnormal"] = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])
    
    # Habits
    data["exercise_hours_per_week"] = np.random.uniform(0, 15, n_samples)
    data["diet_quality"] = np.random.randint(1, 6, n_samples)
    data["alcohol_units_per_week"] = np.random.uniform(0, 25, n_samples)
    data["healthcare_access"] = np.random.randint(1, 6, n_samples)

    # Target
    risk_score = (
        0.5 * (data["pack_years"] / 50) +
        0.2 * data["family_history_cancer"] +
        0.15 * data["occupational_exposure"] +
        0.1 * (data["air_pollution_index"] / 100) +
        0.1 * data["xray_abnormal"] +
        0.1 * data["copd"]
    )
    prob = 1 / (1 + np.exp(-(risk_score - 0.4) * 10))
    data[TARGET_COLUMN] = np.array([np.random.choice([0, 1], p=[1-p, p]) for p in prob]).astype(int)

    df = pd.DataFrame(data)
    # Reorder columns to match standard dataset
    df = df[REQUIRED_COLUMNS]

    logger.info(f"Generated dataset shape: {df.shape}")
    logger.info(f"Target distribution: {df[TARGET_COLUMN].value_counts().to_dict()}")

    return df


if __name__ == "__main__":
    # Generate and save sample dataset for development
    df = generate_sample_dataset()
    validate_data(df)
    save_raw_data(df)
    print(f"Sample dataset saved to {RAW_DATA_FILE}")
