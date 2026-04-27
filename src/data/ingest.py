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
)

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
        raise
    except Exception as e:
        logger.error(f"Error loading CSV: {e}")
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
        "GENDER": np.random.choice(["M", "F"], n_samples),
        "AGE": np.random.randint(20, 85, n_samples),
    }

    for col in BINARY_COLUMNS:
        data[col] = np.random.choice([1, 2], n_samples)

    data[TARGET_COLUMN] = np.random.choice(["YES", "NO"], n_samples, p=[0.4, 0.6])

    df = pd.DataFrame(data)
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

    # AGE should be numeric
    if not pd.api.types.is_numeric_dtype(df["AGE"]):
        errors.append("AGE column must be numeric")

    # GENDER should be string/object
    if df["GENDER"].dtype not in ["object", "string"]:
        valid_genders = df["GENDER"].unique()
        if not all(g in ["M", "F", 1, 0] for g in valid_genders):
            errors.append(f"GENDER contains invalid values: {valid_genders}")

    # Binary columns should contain only 1 or 2
    for col in BINARY_COLUMNS:
        if col in df.columns:
            unique_vals = set(df[col].dropna().unique())
            valid_vals = {1, 2, 0}  # Allow 0 for already-processed data
            if not unique_vals.issubset(valid_vals):
                errors.append(f"{col} contains invalid values: {unique_vals}")

    # Target should be YES/NO or 1/0
    if TARGET_COLUMN in df.columns:
        unique_target = set(df[TARGET_COLUMN].dropna().unique())
        valid_target = {"YES", "NO", 1, 0}
        if not unique_target.issubset(valid_target):
            errors.append(f"{TARGET_COLUMN} contains invalid values: {unique_target}")

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


def generate_sample_dataset(n_samples: int = 309) -> pd.DataFrame:
    """
    Generate a realistic synthetic lung cancer survey dataset.

    This mimics the structure of the Kaggle 'Survey Lung Cancer' dataset.
    Used for development and testing when the real dataset is unavailable.

    Args:
        n_samples: Number of samples to generate.

    Returns:
        DataFrame with synthetic data.
    """
    logger.info(f"Generating synthetic dataset with {n_samples} samples...")
    np.random.seed(42)

    ages = np.random.randint(21, 87, n_samples)
    genders = np.random.choice(["M", "F"], n_samples, p=[0.52, 0.48])

    data = {"GENDER": genders, "AGE": ages}

    # Generate correlated binary features
    # Higher risk factors for older, smoking individuals
    smoking = np.random.choice([1, 2], n_samples, p=[0.35, 0.65])
    data["SMOKING"] = smoking

    for col in BINARY_COLUMNS:
        if col == "SMOKING":
            continue
        # Correlate symptoms slightly with smoking
        p_yes = np.where(smoking == 2, 0.6, 0.35)
        data[col] = np.array([
            np.random.choice([1, 2], p=[1 - p, p]) for p in p_yes
        ])

    # Generate target: correlated with risk factors
    risk_score = (
        (smoking == 2).astype(int) * 2
        + (data["CHRONIC_DISEASE"] == 2).astype(int)
        + (data["COUGHING"] == 2).astype(int)
        + (data["SHORTNESS_OF_BREATH"] == 2).astype(int)
        + (ages > 55).astype(int)
    )
    # Higher risk score → higher probability of lung cancer
    prob_cancer = np.clip(risk_score / 8 + 0.1, 0.05, 0.95)
    target = np.array([
        np.random.choice(["YES", "NO"], p=[p, 1 - p]) for p in prob_cancer
    ])
    data[TARGET_COLUMN] = target

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
