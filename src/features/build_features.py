"""
Feature Engineering Module for the Lung Cancer Prediction project.

Creates derived features from the preprocessed dataset to improve
model performance and interpretability.
"""

import pandas as pd
import numpy as np
from typing import List

from src.utils.logger import get_logger
from src.utils.config import BINARY_COLUMNS

logger = get_logger(__name__)


def _create_smoking_impact(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create smoking impact feature.
    Calculates impact as smoker flag × pack_years.
    """
    df = df.copy()
    if "smoker" in df.columns and "pack_years" in df.columns:
        df["smoking_impact"] = df["smoker"] * df["pack_years"]
        logger.info("Created feature: smoking_impact = smoker × pack_years")
    return df


def _create_clinical_severity_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a composite clinical severity score from imaging and symptoms.
    """
    df = df.copy()
    risk_columns = ["chest_pain", "fatigue", "xray_abnormal"]

    available = [col for col in risk_columns if col in df.columns]
    if available:
        df["clinical_severity_score"] = df[available].sum(axis=1)
        logger.info(f"Created feature: clinical_severity_score = sum({available})")
    return df


def _create_symptom_count(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count the total number of positive symptoms for each patient.
    """
    df = df.copy()
    symptom_cols = [col for col in BINARY_COLUMNS if col in df.columns]

    if symptom_cols:
        df["SYMPTOM_COUNT"] = df[symptom_cols].sum(axis=1)
        logger.info(
            f"Created feature: SYMPTOM_COUNT (from {len(symptom_cols)} symptom columns)"
        )
    return df


def _create_lifestyle_health_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a lifestyle health score (protective factors).
    """
    df = df.copy()
    lifestyle_cols = ["exercise_hours_per_week", "diet_quality", "healthcare_access"]
    available = [col for col in lifestyle_cols if col in df.columns]

    if available:
        df["lifestyle_health_score"] = df[available].sum(axis=1)
        logger.info(f"Created feature: lifestyle_health_score = sum({available})")
    return df


def _create_respiratory_condition_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a respiratory condition score from clinical history.
    """
    df = df.copy()
    resp_cols = ["copd", "asthma", "previous_tb", "chronic_cough", "shortness_of_breath"]
    available = [col for col in resp_cols if col in df.columns]

    if available:
        df["respiratory_condition_score"] = df[available].sum(axis=1)
        logger.info(f"Created feature: respiratory_condition_score = sum({available})")
    return df


def _drop_redundant_columns(
    df: pd.DataFrame,
    correlation_threshold: float = 0.95,
) -> pd.DataFrame:
    """
    Drop highly correlated redundant columns.

    Args:
        df: DataFrame with features.
        correlation_threshold: Correlation threshold above which columns are dropped.

    Returns:
        DataFrame with redundant columns removed.
    """
    df = df.copy()
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.shape[1] < 2:
        return df

    corr_matrix = numeric_df.corr().abs()
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    to_drop = [
        col for col in upper_triangle.columns
        if any(upper_triangle[col] > correlation_threshold)
    ]

    if to_drop:
        logger.info(
            f"Dropping {len(to_drop)} redundant columns "
            f"(correlation > {correlation_threshold}): {to_drop}"
        )
        df = df.drop(columns=to_drop)
    else:
        logger.info("No redundant columns detected")

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature engineering pipeline.

    Steps:
        1. Create smoking risk proxy
        2. Create health risk score
        3. Create symptom count
        4. Create lifestyle risk score
        5. Create respiratory score
        6. Drop redundant columns (if highly correlated)

    Args:
        df: Preprocessed DataFrame (features only, no target).

    Returns:
        DataFrame with engineered features added.
    """
    logger.info("=" * 60)
    logger.info("Starting feature engineering pipeline...")
    logger.info(f"Input shape: {df.shape}")
    logger.info(f"Input columns: {list(df.columns)}")

    # Build derived features
    df = _create_smoking_impact(df)
    df = _create_clinical_severity_score(df)
    df = _create_symptom_count(df)
    df = _create_lifestyle_health_score(df)
    df = _create_respiratory_condition_score(df)

    # Drop redundant columns
    df = _drop_redundant_columns(df)

    logger.info(f"Feature engineering complete. Output shape: {df.shape}")
    logger.info(f"Final columns: {list(df.columns)}")
    logger.info("=" * 60)

    return df


if __name__ == "__main__":
    from src.data.ingest import load_csv
    from src.data.preprocess import preprocess_data

    df = load_csv()
    X, y, _ = preprocess_data(df)
    X_featured = build_features(X)
    print(f"Features shape: {X_featured.shape}")
    print(f"New columns: {list(X_featured.columns)}")
