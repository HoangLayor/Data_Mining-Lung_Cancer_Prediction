"""
Data Preprocessing Module for the Lung Cancer Prediction project.

Handles missing values, encoding, scaling, outlier detection,
and builds a reproducible sklearn Pipeline.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Tuple, Optional

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

from src.utils.logger import get_logger
from src.utils.config import (
    BINARY_COLUMNS,
    NUMERICAL_COLUMNS,
    CATEGORICAL_COLUMNS,
    TARGET_COLUMN,
    PREPROCESSOR_FILE,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
)

logger = get_logger(__name__)


def _cap_outliers_iqr(series: pd.Series, factor: float = 1.5) -> pd.Series:
    """
    Cap outliers using the Interquartile Range (IQR) method.

    Args:
        series: Pandas Series to process.
        factor: IQR multiplier for defining outlier bounds.

    Returns:
        Series with outliers capped.
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR

    outliers_count = ((series < lower_bound) | (series > upper_bound)).sum()
    if outliers_count > 0:
        logger.info(
            f"  Capping {outliers_count} outliers in '{series.name}' "
            f"[{lower_bound:.1f}, {upper_bound:.1f}]"
        )

    return series.clip(lower=lower_bound, upper=upper_bound)


def _encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode the target column. 
    Handles both 'YES'/'NO' and already numeric 0/1.
    """
    df = df.copy()
    if TARGET_COLUMN in df.columns:
        # If all values are null (e.g. during prediction), just drop it or leave it as is
        if df[TARGET_COLUMN].isnull().all():
            logger.info(f"Target column '{TARGET_COLUMN}' is null, skipping encoding.")
            return df
            
        if pd.api.types.is_string_dtype(df[TARGET_COLUMN]):
            # Case insensitive mapping
            mapping = {"YES": 1, "NO": 0, "1": 1, "0": 0}
            df[TARGET_COLUMN] = df[TARGET_COLUMN].str.upper().map(mapping)
            logger.info(f"Encoded target '{TARGET_COLUMN}': string -> numeric")
        else:
            # Ensure it's 0/1 if numeric, but handle NaNs safely
            df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors='coerce')
            if not df[TARGET_COLUMN].isnull().any():
                df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
    return df


def _encode_gender(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode gender column.
    Handles 'M'/'F', 'MALE'/'FEMALE', or already numeric 0/1.
    """
    df = df.copy()
    col = "gender" if "gender" in df.columns else "GENDER"
    if col in df.columns:
        if pd.api.types.is_string_dtype(df[col]):
            mapping = {"M": 1, "F": 0, "MALE": 1, "FEMALE": 0, "1": 1, "0": 0}
            df[col] = df[col].str.upper().map(mapping)
            logger.info(f"Encoded {col}: string -> numeric")
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    return df


def _remap_binary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remap binary columns from 1/2 encoding to 0/1 encoding.
    Skips if columns are already 0/1.
    """
    df = df.copy()
    remapped = []
    for col in BINARY_COLUMNS:
        if col in df.columns:
            # Dropna to find unique values
            unique_vals = set(df[col].dropna().unique())
            # If values are {1, 2}, remap to {0, 1}
            if unique_vals.issubset({1, 2}) and unique_vals != {0, 1}:
                df[col] = df[col].map({1: 0, 2: 1})
                remapped.append(col)
    if remapped:
        logger.info(f"Remapped {len(remapped)} binary columns from 1/2 -> 0/1: {remapped}")
    return df


def _handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values: median for numerical, mode for categorical."""
    df = df.copy()
    missing_summary = df.isnull().sum()
    total_missing = missing_summary.sum()

    if total_missing == 0:
        logger.info("No missing values detected")
        return df

    logger.info(f"Missing values detected: {total_missing} total")
    logger.info(f"Missing value summary:\n{missing_summary[missing_summary > 0].to_string()}")

    # Numerical columns: fill with median
    for col in NUMERICAL_COLUMNS:
        if col in df.columns and df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.info(f"  Filled '{col}' missing values with median: {median_val}")

    # Categorical / Binary columns: fill with mode
    cat_binary_cols = CATEGORICAL_COLUMNS + BINARY_COLUMNS
    for col in cat_binary_cols:
        if col in df.columns and df[col].isnull().any():
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            logger.info(f"  Filled '{col}' missing values with mode: {mode_val}")

    return df


def preprocess_data(
    df: pd.DataFrame,
    fit_preprocessor: bool = True,
    preprocessor_path: Optional[str] = None,
    save_output: bool = False
) -> Tuple[pd.DataFrame, Optional[pd.Series], object]:
    """
    Full preprocessing pipeline.

    Steps:
        1. Encode target (YES/NO → 1/0)
        2. Encode GENDER (M/F → 1/0)
        3. Remap binary columns (1/2 → 0/1)
        4. Handle missing values
        5. Cap outliers (IQR on AGE)
        6. Scale numerical features (StandardScaler)

    Args:
        df: Raw DataFrame.
        fit_preprocessor: Whether to fit a new preprocessor or load existing.
        preprocessor_path: Path to an existing preprocessor pickle file.

    Returns:
        Tuple of (X_processed, y, preprocessor).
    """
    logger.info("=" * 60)
    logger.info("Starting data preprocessing pipeline...")
    logger.info(f"Input shape: {df.shape}")

    # Step 1: Encode target
    df = _encode_target(df)

    # Step 2: Encode GENDER
    df = _encode_gender(df)

    # Step 3: Remap binary columns
    df = _remap_binary_columns(df)

    # Step 4: Handle missing values
    df = _handle_missing_values(df)

    # Step 5: Cap outliers on AGE
    logger.info("Detecting outliers using IQR method...")
    for col in NUMERICAL_COLUMNS:
        if col in df.columns:
            df[col] = _cap_outliers_iqr(df[col])

    # Separate features and target
    if TARGET_COLUMN in df.columns:
        y = df[TARGET_COLUMN].copy()
        X = df.drop(columns=[TARGET_COLUMN]).copy()
    else:
        y = None
        X = df.copy()

    # Step 6: Scale numerical features
    if fit_preprocessor:
        logger.info("Fitting StandardScaler on numerical features...")
        scaler = StandardScaler()
        X[NUMERICAL_COLUMNS] = scaler.fit_transform(X[NUMERICAL_COLUMNS])

        # Save preprocessor
        save_preprocessor(scaler, preprocessor_path)
    else:
        # Load existing preprocessor
        load_path = Path(preprocessor_path) if preprocessor_path else PREPROCESSOR_FILE
        logger.info(f"Loading existing preprocessor from: {load_path}")
        scaler = joblib.load(load_path)
        X[NUMERICAL_COLUMNS] = scaler.transform(X[NUMERICAL_COLUMNS])

    logger.info(f"Preprocessing complete. Output shape: X={X.shape}, y={'None' if y is None else y.shape}")
    logger.info(f"Feature columns: {list(X.columns)}")
    if y is not None:
        logger.info(f"Target distribution: {y.value_counts().to_dict()}")
    logger.info("=" * 60)

    # Save processed data only if save_output is True and y is present (training mode)
    if save_output and y is not None:
        processed_df = X.copy()
        processed_df[TARGET_COLUMN] = y
        processed_path = PROCESSED_DATA_DIR / "processed_data.csv"
        processed_df.to_csv(processed_path, index=False)
        logger.info(f"Processed data saved to: {processed_path}")

    return X, y, scaler


def save_preprocessor(preprocessor: object, path: Optional[str] = None) -> Path:
    """
    Save the preprocessing object to disk.

    Args:
        preprocessor: Fitted preprocessor (e.g., StandardScaler).
        path: Output path. Defaults to PREPROCESSOR_FILE.

    Returns:
        Path where the preprocessor was saved.
    """
    output_path = Path(path) if path else PREPROCESSOR_FILE
    output_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(preprocessor, output_path)
    logger.info(f"Preprocessor saved to: {output_path}")

    return output_path


if __name__ == "__main__":
    from src.data.ingest import load_csv

    df = load_csv()
    X, y, preprocessor = preprocess_data(df, save_output=True)
    print(f"Preprocessed data: X={X.shape}, y={y.shape}")
    print(f"Class balance: {y.value_counts().to_dict()}")
