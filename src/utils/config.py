"""
Configuration module for the Lung Cancer Prediction project.

Centralizes all paths, constants, and hyperparameters for reproducibility.
"""

import os
from pathlib import Path

# ──────────────────────────────────────────────
# Project Root
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ──────────────────────────────────────────────
# Directory Paths
# ──────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
PLOTS_DIR = MODELS_DIR / "plots"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, PLOTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# Data Configuration
# ──────────────────────────────────────────────
RAW_DATA_FILE = RAW_DATA_DIR / "survey_lung_cancer.csv"
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "processed_data.csv"
NEW_DATA_FILE = RAW_DATA_DIR / "new_data.csv"

# Required columns in the raw dataset (30 total)
REQUIRED_COLUMNS = [
    "age", "gender", "education_years", "income_level", "smoker",
    "smoking_years", "cigarettes_per_day", "pack_years", "passive_smoking",
    "air_pollution_index", "occupational_exposure", "radon_exposure",
    "family_history_cancer", "copd", "asthma", "previous_tb",
    "chronic_cough", "chest_pain", "shortness_of_breath", "fatigue",
    "bmi", "oxygen_saturation", "fev1_x10", "crp_level", "xray_abnormal",
    "exercise_hours_per_week", "diet_quality", "alcohol_units_per_week",
    "healthcare_access", "lung_cancer_risk"
]

# Target column
TARGET_COLUMN = "lung_cancer_risk"

# Binary feature columns (0=No, 1=Yes)
BINARY_COLUMNS = [
    "gender", "smoker", "passive_smoking", "occupational_exposure",
    "radon_exposure", "family_history_cancer", "copd", "asthma",
    "previous_tb", "chronic_cough", "chest_pain", "shortness_of_breath",
    "fatigue", "xray_abnormal"
]

# Numerical columns for scaling
NUMERICAL_COLUMNS = [
    "age", "education_years", "income_level", "smoking_years",
    "cigarettes_per_day", "pack_years", "air_pollution_index",
    "bmi", "oxygen_saturation", "fev1_x10", "crp_level",
    "exercise_hours_per_week", "diet_quality", "alcohol_units_per_week",
    "healthcare_access"
]

# Categorical columns
CATEGORICAL_COLUMNS = []

# ──────────────────────────────────────────────
# Model Configuration
# ──────────────────────────────────────────────
RANDOM_SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Model file paths
PREPROCESSOR_FILE = MODELS_DIR / "preprocessor.pkl"
BEST_MODEL_FILE = MODELS_DIR / "best_model.pkl"
METRICS_FILE = MODELS_DIR / "metrics.json"
EVALUATION_REPORT_FILE = MODELS_DIR / "evaluation_report.json"
MODEL_METADATA_FILE = MODELS_DIR / "model_metadata.json"

# ──────────────────────────────────────────────
# Hyperparameter Grids
# ──────────────────────────────────────────────
LOGISTIC_REGRESSION_PARAMS = {
    "C": [0.01, 0.1, 1, 10, 100],
    "penalty": ["l2"],
    "solver": ["lbfgs"],
    "max_iter": [1000],
    "class_weight": ["balanced"],
}

RANDOM_FOREST_PARAMS = {
    "n_estimators": [100, 200, 300],
    "max_depth": [5, 10, 15, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "class_weight": ["balanced"],
}

XGBOOST_PARAMS = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.3],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
}

# ──────────────────────────────────────────────
# Logging Configuration
# ──────────────────────────────────────────────
LOG_FILE = LOGS_DIR / "app.log"
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ──────────────────────────────────────────────
# API Configuration
# ──────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000
