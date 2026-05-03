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
RAW_DATA_FILE = RAW_DATA_DIR / "lung_cancer.csv"
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
# Hyperparameter Optimization (Optuna)
# ──────────────────────────────────────────────
# We now use Optuna for hyperparameter optimization defined in src/models/train.py.
# The previous fixed grids have been removed for flexibility.

# ──────────────────────────────────────────────
# Logging Configuration
# ──────────────────────────────────────────────
LOG_FILE = LOGS_DIR / "app.log"
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ──────────────────────────────────────────────
# API & Database Configuration
# ──────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000

# PostgreSQL Configuration
# Priority: DATABASE_URL env var > Individual POSTGRES_* env vars > Defaults
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "123456")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5433")
POSTGRES_DB = os.getenv("POSTGRES_DB", "lung_cancer_db")

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)

