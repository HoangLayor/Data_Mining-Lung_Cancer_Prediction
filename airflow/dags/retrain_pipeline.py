"""
Airflow DAG for Lung Cancer Model Retraining Pipeline.

DAG: lung_cancer_retrain
Schedule: Weekly (every Sunday at 2:00 AM)
Pipeline: ingest → preprocess → feature → train → evaluate → save

This DAG orchestrates the full retraining pipeline for the
lung cancer risk prediction model.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

# ──────────────────────────────────────────────
# Add project root to Python path
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ──────────────────────────────────────────────
# Default DAG arguments
# ──────────────────────────────────────────────
default_args = {
    "owner": "data-engineering",
    "depends_on_past": False,
    "email": ["admin@example.com"],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
    "start_date": datetime(2025, 1, 1),
}


# ──────────────────────────────────────────────
# Task Functions
# ──────────────────────────────────────────────
    from src.data.ingest import load_from_db, validate_data, generate_sample_dataset, save_raw_data
    from src.utils.logger import get_logger
    from src.utils.database import init_db
    from src.utils.models import Patient
    from src.utils.config import RAW_DATA_FILE

    logger = get_logger("airflow.ingest")
    logger.info("=" * 50)
    logger.info("AIRFLOW TASK: ingest_data — STARTED")

    try:
        # Initialize tables if they don't exist
        init_db()
        
        df = load_from_db()
        
        if len(df) == 0:
            logger.info("No data found in database. Generating sample dataset...")
            df = generate_sample_dataset()
            # In a real scenario, we'd insert this into the DB
            # For this pipeline, we'll just validate it
            
        validate_data(df)
        logger.info(f"Ingestion complete: {len(df)} records loaded from database")
        logger.info("AIRFLOW TASK: ingest_data — SUCCESS")

        return True
    except Exception as e:
        logger.error(f"AIRFLOW TASK: ingest_data — FAILED: {e}")
        raise


def task_preprocess_data(**kwargs):
    """Task 2: Preprocess the ingested data."""
    from src.data.ingest import load_from_db
    from src.data.preprocess import preprocess_data
    from src.utils.logger import get_logger

    logger = get_logger("airflow.preprocess")
    logger.info("=" * 50)
    logger.info("AIRFLOW TASK: preprocess_data — STARTED")

    try:
        df = load_from_db()
        X, y, preprocessor = preprocess_data(df)

        logger.info(f"Preprocessing complete: X={X.shape}, y={y.shape}")
        logger.info("AIRFLOW TASK: preprocess_data — SUCCESS")
        return True
    except Exception as e:
        logger.error(f"AIRFLOW TASK: preprocess_data — FAILED: {e}")
        raise


def task_build_features(**kwargs):
    """Task 3: Build engineered features."""
    from src.data.ingest import load_from_db
    from src.data.preprocess import preprocess_data
    from src.features.build_features import build_features
    from src.utils.logger import get_logger

    logger = get_logger("airflow.features")
    logger.info("=" * 50)
    logger.info("AIRFLOW TASK: build_features — STARTED")

    try:
        df = load_from_db()
        X, y, _ = preprocess_data(df, fit_preprocessor=False)
        X_featured = build_features(X)

        logger.info(f"Feature engineering complete: {X_featured.shape}")
        logger.info("AIRFLOW TASK: build_features — SUCCESS")
        return True
    except Exception as e:
        logger.error(f"AIRFLOW TASK: build_features — FAILED: {e}")
        raise


def task_train_model(**kwargs):
    """Task 4: Train models and select the best."""
    from sklearn.model_selection import train_test_split
    from src.data.ingest import load_csv
    from src.data.preprocess import preprocess_data
    from src.features.build_features import build_features
    from src.models.train import train_model, save_model
    from src.utils.logger import get_logger
    from src.utils.config import TEST_SIZE, RANDOM_SEED

    logger = get_logger("airflow.train")
    logger.info("=" * 50)
    logger.info("AIRFLOW TASK: train_model — STARTED")

    try:
        # Full pipeline
        df = load_from_db()
        X, y, _ = preprocess_data(df)
        X = build_features(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
        )

        best_model, results = train_model(X_train, y_train)
        model_path = save_model(best_model, results, len(X_train))

        logger.info(f"Training complete. Best model: {results['best_model']}")
        logger.info("AIRFLOW TASK: train_model — SUCCESS")

        # Push model path for downstream tasks
        kwargs["ti"].xcom_push(key="model_path", value=str(model_path))
        return True
    except Exception as e:
        logger.error(f"AIRFLOW TASK: train_model — FAILED: {e}")
        raise


def task_evaluate_model(**kwargs):
    """Task 5: Evaluate the trained model."""
    import joblib
    from sklearn.model_selection import train_test_split
    from src.data.ingest import load_csv
    from src.data.preprocess import preprocess_data
    from src.features.build_features import build_features
    from src.models.evaluate import evaluate_model, plot_metrics, find_optimal_threshold
    from src.utils.logger import get_logger
    from src.utils.config import TEST_SIZE, RANDOM_SEED, BEST_MODEL_FILE

    logger = get_logger("airflow.evaluate")
    logger.info("=" * 50)
    logger.info("AIRFLOW TASK: evaluate_model — STARTED")

    try:
        # Load data and model
        df = load_from_db()
        X, y, _ = preprocess_data(df, fit_preprocessor=False)
        X = build_features(X)

        _, X_test, _, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
        )

        model = joblib.load(BEST_MODEL_FILE)

        # Evaluate
        metrics = evaluate_model(model, X_test, y_test)
        threshold, opt_metrics = find_optimal_threshold(model, X_test, y_test)
        plot_metrics(model, X_test, y_test)

        logger.info(f"Evaluation complete. ROC-AUC: {metrics['roc_auc']:.4f}")
        logger.info(f"Optimal threshold: {threshold:.2f}")
        logger.info("AIRFLOW TASK: evaluate_model — SUCCESS")
        return True
    except Exception as e:
        logger.error(f"AIRFLOW TASK: evaluate_model — FAILED: {e}")
        raise


def task_save_model(**kwargs):
    """Task 6: Final model save and cleanup."""
    from src.utils.logger import get_logger
    from src.utils.config import BEST_MODEL_FILE, MODELS_DIR

    logger = get_logger("airflow.save")
    logger.info("=" * 50)
    logger.info("AIRFLOW TASK: save_model — STARTED")

    try:
        # Verify model exists
        if not BEST_MODEL_FILE.exists():
            raise FileNotFoundError(f"Best model not found at {BEST_MODEL_FILE}")

        model_size = BEST_MODEL_FILE.stat().st_size / 1024  # KB
        logger.info(f"Model file verified: {BEST_MODEL_FILE} ({model_size:.1f} KB)")

        # List all model versions
        versions = sorted(MODELS_DIR.glob("model_v*.pkl"))
        logger.info(f"Total model versions: {len(versions)}")
        for v in versions:
            logger.info(f"  {v.name} ({v.stat().st_size / 1024:.1f} KB)")

        logger.info("AIRFLOW TASK: save_model — SUCCESS")
        logger.info("=" * 50)
        logger.info(">>> RETRAINING PIPELINE COMPLETED SUCCESSFULLY")
        return True
    except Exception as e:
        logger.error(f"AIRFLOW TASK: save_model — FAILED: {e}")
        raise


# ──────────────────────────────────────────────
# DAG Definition
# ──────────────────────────────────────────────
with DAG(
    dag_id="lung_cancer_retrain",
    default_args=default_args,
    description="Lung Cancer Model Retraining Pipeline",
    schedule_interval="@weekly",
    catchup=False,
    tags=["ml", "lung-cancer", "retraining"],
) as dag:

    t1_ingest = PythonOperator(
        task_id="ingest_data",
        python_callable=task_ingest_data,
        provide_context=True,
    )

    t2_preprocess = PythonOperator(
        task_id="preprocess_data",
        python_callable=task_preprocess_data,
        provide_context=True,
    )

    t3_features = PythonOperator(
        task_id="build_features",
        python_callable=task_build_features,
        provide_context=True,
    )

    t4_train = PythonOperator(
        task_id="train_model",
        python_callable=task_train_model,
        provide_context=True,
    )

    t5_evaluate = PythonOperator(
        task_id="evaluate_model",
        python_callable=task_evaluate_model,
        provide_context=True,
    )

    t6_save = PythonOperator(
        task_id="save_model",
        python_callable=task_save_model,
        provide_context=True,
    )

    # Pipeline: ingest → preprocess → feature → train → evaluate → save
    t1_ingest >> t2_preprocess >> t3_features >> t4_train >> t5_evaluate >> t6_save
