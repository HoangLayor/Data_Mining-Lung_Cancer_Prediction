import os
import sys
from datetime import datetime
import pandas as pd
import logging

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

# Ensure project root is in python path
for path in ['/opt/airflow', os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))]:
    if os.path.exists(path) and path not in sys.path:
        sys.path.insert(0, path)

from src.data.ingest import load_from_db
from src.data.preprocess import preprocess_data
from src.features.build_features import build_features
from src.utils.database import get_db
from src.utils.config import TARGET_COLUMN

logger = logging.getLogger(__name__)

def preprocess_and_save_to_db():
    logger.info("Loading raw data from database...")
    df = load_from_db()
    
    if df.empty:
        logger.warning("No data found in database. Skipping preprocessing.")
        return
        
    logger.info("Preprocessing data...")
    X, y, preprocessor = preprocess_data(df, fit_preprocessor=True)
    X_featured = build_features(X)
    
    logger.info("Combining features and target...")
    processed_df = X_featured.copy()
    processed_df[TARGET_COLUMN] = y
    
    logger.info("Saving preprocessed data to 'preprocessed_patients' table...")
    try:
        with get_db() as db:
            processed_df.to_sql("preprocessed_patients", db.bind, if_exists="replace", index=False)
            logger.info(f"Successfully saved {len(processed_df)} preprocessed records to database.")
    except Exception as e:
        logger.error(f"Error saving preprocessed data: {e}")
        raise e

with DAG(
    dag_id="preprocess_data_dag",
    start_date=datetime(2025, 1, 1),
    schedule_interval=None, # Triggered externally
    catchup=False,
    tags=["preprocessing", "postgres"]
) as dag:
    
    preprocess_task = PythonOperator(
        task_id="preprocess_and_save",
        python_callable=preprocess_and_save_to_db
    )
    
    # Triggers for the 3 training models
    trigger_xgboost = TriggerDagRunOperator(
        task_id="trigger_xgboost_training",
        trigger_dag_id="xgboost_training",
        wait_for_completion=True,
        poke_interval=30
    )
    
    trigger_rf = TriggerDagRunOperator(
        task_id="trigger_random_forest_training",
        trigger_dag_id="random_forest_training",
        wait_for_completion=True,
        poke_interval=30
    )
    
    trigger_lr = TriggerDagRunOperator(
        task_id="trigger_logistic_regression_training",
        trigger_dag_id="logistic_regression_training",
        wait_for_completion=True,
        poke_interval=30
    )
    
    trigger_promotion = TriggerDagRunOperator(
        task_id="trigger_model_promotion",
        trigger_dag_id="promote_best_model",
        wait_for_completion=False
    )
    
    preprocess_task >> [trigger_xgboost, trigger_rf, trigger_lr] >> trigger_promotion
