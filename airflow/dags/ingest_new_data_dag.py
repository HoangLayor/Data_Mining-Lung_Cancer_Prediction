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

from src.utils.config import NEW_DATA_FILE
from src.utils.database import get_db
from src.utils.models import Patient

logger = logging.getLogger(__name__)

def ingest_new_data_to_db():
    """
    Reads the new data CSV file and inserts the records into the Postgres database.
    After successful ingestion, the file is removed to prevent duplicate processing.
    """
    if not os.path.exists(NEW_DATA_FILE):
        logger.info(f"File {NEW_DATA_FILE} not found. Nothing to ingest.")
        return
        
    df = pd.read_csv(NEW_DATA_FILE)
    if df.empty:
        logger.info("CSV is empty.")
        return
        
    logger.info(f"Ingesting {len(df)} records into the database.")
    
    records = df.to_dict(orient="records")
    
    try:
        with get_db() as db:
            for rec in records:
                patient = Patient(**rec, source="api_csv")
                db.add(patient)
            db.commit()
            logger.info("Database ingestion complete [OK]")
    except Exception as e:
        logger.error(f"Error inserting records into database: {e}")
        raise e
    
    # After successful ingestion, remove the file
    os.remove(NEW_DATA_FILE)
    logger.info(f"Ingestion completed and file {NEW_DATA_FILE} removed.")

with DAG(
    dag_id="ingest_new_data_to_db",
    start_date=datetime(2025, 1, 1),
    schedule_interval="@monthly",
    catchup=False,
    tags=["ingestion", "postgres"]
) as dag:
    
    ingest_task = PythonOperator(
        task_id="ingest_new_data",
        python_callable=ingest_new_data_to_db
    )
    
    trigger_preprocess = TriggerDagRunOperator(
        task_id="trigger_preprocess",
        trigger_dag_id="preprocess_data_dag",
        wait_for_completion=False
    )
    
    ingest_task >> trigger_preprocess
