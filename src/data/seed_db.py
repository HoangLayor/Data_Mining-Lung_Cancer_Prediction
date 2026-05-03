"""
Script to seed the PostgreSQL database with initial data from CSV files.
"""

import pandas as pd
from src.utils.database import init_db, get_db
from src.utils.models import Patient
from src.utils.config import RAW_DATA_FILE
from src.utils.logger import get_logger

logger = get_logger(__name__)

def seed_patients():
    """Load initial data from survey_lung_cancer.csv into Postgres."""
    if not RAW_DATA_FILE.exists():
        logger.error(f"Raw data file not found: {RAW_DATA_FILE}")
        return

    logger.info(f"Reading data from {RAW_DATA_FILE}...")
    df = pd.read_csv(RAW_DATA_FILE)
    
    # Ensure columns match (assuming columns are already lowercase in the CSV or handled by preprocess)
    # If the CSV has different column names, we'd need to map them here.
    # Our recent ingest.py generates correct columns.
    
    records = df.to_dict(orient="records")
    
    with get_db() as db:
        logger.info(f"Seeding {len(records)} records into 'patients' table...")
        for rec in records:
            patient = Patient(**rec, source="raw")
            db.add(patient)
        db.commit()
    
    logger.info("Database seeding complete [OK]")

if __name__ == "__main__":
    init_db()
    seed_patients()
