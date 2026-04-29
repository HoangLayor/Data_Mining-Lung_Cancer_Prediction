"""
Database utility module for the Lung Cancer Prediction project.

Handles SQLAlchemy engine creation, session management, and base model definition.
"""

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager

from src.utils.config import DATABASE_URL
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Create engine
engine = create_engine(DATABASE_URL)

# Create SessionLocal class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create Base class for models
Base = declarative_base()

@contextmanager
def get_db():
    """
    Context manager for database sessions.
    Usage:
        with get_db() as db:
            db.query(...)
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """Initialize the database by creating all tables."""
    try:
        logger.info(f"Initializing database at: {DATABASE_URL}")
        Base.metadata.create_all(bind=engine)
        logger.info("Database initialization successful [OK]")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise
