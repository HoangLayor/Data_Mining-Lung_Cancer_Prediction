"""
SQLAlchemy models for the Lung Cancer Prediction project.
"""

from sqlalchemy import Column, Integer, Float, String, JSON, DateTime, func
from src.utils.database import Base

class Patient(Base):
    """Table for storing patient data from both raw sources and API."""
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    
    # Demographic & Lifestyle
    age = Column(Integer)
    gender = Column(Integer)
    education_years = Column(Integer)
    income_level = Column(Integer)
    smoker = Column(Integer)
    smoking_years = Column(Float)
    cigarettes_per_day = Column(Float)
    pack_years = Column(Float)
    passive_smoking = Column(Integer)
    
    # Environmental & Clinical
    air_pollution_index = Column(Float)
    occupational_exposure = Column(Integer)
    radon_exposure = Column(Integer)
    family_history_cancer = Column(Integer)
    copd = Column(Integer)
    asthma = Column(Integer)
    previous_tb = Column(Integer)
    chronic_cough = Column(Integer)
    chest_pain = Column(Integer)
    shortness_of_breath = Column(Integer)
    fatigue = Column(Integer)
    
    # Measurements
    bmi = Column(Float)
    oxygen_saturation = Column(Float)
    fev1_x10 = Column(Float)
    crp_level = Column(Float)
    xray_abnormal = Column(Integer)
    
    # Habits & Access
    exercise_hours_per_week = Column(Float)
    diet_quality = Column(Integer)
    alcohol_units_per_week = Column(Float)
    healthcare_access = Column(Integer)
    
    # Target
    lung_cancer_risk = Column(Integer, nullable=True)
    
    # Metadata
    source = Column(String, default="raw") # 'raw' or 'api'
    created_at = Column(DateTime, server_default=func.now())

class ModelMetadata(Base):
    """Table for storing model training history and metrics."""
    __tablename__ = "model_metadata"

    version = Column(Integer, primary_key=True)
    date = Column(DateTime, server_default=func.now())
    model_type = Column(String)
    dataset_size = Column(Integer)
    feature_names = Column(JSON)
    metrics = Column(JSON)
    parameters = Column(JSON)
    file_path = Column(String)
