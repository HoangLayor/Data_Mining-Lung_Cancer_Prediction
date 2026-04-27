# Lung Cancer Risk Prediction

A production-ready machine learning pipeline for predicting lung cancer risk based on patient survey data. Built with Python, scikit-learn, XGBoost, FastAPI, and Apache Airflow.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Pipeline Stages](#pipeline-stages)
- [API Usage](#api-usage)
- [Model Performance](#model-performance)
- [Airflow Orchestration](#airflow-orchestration)
- [Testing](#testing)
- [Docker Deployment](#docker-deployment)

---

## 🎯 Overview

This project implements an end-to-end ML pipeline for lung cancer risk prediction using survey-based patient data. The system includes:

- **Data Pipeline**: Ingestion, validation, preprocessing, and feature engineering
- **Model Training**: Logistic Regression, Random Forest, and XGBoost with hyperparameter tuning
- **Evaluation**: Comprehensive metrics, ROC/PR curves, threshold optimization
- **Explainability**: SHAP-based global and local feature importance
- **API**: FastAPI REST endpoints for real-time predictions
- **Orchestration**: Airflow DAG for automated retraining
- **Versioning**: File-based model versioning with metadata tracking

---

## 🏗️ Architecture

```
┌─────────────┐    ┌──────────────┐    ┌─────────────────┐
│  Raw Data   │───▶│ Preprocessing│───▶│   Feature Eng.  │
│  (CSV/API)  │    │  & Cleaning  │    │  (Derived Feats)│
└─────────────┘    └──────────────┘    └────────┬────────┘
                                                │
                   ┌──────────────┐    ┌────────▼────────┐
                   │  Evaluation  │◀───│  Model Training │
                   │  & Plotting  │    │ (LR/RF/XGBoost) │
                   └──────┬───────┘    └─────────────────┘
                          │
              ┌───────────▼───────────┐
              │   FastAPI Serving     │
              │   POST /predict       │
              └───────────────────────┘
```

---

## 📁 Project Structure

```
project/
├── data/
│   ├── raw/                    # Raw CSV datasets
│   └── processed/              # Cleaned & transformed data
├── src/
│   ├── data/
│   │   ├── ingest.py           # Data loading & validation
│   │   └── preprocess.py       # Cleaning, encoding, scaling
│   ├── features/
│   │   └── build_features.py   # Feature engineering
│   ├── models/
│   │   ├── train.py            # Model training pipeline
│   │   ├── evaluate.py         # Evaluation & visualization
│   │   ├── predict.py          # Inference module
│   │   └── explain.py          # SHAP explainability
│   └── utils/
│       ├── config.py           # Central configuration
│       └── logger.py           # Logging utility
├── api/
│   └── main.py                 # FastAPI application
├── airflow/dags/
│   └── retrain_pipeline.py     # Airflow retraining DAG
├── models/                     # Saved model artifacts
├── logs/                       # Application logs
├── tests/                      # pytest test suite
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Sample Data & Train

```bash
# Generate synthetic dataset
python -m src.data.ingest

# Run full pipeline: preprocess → features → train → evaluate
python -m src.models.train
python -m src.models.evaluate
```

### 3. Start API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Make a Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 62,
    "gender": 1,
    "education_years": 16,
    "income_level": 3,
    "smoker": 1,
    "smoking_years": 25.0,
    "cigarettes_per_day": 20.0,
    "pack_years": 25.0,
    "passive_smoking": 0,
    "air_pollution_index": 45.0,
    "occupational_exposure": 1,
    "radon_exposure": 0,
    "family_history_cancer": 1,
    "copd": 0,
    "asthma": 0,
    "previous_tb": 0,
    "chronic_cough": 1,
    "chest_pain": 1,
    "shortness_of_breath": 0,
    "fatigue": 1,
    "bmi": 26.5,
    "oxygen_saturation": 98.0,
    "fev1_x10": 38.0,
    "crp_level": 1.5,
    "xray_abnormal": 0,
    "exercise_hours_per_week": 3.0,
    "diet_quality": 4,
    "alcohol_units_per_week": 5.0,
    "healthcare_access": 4
  }'
```

Response:
```json
{
  "lung_cancer_risk": 1,
  "probability": 0.9234,
  "risk_level": "High"
}
```

---

## 🔄 Pipeline Stages

| Stage | Module | Description |
|-------|--------|-------------|
| 1. Ingest | `src/data/ingest.py` | Load CSV/API data, validate schema |
| 2. Preprocess | `src/data/preprocess.py` | Handle missing values, encode, scale, cap outliers |
| 3. Features | `src/features/build_features.py` | Create derived features (smoking risk, health score, etc.) |
| 4. Train | `src/models/train.py` | Train 3 models, hyperparameter tuning, select best |
| 5. Evaluate | `src/models/evaluate.py` | Metrics, plots, threshold optimization |
| 6. Explain | `src/models/explain.py` | SHAP global & local explanations |
| 7. Predict | `src/models/predict.py` | Load model, preprocess input, return prediction |

---

## 📊 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `GET` | `/health` | Detailed health status |
| `POST` | `/predict` | Predict lung cancer risk |
| `POST` | `/update-data` | Submit new data for retraining |
| `POST` | `/reload-model` | Reload model from disk |
| `GET` | `/docs` | Swagger UI documentation |

---

## 📈 Model Performance

Models are evaluated using:
- **Recall** (priority — minimize false negatives)
- Precision
- F1-Score
- ROC-AUC

The best model is selected by **recall** to minimize missed diagnoses.

Evaluation outputs include:
- Confusion matrix heatmap
- ROC curve
- Precision-Recall curve
- Threshold analysis plot
- SHAP feature importance

---

## ⚙️ Airflow Orchestration

The retraining DAG (`lung_cancer_retrain`) runs weekly:

```
ingest_data → preprocess_data → build_features → train_model → evaluate_model → save_model
```

Configuration:
- Schedule: `@weekly`
- Retries: 3
- Retry delay: 5 minutes
- Email on failure: configurable

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test module
pytest tests/test_ingest.py -v
pytest tests/test_preprocess.py -v
pytest tests/test_predict.py -v
```

---

## 🐳 Docker Deployment

```bash
# Build image
docker build -t lung-cancer-prediction .

# Run container
docker run -p 8000:8000 lung-cancer-prediction
```

---

## 📝 Dataset

Based on the [Kaggle Lung Cancer Prediction Dataset](https://www.kaggle.com/datasets/dhrubangtalukdar/lung-cancer-prediction-dataset) with 30 features:

| Variable | Description |
|---|---|
| age | Age of the individual in years |
| gender | 0 = Female, 1 = Male |
| education_years | Total years of formal education |
| income_level | 1 = lowest, 5 = highest |
| smoker | 0 = No, 1 = Yes |
| smoking_years | Total number of years smoked |
| cigarettes_per_day | Average cigarettes per day |
| pack_years | Cumulative smoking exposure |
| passive_smoking | Exposure to secondhand smoke (0/1) |
| air_pollution_index | Air quality exposure index |
| occupational_exposure | Hazardous substance exposure at work (0/1) |
| radon_exposure | History of radon exposure (0/1) |
| family_history_cancer | Family history of cancer (0/1) |
| copd | Diagnosis of COPD (0/1) |
| asthma | History of asthma (0/1) |
| previous_tb | History of tuberculosis (0/1) |
| chronic_cough | Long-term cough symptoms (0/1) |
| chest_pain | Reports of chest pain (0/1) |
| shortness_of_breath | Breathing difficulty (0/1) |
| fatigue | Persistent fatigue symptoms (0/1) |
| bmi | Body mass index |
| oxygen_saturation | Blood oxygen saturation level (%) |
| fev1_x10 | Lung function measure (FEV1) |
| crp_level | C-reactive protein level (inflammation) |
| xray_abnormal | Abnormal imaging findings (0/1) |
| exercise_hours_per_week | Weekly physical activity duration |
| diet_quality | Overall dietary quality (1-5) |
| alcohol_units_per_week | Average alcohol consumption per week |
| healthcare_access | Access to healthcare services (1-5) |
| lung_cancer_risk | **Target** (0 = No, 1 = Yes) |

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. It should **not** be used for clinical decision-making or as a substitute for professional medical advice.

---

## 📄 License

MIT License
