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
    "GENDER": "M",
    "AGE": 65,
    "SMOKING": 2,
    "YELLOW_FINGERS": 2,
    "ANXIETY": 1,
    "PEER_PRESSURE": 1,
    "CHRONIC_DISEASE": 2,
    "FATIGUE": 2,
    "ALLERGY": 1,
    "WHEEZING": 2,
    "ALCOHOL_CONSUMING": 2,
    "COUGHING": 2,
    "SHORTNESS_OF_BREATH": 2,
    "SWALLOWING_DIFFICULTY": 1,
    "CHEST_PAIN": 2
  }'
```

Response:
```json
{
  "prediction": 1,
  "probability": 0.8523,
  "risk_level": "High",
  "label": "Lung Cancer Detected"
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

Based on the [Kaggle Survey Lung Cancer](https://www.kaggle.com/datasets/mysarahmadbhat/lung-cancer) dataset with 16 features:

| Feature | Description |
|---------|-------------|
| GENDER | M / F |
| AGE | Patient age |
| SMOKING | 1=No, 2=Yes |
| YELLOW_FINGERS | 1=No, 2=Yes |
| ANXIETY | 1=No, 2=Yes |
| PEER_PRESSURE | 1=No, 2=Yes |
| CHRONIC_DISEASE | 1=No, 2=Yes |
| FATIGUE | 1=No, 2=Yes |
| ALLERGY | 1=No, 2=Yes |
| WHEEZING | 1=No, 2=Yes |
| ALCOHOL_CONSUMING | 1=No, 2=Yes |
| COUGHING | 1=No, 2=Yes |
| SHORTNESS_OF_BREATH | 1=No, 2=Yes |
| SWALLOWING_DIFFICULTY | 1=No, 2=Yes |
| CHEST_PAIN | 1=No, 2=Yes |
| LUNG_CANCER | YES / NO (target) |

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. It should **not** be used for clinical decision-making or as a substitute for professional medical advice.

---

## 📄 License

MIT License
