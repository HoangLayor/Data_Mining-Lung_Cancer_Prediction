import os
import sys
import mlflow
import mlflow.sklearn
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, average_precision_score
from imblearn.over_sampling import SMOTE
import optuna
import numpy as np

sys.path.append("/training/code")

from src.data.ingest import load_preprocessed_from_db
from src.utils.config import TARGET_COLUMN

def objective(trial, X_train, y_train, X_test, y_test):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "eval_metric": "logloss",
        "random_state": 42
    }
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return recall_score(y_test, y_pred)

def main():
    print("Loading preprocessed data from DB...")
    df = load_preprocessed_from_db()
    
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    
    # 1. Split first to avoid leakage
    X_train_raw, X_test, y_train_raw, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 2. Apply SMOTE only on training data
    print("Applying SMOTE to training set...")
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train_raw, y_train_raw)
    
    print("Starting MLflow tracking...")
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    mlflow.set_experiment("Lung_Cancer_Prediction")
    
    print("Optimizing hyperparameters with Optuna...")
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test), n_trials=20)
    
    best_params = study.best_params
    print(f"Best params: {best_params}")
    
    with mlflow.start_run(run_name="xgboost_optuna"):
        mlflow.set_tag("model_type", "xgboost")
        mlflow.log_params(best_params)
        mlflow.log_metric("best_recall", study.best_value)
        
        print("Training final XGBoost model with best params...")
        model = XGBClassifier(**best_params, eval_metric="logloss", random_state=42)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "roc_auc": float(roc_auc_score(y_test, y_prob)),
            "f1_score": float(f1_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred)),
            "pr_auc": float(average_precision_score(y_test, y_prob))
        }
        
        print(f"Final Metrics: {metrics}")
        mlflow.log_metrics(metrics)
        
        # Log model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="lung_cancer_model"
        )
        print("Training completed and model registered.")

if __name__ == "__main__":
    main()
