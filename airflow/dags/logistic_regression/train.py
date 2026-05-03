import os
import sys
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, average_precision_score
from imblearn.over_sampling import SMOTE
import optuna
import numpy as np

# Add project root to python path since it's mounted
sys.path.append("/training/code")

from src.data.ingest import load_preprocessed_from_db
from src.utils.config import TARGET_COLUMN

def objective(trial, X_train, y_train, X_test, y_test):
    params = {
        "C": trial.suggest_float("C", 1e-4, 10.0, log=True),
        "penalty": trial.suggest_categorical("penalty", ["l2"]),
        "solver": trial.suggest_categorical("solver", ["lbfgs", "newton-cg"]),
        "max_iter": 1000,
        "class_weight": "balanced",
        "random_state": 42
    }
    model = LogisticRegression(**params)
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
    
    with mlflow.start_run(run_name="logistic_regression_optuna"):
        mlflow.set_tag("model_type", "logistic_regression")
        mlflow.log_params(best_params)
        mlflow.log_metric("best_recall", study.best_value)
        
        print("Training final Logistic Regression model with best params...")
        model = LogisticRegression(**best_params, max_iter=1000, class_weight="balanced", random_state=42)
        model.fit(X_train, y_train)
        
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
