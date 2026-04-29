"""
Model Training Module for the Lung Cancer Prediction project.

Trains multiple models with hyperparameter tuning, handles class imbalance,
and selects the best model based on recall.
"""

import json
import time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, Optional

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    StratifiedKFold,
    RandomizedSearchCV,
    cross_val_score,
)
from sklearn.metrics import (
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    make_scorer,
)
import optuna
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from src.utils.logger import get_logger
from src.utils.config import (
    RANDOM_SEED,
    CV_FOLDS,
    MODELS_DIR,
    METRICS_FILE,
    BEST_MODEL_FILE,
    MODEL_METADATA_FILE,
    DATABASE_URL,
)
from src.utils.database import get_db
from src.utils.models import ModelMetadata

logger = get_logger(__name__)


def _get_models() -> Dict[str, Any]:
    """Return dictionary of model instances."""
    return {
        "LogisticRegression": LogisticRegression(
            random_state=RANDOM_SEED,
            max_iter=1000,
            class_weight="balanced",
        ),
        "RandomForest": RandomForestClassifier(
            random_state=RANDOM_SEED,
            class_weight="balanced",
        ),
        "XGBoost": XGBClassifier(
            random_state=RANDOM_SEED,
            eval_metric="logloss",
            use_label_encoder=False,
        ),
    }


def _get_objective(model_name: str, X: pd.DataFrame, y: pd.Series, use_smote: bool = True):
    """
    Create an objective function for Optuna.
    
    Args:
        model_name: Name of the model.
        X: Feature matrix.
        y: Target vector.
        use_smote: Whether to apply SMOTE within CV.
        
    Returns:
        Objective function for Optuna.
    """
    def objective(trial):
        if model_name == "LogisticRegression":
            params = {
                "C": trial.suggest_float("C", 0.001, 100.0, log=True),
                "penalty": "l2",
                "solver": "lbfgs",
                "max_iter": 1000,
                "class_weight": "balanced",
            }
            model = LogisticRegression(**params, random_state=RANDOM_SEED)
        
        elif model_name == "RandomForest":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 400),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "class_weight": "balanced",
            }
            model = RandomForestClassifier(**params, random_state=RANDOM_SEED)
        
        elif model_name == "XGBoost":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 400),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "gamma": trial.suggest_float("gamma", 0, 5),
                "eval_metric": "logloss",
            }
            model = XGBClassifier(**params, random_state=RANDOM_SEED)
        
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # Use pipeline if SMOTE is requested
        if use_smote:
            pipeline = ImbPipeline([
                ("smote", SMOTE(random_state=RANDOM_SEED)),
                ("model", model)
            ])
            eval_obj = pipeline
        else:
            eval_obj = model

        # Cross-validation
        skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        # We optimize for recall
        scores = cross_val_score(eval_obj, X, y, cv=skf, scoring="recall", n_jobs=-1)
        return scores.mean()

    return objective


def cross_validate_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int = CV_FOLDS,
    use_smote: bool = True,
) -> Dict[str, float]:
    """
    Perform Stratified K-Fold Cross Validation.

    Args:
        model: Sklearn-compatible model.
        X: Feature matrix.
        y: Target vector.
        cv_folds: Number of folds.
        use_smote: Whether to apply SMOTE within each fold.

    Returns:
        Dictionary with mean metrics across folds.
    """
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED)

    # Use pipeline if SMOTE is requested to avoid data leakage
    if use_smote:
        pipeline = ImbPipeline([
            ("smote", SMOTE(random_state=RANDOM_SEED)),
            ("model", model)
        ])
        eval_obj = pipeline
    else:
        eval_obj = model

    scorers = {
        "recall": make_scorer(recall_score),
        "precision": make_scorer(precision_score),
        "f1": make_scorer(f1_score),
        "roc_auc": make_scorer(roc_auc_score),
    }

    results = {}
    for metric_name, scorer in scorers.items():
        # Note: cross_val_score will use the pipeline (SMOTE) only on the training folds
        scores = cross_val_score(eval_obj, X, y, cv=skf, scoring=scorer, n_jobs=-1)
        results[metric_name] = {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
        }

    return results


def _apply_smote(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """Apply SMOTE to handle class imbalance."""
    logger.info("Applying SMOTE for class imbalance...")
    logger.info(f"  Before SMOTE: {y.value_counts().to_dict()}")

    smote = SMOTE(random_state=RANDOM_SEED)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    logger.info(f"  After SMOTE: {pd.Series(y_resampled).value_counts().to_dict()}")

    return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled)


def _get_next_model_version() -> int:
    """Determine the next model version number."""
    existing = list(MODELS_DIR.glob("model_v*.pkl"))
    if not existing:
        return 1
    versions = []
    for p in existing:
        try:
            v = int(p.stem.replace("model_v", ""))
            versions.append(v)
        except ValueError:
            continue
    return max(versions) + 1 if versions else 1


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    use_smote: bool = True,
    n_iter: int = 20,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Train multiple models with hyperparameter tuning and select the best.

    Args:
        X_train: Training feature matrix.
        y_train: Training target vector.
        use_smote: Whether to apply SMOTE before training.
        n_iter: Number of iterations for RandomizedSearchCV.

    Returns:
        Tuple of (best_model, all_results_dict).
    """
    logger.info("=" * 60)
    logger.info("Starting model training pipeline...")
    logger.info(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
    logger.info(f"Class distribution: {y_train.value_counts().to_dict()}")

    models = _get_models()
    
    all_results = {}
    best_model = None
    best_recall = 0.0
    best_model_name = ""

    # Set optuna logging level
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    for name, model in models.items():
        logger.info("-" * 40)
        logger.info(f"Optimizing: {name} (using Optuna)")
        start_time = time.time()

        try:
            # Create study
            study = optuna.create_study(direction="maximize")
            objective_func = _get_objective(name, X_train, y_train, use_smote=use_smote)
            
            # Run optimization
            study.optimize(objective_func, n_trials=n_iter)
            
            # Get best parameters
            best_params = study.best_params
            
            # Re-train best model
            if name == "LogisticRegression":
                best_params.update({"penalty": "l2", "solver": "lbfgs", "max_iter": 1000, "class_weight": "balanced"})
                tuned_model = LogisticRegression(**best_params, random_state=RANDOM_SEED)
            elif name == "RandomForest":
                best_params.update({"class_weight": "balanced"})
                tuned_model = RandomForestClassifier(**best_params, random_state=RANDOM_SEED)
            elif name == "XGBoost":
                best_params.update({"eval_metric": "logloss"})
                tuned_model = XGBClassifier(**best_params, random_state=RANDOM_SEED)
            
            # Re-train best model on full training set (with SMOTE if requested)
            if use_smote:
                smote = SMOTE(random_state=RANDOM_SEED)
                X_train_final, y_train_final = smote.fit_resample(X_train, y_train)
            else:
                X_train_final, y_train_final = X_train, y_train

            tuned_model.fit(X_train_final, y_train_final)
            
            # Evaluate best model with full CV using pipeline to avoid leakage
            cv_results = cross_validate_model(tuned_model, X_train, y_train, use_smote=use_smote)

            elapsed = time.time() - start_time

            result = {
                "best_params": best_params,
                "cv_results": cv_results,
                "training_time_seconds": round(elapsed, 2),
            }
            all_results[name] = result

            recall_mean = cv_results["recall"]["mean"]
            logger.info(f"  Best params: {best_params}")
            logger.info(f"  Recall:    {recall_mean:.4f} ± {cv_results['recall']['std']:.4f}")
            logger.info(f"  Precision: {cv_results['precision']['mean']:.4f}")
            logger.info(f"  F1:        {cv_results['f1']['mean']:.4f}")
            logger.info(f"  ROC-AUC:   {cv_results['roc_auc']['mean']:.4f}")
            logger.info(f"  Time:      {elapsed:.2f}s")

            # Track best model (by recall)
            if recall_mean > best_recall:
                best_recall = recall_mean
                best_model = tuned_model
                best_model_name = name

        except Exception as e:
            logger.error(f"Error optimizing {name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            all_results[name] = {"error": str(e)}

    logger.info("=" * 60)
    logger.info(f">>> Best model: {best_model_name} (Recall: {best_recall:.4f})")

    # Save results
    all_results["best_model"] = best_model_name
    all_results["best_recall"] = best_recall
    all_results["feature_names"] = list(X_train.columns)

    return best_model, all_results




def save_model(
    model: Any,
    metrics: Dict[str, Any],
    dataset_size: int,
    version: Optional[int] = None,
) -> Path:
    """
    Save the trained model with versioning and metadata.

    Args:
        model: Trained model to save.
        metrics: Training metrics dictionary.
        dataset_size: Number of training samples.
        version: Explicit version number. Auto-increments if None.

    Returns:
        Path where the model was saved.
    """
    if version is None:
        version = _get_next_model_version()

    # Save model
    model_path = MODELS_DIR / f"model_v{version}.pkl"
    joblib.dump(model, model_path)
    logger.info(f"Model saved to: {model_path}")

    # Also save as 'best_model.pkl' for easy access
    joblib.dump(model, BEST_MODEL_FILE)
    logger.info(f"Best model also saved to: {BEST_MODEL_FILE}")

    # Save metrics
    metrics_output = {
        "version": version,
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics,
    }
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics_output, f, indent=2, default=str)
    logger.info(f"Metrics saved to: {METRICS_FILE}")

    # Save metadata to JSON (as backup)
    metadata = {
        "version": version,
        "date": datetime.now().isoformat(),
        "model_type": type(model).__name__,
        "dataset_size": dataset_size,
        "feature_names": list(metrics.get("feature_names", [])),
        "metrics": {
            k: v for k, v in metrics.items()
            if k not in ["best_model", "best_recall", "feature_names"]
        },
    }

    # Save to database
    try:
        from src.utils.models import ModelMetadata
        with get_db() as db:
            # Check if version exists to update or insert
            db_meta = ModelMetadata(
                version=version,
                model_type=type(model).__name__,
                dataset_size=dataset_size,
                feature_names=metadata["feature_names"],
                metrics=metadata["metrics"],
                parameters=metrics.get(metrics.get("best_model", ""), {}).get("best_params", {}),
                file_path=str(model_path)
            )
            db.merge(db_meta)
            db.commit()
            logger.info(f"Model metadata saved to database (version {version}) [OK]")
    except Exception as e:
        logger.error(f"Failed to save metadata to database: {e}")

    # Legacy JSON support
    try:
        all_metadata = []
        if MODEL_METADATA_FILE.exists():
            with open(MODEL_METADATA_FILE, "r") as f:
                all_metadata = json.load(f)
        
        all_metadata.append(metadata)
        with open(MODEL_METADATA_FILE, "w") as f:
            json.dump(all_metadata, f, indent=2)
        logger.info(f"Model metadata saved to: {MODEL_METADATA_FILE}")
    except Exception as e:
        logger.error(f"Failed to save JSON metadata: {e}")

    return model_path


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from src.data.ingest import load_csv
    from src.data.preprocess import preprocess_data
    from src.features.build_features import build_features
    from src.utils.config import TEST_SIZE

    # Load and preprocess
    df = load_csv()
    X, y, _ = preprocess_data(df)
    X = build_features(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    # Train
    best_model, results = train_model(X_train, y_train)

    # Save
    save_model(best_model, results, len(X_train))
    print(f"Training complete. Best model: {results['best_model']}")
