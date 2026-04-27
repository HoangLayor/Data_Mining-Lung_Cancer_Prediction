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
    LOGISTIC_REGRESSION_PARAMS,
    RANDOM_FOREST_PARAMS,
    XGBOOST_PARAMS,
)

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


def _get_param_grids() -> Dict[str, Dict]:
    """Return hyperparameter grids for each model."""
    return {
        "LogisticRegression": LOGISTIC_REGRESSION_PARAMS,
        "RandomForest": RANDOM_FOREST_PARAMS,
        "XGBoost": XGBOOST_PARAMS,
    }


def cross_validate_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int = CV_FOLDS,
) -> Dict[str, float]:
    """
    Perform Stratified K-Fold Cross Validation.

    Args:
        model: Sklearn-compatible model.
        X: Feature matrix.
        y: Target vector.
        cv_folds: Number of folds.

    Returns:
        Dictionary with mean metrics across folds.
    """
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED)

    scorers = {
        "recall": make_scorer(recall_score),
        "precision": make_scorer(precision_score),
        "f1": make_scorer(f1_score),
        "roc_auc": make_scorer(roc_auc_score),
    }

    results = {}
    for metric_name, scorer in scorers.items():
        scores = cross_val_score(model, X, y, cv=skf, scoring=scorer)
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

    # Apply SMOTE if requested
    if use_smote:
        X_train_balanced, y_train_balanced = _apply_smote(X_train, y_train)
    else:
        X_train_balanced, y_train_balanced = X_train, y_train

    models = _get_models()
    param_grids = _get_param_grids()
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    all_results = {}
    best_model = None
    best_recall = 0.0
    best_model_name = ""

    for name, model in models.items():
        logger.info("-" * 40)
        logger.info(f"Training: {name}")
        start_time = time.time()

        try:
            # Hyperparameter tuning with RandomizedSearchCV
            search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_grids[name],
                n_iter=min(n_iter, _count_combinations(param_grids[name])),
                scoring="recall",
                cv=skf,
                random_state=RANDOM_SEED,
                n_jobs=-1,
                verbose=0,
            )
            search.fit(X_train_balanced, y_train_balanced)

            # Get best model and cross-validate
            tuned_model = search.best_estimator_
            cv_results = cross_validate_model(tuned_model, X_train_balanced, y_train_balanced)

            elapsed = time.time() - start_time

            result = {
                "best_params": search.best_params_,
                "cv_results": cv_results,
                "training_time_seconds": round(elapsed, 2),
            }
            all_results[name] = result

            recall_mean = cv_results["recall"]["mean"]
            logger.info(f"  Best params: {search.best_params_}")
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
            logger.error(f"Error training {name}: {e}")
            all_results[name] = {"error": str(e)}

    logger.info("=" * 60)
    logger.info(f">>> Best model: {best_model_name} (Recall: {best_recall:.4f})")

    # Save results
    all_results["best_model"] = best_model_name
    all_results["best_recall"] = best_recall

    return best_model, all_results


def _count_combinations(param_grid: Dict) -> int:
    """Count total hyperparameter combinations."""
    count = 1
    for values in param_grid.values():
        if isinstance(values, list):
            count *= len(values)
    return count


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

    # Save metadata
    metadata = {
        "version": version,
        "date": datetime.now().isoformat(),
        "model_type": type(model).__name__,
        "dataset_size": dataset_size,
        "metrics": {
            k: v for k, v in metrics.items()
            if k not in ["best_model", "best_recall"]
        },
    }

    # Load existing metadata or create new
    all_metadata = []
    if MODEL_METADATA_FILE.exists():
        with open(MODEL_METADATA_FILE, "r") as f:
            all_metadata = json.load(f)

    all_metadata.append(metadata)
    with open(MODEL_METADATA_FILE, "w") as f:
        json.dump(all_metadata, f, indent=2, default=str)
    logger.info(f"Model metadata saved to: {MODEL_METADATA_FILE}")

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
