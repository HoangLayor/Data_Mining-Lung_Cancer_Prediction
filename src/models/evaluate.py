"""
Model Evaluation Module for the Lung Cancer Prediction project.

Generates evaluation reports, confusion matrices, ROC curves,
Precision-Recall curves, and performs threshold tuning.
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    recall_score,
    precision_score,
    f1_score,
    accuracy_score,
)

from src.utils.logger import get_logger
from src.utils.config import PLOTS_DIR, MODELS_DIR, EVALUATION_REPORT_FILE

logger = get_logger(__name__)

# Set plot style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Comprehensive model evaluation.

    Args:
        model: Trained model with predict and predict_proba methods.
        X_test: Test feature matrix.
        y_test: Test target vector.
        threshold: Classification threshold.

    Returns:
        Dictionary with all evaluation metrics.
    """
    logger.info("=" * 60)
    logger.info("Starting model evaluation...")
    logger.info(f"Test data shape: X={X_test.shape}, y={y_test.shape}")

    # Get predictions
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    # Compute metrics
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "average_precision": float(average_precision_score(y_test, y_proba)),
        "threshold_used": threshold,
    }

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    metrics["confusion_matrix"] = cm.tolist()
    metrics["true_negatives"] = int(cm[0, 0])
    metrics["false_positives"] = int(cm[0, 1])
    metrics["false_negatives"] = int(cm[1, 0])
    metrics["true_positives"] = int(cm[1, 1])

    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    metrics["classification_report"] = report

    # Log results
    logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"  Recall:    {metrics['recall']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  F1-Score:  {metrics['f1_score']:.4f}")
    logger.info(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    logger.info(f"  Threshold: {threshold}")
    logger.info(f"  Confusion Matrix:\n{cm}")

    # Save evaluation report
    with open(EVALUATION_REPORT_FILE, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info(f"Evaluation report saved to: {EVALUATION_REPORT_FILE}")

    logger.info("=" * 60)
    return metrics


def find_optimal_threshold(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    min_recall: float = 0.8,
) -> Tuple[float, Dict[str, float]]:
    """
    Find the optimal classification threshold that achieves minimum recall.

    Args:
        model: Trained model.
        X_test: Test features.
        y_test: Test target.
        min_recall: Minimum acceptable recall.

    Returns:
        Tuple of (optimal_threshold, metrics_at_threshold).
    """
    logger.info(f"Finding optimal threshold (min_recall >= {min_recall})...")

    y_proba = model.predict_proba(X_test)[:, 1]
    thresholds = np.arange(0.1, 0.9, 0.01)

    best_threshold = 0.5
    best_f1 = 0.0

    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        if recall >= min_recall and f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh

    # Compute metrics at optimal threshold
    y_pred_optimal = (y_proba >= best_threshold).astype(int)
    optimal_metrics = {
        "threshold": float(best_threshold),
        "recall": float(recall_score(y_test, y_pred_optimal)),
        "precision": float(precision_score(y_test, y_pred_optimal)),
        "f1_score": float(f1_score(y_test, y_pred_optimal)),
    }

    logger.info(f"  Optimal threshold: {best_threshold:.2f}")
    logger.info(f"  Recall:    {optimal_metrics['recall']:.4f}")
    logger.info(f"  Precision: {optimal_metrics['precision']:.4f}")
    logger.info(f"  F1-Score:  {optimal_metrics['f1_score']:.4f}")

    return best_threshold, optimal_metrics


def plot_metrics(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save_dir: Optional[Path] = None,
) -> None:
    """
    Generate and save all evaluation plots.

    Plots:
        1. Confusion Matrix heatmap
        2. ROC Curve
        3. Precision-Recall Curve
        4. Threshold Analysis

    Args:
        model: Trained model.
        X_test: Test features.
        y_test: Test target.
        save_dir: Directory to save plots. Defaults to PLOTS_DIR.
    """
    output_dir = save_dir or PLOTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    logger.info("Generating evaluation plots...")

    # ── 1. Confusion Matrix ──
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Cancer", "Cancer"],
        yticklabels=["No Cancer", "Cancer"],
        ax=ax,
        annot_kws={"size": 16},
    )
    ax.set_xlabel("Predicted", fontsize=14)
    ax.set_ylabel("Actual", fontsize=14)
    ax.set_title("Confusion Matrix", fontsize=16, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  [OK] Confusion matrix saved")

    # ── 2. ROC Curve ──
    fig, ax = plt.subplots(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_score = roc_auc_score(y_test, y_proba)
    ax.plot(fpr, tpr, color="#2196F3", lw=2.5, label=f"ROC Curve (AUC = {auc_score:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random Classifier")
    ax.fill_between(fpr, tpr, alpha=0.15, color="#2196F3")
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title("ROC Curve", fontsize=16, fontweight="bold")
    ax.legend(loc="lower right", fontsize=12)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    plt.tight_layout()
    fig.savefig(output_dir / "roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  [OK] ROC curve saved")

    # ── 3. Precision-Recall Curve ──
    fig, ax = plt.subplots(figsize=(8, 6))
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba)
    avg_precision = average_precision_score(y_test, y_proba)
    ax.plot(
        recall_vals, precision_vals,
        color="#4CAF50", lw=2.5,
        label=f"PR Curve (AP = {avg_precision:.3f})",
    )
    ax.fill_between(recall_vals, precision_vals, alpha=0.15, color="#4CAF50")
    ax.set_xlabel("Recall", fontsize=13)
    ax.set_ylabel("Precision", fontsize=13)
    ax.set_title("Precision-Recall Curve", fontsize=16, fontweight="bold")
    ax.legend(loc="lower left", fontsize=12)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    plt.tight_layout()
    fig.savefig(output_dir / "precision_recall_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  [OK] Precision-Recall curve saved")

    # ── 4. Threshold Analysis ──
    fig, ax = plt.subplots(figsize=(10, 6))
    thresholds = np.arange(0.1, 0.91, 0.01)
    recalls, precisions, f1s = [], [], []

    for thresh in thresholds:
        y_t = (y_proba >= thresh).astype(int)
        recalls.append(recall_score(y_test, y_t, zero_division=0))
        precisions.append(precision_score(y_test, y_t, zero_division=0))
        f1s.append(f1_score(y_test, y_t, zero_division=0))

    ax.plot(thresholds, recalls, label="Recall", color="#F44336", lw=2)
    ax.plot(thresholds, precisions, label="Precision", color="#2196F3", lw=2)
    ax.plot(thresholds, f1s, label="F1-Score", color="#FF9800", lw=2)
    ax.axhline(y=0.8, color="gray", linestyle="--", alpha=0.5, label="Recall = 0.8")
    ax.set_xlabel("Threshold", fontsize=13)
    ax.set_ylabel("Score", fontsize=13)
    ax.set_title("Threshold Analysis", fontsize=16, fontweight="bold")
    ax.legend(fontsize=12)
    ax.set_xlim([0.1, 0.9])
    ax.set_ylim([0.0, 1.05])
    plt.tight_layout()
    fig.savefig(output_dir / "threshold_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  [OK] Threshold analysis saved")

    logger.info(f"All plots saved to: {output_dir}")


if __name__ == "__main__":
    import joblib
    from sklearn.model_selection import train_test_split
    from src.data.ingest import load_csv
    from src.data.preprocess import preprocess_data
    from src.features.build_features import build_features
    from src.utils.config import TEST_SIZE, RANDOM_SEED, BEST_MODEL_FILE

    # Load and preprocess
    df = load_csv()
    X, y, _ = preprocess_data(df)
    X = build_features(X)

    # Split
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    # Load model
    model = joblib.load(BEST_MODEL_FILE)

    # Evaluate
    metrics = evaluate_model(model, X_test, y_test)
    threshold, opt_metrics = find_optimal_threshold(model, X_test, y_test)
    plot_metrics(model, X_test, y_test)

    print(f"Evaluation complete. Optimal threshold: {threshold:.2f}")
