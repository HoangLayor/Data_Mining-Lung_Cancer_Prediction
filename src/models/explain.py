"""
Model Explainability Module for the Lung Cancer Prediction project.

Uses SHAP (SHapley Additive exPlanations) for global and local
model interpretability.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from pathlib import Path
from typing import Optional

from src.utils.logger import get_logger
from src.utils.config import PLOTS_DIR

logger = get_logger(__name__)


def explain_global(
    model,
    X: pd.DataFrame,
    save_dir: Optional[Path] = None,
    max_display: int = 15,
) -> shap.Explanation:
    """
    Generate global feature importance explanations using SHAP.

    Produces:
        - Summary bar plot (mean |SHAP values|)
        - Summary beeswarm plot (SHAP value distribution)

    Args:
        model: Trained model.
        X: Feature matrix (test or full dataset).
        save_dir: Directory to save plots.
        max_display: Maximum number of features to display.

    Returns:
        SHAP Explanation object.
    """
    output_dir = save_dir or PLOTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating global SHAP explanations...")

    # Create SHAP explainer
    try:
        from sklearn.linear_model import LogisticRegression
        if isinstance(model, LogisticRegression):
            explainer = shap.LinearExplainer(model, X)
            logger.info("Using LinearExplainer for Logistic Regression")
        else:
            # Try TreeExplainer for tree-based models
            explainer = shap.TreeExplainer(model)
            logger.info("Using TreeExplainer")
    except Exception as e:
        # Fall back to KernelExplainer for other models
        logger.info(f"Preferred explainer not supported ({e}), using KernelExplainer (slower)...")
        # Use a small background sample for efficiency
        background = shap.sample(X, min(100, len(X)))
        explainer = shap.KernelExplainer(model.predict_proba, background)

    shap_values = explainer.shap_values(X)

    # Handle multi-output (binary classification)
    if isinstance(shap_values, list):
        shap_values_plot = shap_values[1]  # Class 1 (cancer)
    else:
        shap_values_plot = shap_values

    # ── Bar Plot (Feature Importance) ──
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values_plot, X,
        plot_type="bar",
        max_display=max_display,
        show=False,
    )
    plt.title("Global Feature Importance (SHAP)", fontsize=16, fontweight="bold")
    plt.tight_layout()
    fig = plt.gcf()
    fig.savefig(output_dir / "shap_global_bar.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    logger.info("  [OK] SHAP bar plot saved")

    # ── Beeswarm Plot (Feature Impact Distribution) ──
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values_plot, X,
        max_display=max_display,
        show=False,
    )
    plt.title("SHAP Feature Impact Distribution", fontsize=16, fontweight="bold")
    plt.tight_layout()
    fig = plt.gcf()
    fig.savefig(output_dir / "shap_global_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    logger.info("  [OK] SHAP beeswarm plot saved")

    logger.info(f"Global SHAP plots saved to: {output_dir}")

    return shap_values


def explain_local(
    model,
    sample: pd.DataFrame,
    X_background: Optional[pd.DataFrame] = None,
    save_dir: Optional[Path] = None,
    sample_index: int = 0,
) -> None:
    """
    Generate local SHAP explanation for a single prediction.

    Produces:
        - Waterfall plot for the individual prediction

    Args:
        model: Trained model.
        sample: Single-row DataFrame (or small batch) to explain.
        X_background: Background dataset for KernelExplainer.
        save_dir: Directory to save plots.
        sample_index: Index of sample to plot (if batch).
    """
    output_dir = save_dir or PLOTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating local SHAP explanation...")

    # Create explainer
    try:
        explainer = shap.TreeExplainer(model)
        logger.info("Using TreeExplainer for local explanation")
    except Exception:
        if X_background is None:
            logger.warning("No background data provided for KernelExplainer, using sample")
            X_background = sample
        background = shap.sample(X_background, min(50, len(X_background)))
        explainer = shap.KernelExplainer(model.predict_proba, background)

    shap_values = explainer.shap_values(sample)

    # Handle multi-output (binary classification)
    if isinstance(shap_values, list):
        sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
        # Some explainers return (samples, features, classes)
        sv = shap_values[:, :, 1] if shap_values.shape[2] > 1 else shap_values[:, :, 0]
    else:
        sv = shap_values

    # Get prediction for context
    prediction = model.predict(sample)[0]
    proba = model.predict_proba(sample)[0]
    logger.info(f"  Sample prediction: {prediction} (probability: {proba})")

    # ── Waterfall Plot ──
    try:
        if isinstance(sv, np.ndarray) and sv.ndim == 2:
            sv_row = sv[sample_index]
        else:
            sv_row = sv

        # Create SHAP Explanation object for waterfall plot
        expected_value = explainer.expected_value
        if isinstance(expected_value, (list, np.ndarray)) and len(expected_value) > 1:
            expected_value = expected_value[1]
        elif isinstance(expected_value, (list, np.ndarray)):
            expected_value = expected_value[0]

        explanation = shap.Explanation(
            values=sv_row,
            base_values=expected_value,
            data=sample.iloc[sample_index].values if hasattr(sample, 'iloc') else sample,
            feature_names=list(sample.columns) if hasattr(sample, 'columns') else None,
        )

        fig, ax = plt.subplots(figsize=(10, 8))
        shap.plots.waterfall(explanation, show=False)
        plt.title("Local SHAP Explanation (Single Prediction)", fontsize=14, fontweight="bold")
        plt.tight_layout()
        fig = plt.gcf()
        fig.savefig(output_dir / "shap_local_waterfall.png", dpi=150, bbox_inches="tight")
        plt.close("all")
        logger.info("  [OK] SHAP waterfall plot saved")
    except Exception as e:
        logger.warning(f"Could not generate waterfall plot: {e}")
        # Fallback: save a bar chart of SHAP values for this sample
        fig, ax = plt.subplots(figsize=(10, 6))
        if isinstance(sv, np.ndarray) and sv.ndim == 2:
            sv_row = sv[sample_index]
        else:
            sv_row = sv
        features = list(sample.columns) if hasattr(sample, 'columns') else [f"Feature {i}" for i in range(len(sv_row))]
        
        # Ensure sv_row is 1D and contains scalars
        sv_row_flat = np.array(sv_row).flatten()
        colors = ["#F44336" if float(v) > 0 else "#2196F3" for v in sv_row_flat]
        
        ax.barh(features[:len(sv_row_flat)], sv_row_flat, color=colors)
        ax.set_xlabel("SHAP Value", fontsize=13)
        ax.set_title("Local Feature Impact", fontsize=16, fontweight="bold")
        plt.tight_layout()
        fig.savefig(output_dir / "shap_local_bar.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("  [OK] SHAP local bar plot saved (fallback)")

    logger.info(f"Local SHAP plots saved to: {output_dir}")


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

    # Global explanation
    explain_global(model, X_test)

    # Local explanation (first sample)
    explain_local(model, X_test.iloc[:1], X_background=X_test)

    print("SHAP explanations generated successfully!")
