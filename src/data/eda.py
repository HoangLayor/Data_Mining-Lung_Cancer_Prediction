"""
Exploratory Data Analysis (EDA) for Lung Cancer Prediction dataset.

This module covers:
1) Dataset shape and dtypes
2) Target distribution analysis
3) Missing values per column
4) Outlier detection for selected continuous variables
5) Correlation analysis for numerical features
6) Distribution comparison between low-risk and high-risk groups

Usage:
    python -m src.data.eda
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.utils.config import RAW_DATA_FILE, TARGET_COLUMN
from src.utils.logger import get_logger

logger = get_logger(__name__)


EDA_REPORT_DIR = Path("reports") / "eda"
DEFAULT_CONTINUOUS_COLS = ["age", "oxygen_saturation", "fev1_x10", "crp_level"]
DEFAULT_KEY_FEATURES = [
    "pack_years",
    "smoking_years",
    "cigarettes_per_day",
    "copd",
    "chronic_cough",
    "shortness_of_breath",
    "xray_abnormal",
    "oxygen_saturation",
    "fev1_x10",
    "crp_level",
    "education_years",
    "income_level",
    "healthcare_access",
]


def _ensure_output_dir() -> Path:
    EDA_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    return EDA_REPORT_DIR


def _save_text_report(df: pd.DataFrame, output_dir: Path) -> None:
    report_path = output_dir / "eda_report.txt"
    missing_pct = (df.isnull().mean() * 100).sort_values(ascending=False)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("LUNG CANCER DATASET - EDA REPORT\n")
        f.write("=" * 60 + "\n\n")

        f.write("1) Dataset shape and dtypes\n")
        f.write("-" * 60 + "\n")
        f.write(f"Shape: {df.shape}\n\n")
        f.write("Columns and dtypes:\n")
        f.write(df.dtypes.to_string() + "\n\n")

        f.write("2) Target distribution\n")
        f.write("-" * 60 + "\n")
        if TARGET_COLUMN in df.columns:
            target_counts = df[TARGET_COLUMN].value_counts(dropna=False)
            target_pct = df[TARGET_COLUMN].value_counts(normalize=True, dropna=False) * 100
            f.write("Counts:\n")
            f.write(target_counts.to_string() + "\n\n")
            f.write("Percentages (%):\n")
            f.write(target_pct.round(2).to_string() + "\n\n")
        else:
            f.write(f"Target column '{TARGET_COLUMN}' not found.\n\n")

        f.write("3) Missing values by column\n")
        f.write("-" * 60 + "\n")
        f.write("Missing count:\n")
        f.write(df.isnull().sum().sort_values(ascending=False).to_string() + "\n\n")
        f.write("Missing percentage (%):\n")
        f.write(missing_pct.round(2).to_string() + "\n\n")

        f.write("4) Numerical summary\n")
        f.write("-" * 60 + "\n")
        numeric_df = df.select_dtypes(include="number")
        if not numeric_df.empty:
            f.write(numeric_df.describe().T.to_string() + "\n\n")
        else:
            f.write("No numerical columns found.\n\n")

    logger.info(f"Saved EDA text report: {report_path}")


def _plot_target_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    if TARGET_COLUMN not in df.columns:
        logger.warning("Target column missing; skipping target distribution plot.")
        return

    plt.figure(figsize=(7, 5))
    ax = sns.countplot(x=TARGET_COLUMN, data=df)
    ax.set_title("Target Distribution: lung_cancer_risk")
    ax.set_xlabel(TARGET_COLUMN)
    ax.set_ylabel("Count")
    plt.tight_layout()
    output_path = output_dir / "target_distribution.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {output_path}")


def _plot_missing_values(df: pd.DataFrame, output_dir: Path) -> None:
    missing_pct = (df.isnull().mean() * 100).sort_values(ascending=False)
    missing_pct = missing_pct[missing_pct > 0]

    if missing_pct.empty:
        logger.info("No missing values detected; skipping missing plot.")
        return

    plt.figure(figsize=(10, 6))
    sns.barplot(x=missing_pct.values, y=missing_pct.index)
    plt.title("Missing Values Percentage by Column")
    plt.xlabel("Missing (%)")
    plt.ylabel("Columns")
    plt.tight_layout()
    output_path = output_dir / "missing_values_percentage.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {output_path}")


def _plot_outliers(df: pd.DataFrame, output_dir: Path, columns: Iterable[str]) -> None:
    available_cols = [col for col in columns if col in df.columns]
    if not available_cols:
        logger.warning("No configured continuous columns found for outlier analysis.")
        return

    for col in available_cols:
        plt.figure(figsize=(8, 5))
        sns.boxplot(x=df[col])
        plt.title(f"Outlier Check (Boxplot): {col}")
        plt.xlabel(col)
        plt.tight_layout()
        output_path = output_dir / f"outlier_boxplot_{col}.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved: {output_path}")


def _plot_correlation_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] < 2:
        logger.warning("Not enough numerical columns for correlation heatmap.")
        return

    corr = numeric_df.corr()
    plt.figure(figsize=(14, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0, square=False)
    plt.title("Correlation Heatmap (Numerical Features)")
    plt.tight_layout()
    output_path = output_dir / "correlation_heatmap.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {output_path}")


def _plot_group_comparison(df: pd.DataFrame, output_dir: Path, columns: Iterable[str]) -> None:
    if TARGET_COLUMN not in df.columns:
        logger.warning("Target column missing; skipping group comparison plots.")
        return

    available_cols = [col for col in columns if col in df.columns]
    if not available_cols:
        logger.warning("No key features available for group comparison.")
        return

    for col in available_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        plt.figure(figsize=(8, 5))
        sns.boxplot(x=TARGET_COLUMN, y=col, data=df)
        plt.title(f"{col} by {TARGET_COLUMN} Group")
        plt.xlabel(TARGET_COLUMN)
        plt.ylabel(col)
        plt.tight_layout()
        output_path = output_dir / f"group_comparison_{col}.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved: {output_path}")


def run_eda(data_path: Path | None = None) -> Path:
    """
    Run complete EDA workflow and save outputs to reports/eda.

    Args:
        data_path: Optional dataset path. Defaults to RAW_DATA_FILE.

    Returns:
        Output directory path containing all generated artifacts.
    """
    input_path = data_path or RAW_DATA_FILE
    if not input_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {input_path}. "
            "Generate/load raw data first (e.g., python -m src.data.ingest)."
        )

    logger.info(f"Loading dataset for EDA: {input_path}")
    df = pd.read_csv(input_path)
    output_dir = _ensure_output_dir()

    logger.info("Running EDA: shape/dtype, target, missing, outliers, correlation, group comparison")
    _save_text_report(df, output_dir)
    _plot_target_distribution(df, output_dir)
    _plot_missing_values(df, output_dir)
    _plot_outliers(df, output_dir, DEFAULT_CONTINUOUS_COLS)
    _plot_correlation_heatmap(df, output_dir)
    _plot_group_comparison(df, output_dir, DEFAULT_KEY_FEATURES)

    logger.info(f"EDA completed. Artifacts saved in: {output_dir.resolve()}")
    return output_dir


if __name__ == "__main__":
    run_eda()
