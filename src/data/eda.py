"""
Exploratory Data Analysis (EDA) for Lung Cancer Prediction dataset.

This module covers:
1) Dataset shape and dtypes
2) Target distribution analysis
3) Missing values per column
4) Outlier detection for selected continuous variables
5) Correlation analysis for numerical features
6) Per-variable distribution plots matched to variable type (continuous
   hist+KDE, binary bars, ordinal/discrete bars)

Usage:
    python -m src.data.eda
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils.config import RAW_DATA_FILE, TARGET_COLUMN
from src.utils.logger import get_logger

logger = get_logger(__name__)


EDA_REPORT_DIR = Path("reports") / "eda"
DEFAULT_CONTINUOUS_COLS = ["age", "oxygen_saturation", "fev1_x10", "crp_level"]

HIST_BINS_DEFAULT = 30

# Feature groups for EDA distribution figures
FEATURE_DISTRIBUTION_GROUPS: list[tuple[str, str, list[str]]] = [
    ("smoking", "Nhóm thuốc lá", ["pack_years", "smoking_years", "cigarettes_per_day"]),
    (
        "clinical",
        "Nhóm lâm sàng / triệu chứng",
        ["copd", "chronic_cough", "shortness_of_breath", "xray_abnormal"],
    ),
    ("biomarkers", "Nhóm chỉ số đo lường", ["oxygen_saturation", "fev1_x10", "crp_level"]),
    (
        "socioeconomic",
        "Nhóm xã hội – kinh tế",
        ["education_years", "income_level", "healthcare_access"],
    ),
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
    sns.heatmap(
        corr,
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        square=False,
        cbar_kws={"ticks": np.arange(-1.0, 1.01, 0.25), "format": "%.2f"},
    )
    plt.title("Correlation Heatmap (Numerical Features)")
    plt.tight_layout()
    output_path = output_dir / "correlation_heatmap.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {output_path}")


def _plot_binary_counts(df: pd.DataFrame, var: str, slug: str, output_dir: Path) -> None:
    """Binary 0/1: count bar chart (phù hợp hơn histogram liên tục)."""
    plot_df = df[[var]].dropna()
    if plot_df.empty:
        return
    order = sorted(plot_df[var].unique())
    plt.figure(figsize=(6, 5))
    ax = sns.countplot(
        data=plot_df,
        x=var,
        order=order,
        hue=var,
        palette="Blues",
        edgecolor="white",
        legend=False,
    )
    ax.set_title(f"Tần suất {var} (0/1)")
    ax.set_xlabel(var)
    ax.set_ylabel("Count")
    for p in ax.patches:
        ax.annotate(
            f"{int(p.get_height())}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="bottom",
            fontsize=10,
        )
    plt.tight_layout()
    out = output_dir / f"distribution_{slug}_{var}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out}")


def _plot_continuous_hist_kde(series: pd.Series, var: str, slug: str, output_dir: Path) -> None:
    """Continuous physiological / lab-style variables: histogram + KDE."""
    s = series.dropna()
    if s.empty:
        return
    plt.figure(figsize=(8, 5))
    ax = plt.gca()
    sns.histplot(
        s,
        kde=True,
        bins=HIST_BINS_DEFAULT,
        ax=ax,
        color="skyblue",
        edgecolor="white",
    )
    ax.set_title(f"Distribution of {var}")
    ax.set_xlabel(var)
    ax.set_ylabel("Count")
    plt.tight_layout()
    out = output_dir / f"distribution_{slug}_{var}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out}")


def _plot_ordinal_or_discrete_bars(series: pd.Series, var: str, slug: str, output_dir: Path) -> None:
    """Ordinal ranks or discrete years: ordered bar chart of counts (không dùng KDE)."""
    s = series.dropna()
    if s.empty:
        return
    vc = s.value_counts().sort_index()
    levels = vc.index.tolist()
    plot_df = pd.DataFrame({"level": levels, "count": vc.values})
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(
        data=plot_df,
        x="level",
        y="count",
        order=levels,
        color="cadetblue",
        edgecolor="white",
    )
    ax.set_title(f"Tần suất theo mức (có thứ tự): {var}")
    ax.set_xlabel(var)
    ax.set_ylabel("Count")
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels([f"{x:g}" for x in levels])
    plt.tight_layout()
    out = output_dir / f"distribution_{slug}_{var}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out}")


def _plot_feature_group_distributions(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Save one figure per variable; chart type matches data semantics per group.
    """
    for slug, _title, members in FEATURE_DISTRIBUTION_GROUPS:
        plotted = 0
        for var in members:
            if var not in df.columns or not pd.api.types.is_numeric_dtype(df[var]):
                continue
            if slug in ("smoking", "biomarkers"):
                _plot_continuous_hist_kde(df[var], var, slug, output_dir)
            elif slug == "clinical":
                _plot_binary_counts(df, var, slug, output_dir)
            elif slug == "socioeconomic":
                _plot_ordinal_or_discrete_bars(df[var], var, slug, output_dir)
            else:
                _plot_continuous_hist_kde(df[var], var, slug, output_dir)
            plotted += 1
        if plotted == 0:
            logger.warning(f"No plottable variables for group '{slug}'.")


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

    logger.info(
        "Running EDA: shape/dtype, target, missing, outliers, correlation, "
        "grouped distributions (type-matched charts)"
    )
    _save_text_report(df, output_dir)
    _plot_target_distribution(df, output_dir)
    _plot_missing_values(df, output_dir)
    _plot_outliers(df, output_dir, DEFAULT_CONTINUOUS_COLS)
    _plot_correlation_heatmap(df, output_dir)
    _plot_feature_group_distributions(df, output_dir)

    logger.info(f"EDA completed. Artifacts saved in: {output_dir.resolve()}")
    return output_dir


if __name__ == "__main__":
    run_eda()
