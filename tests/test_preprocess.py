"""
Tests for the Data Preprocessing Module.

Tests missing value handling, encoding, scaling, and outlier capping.
"""

import sys
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ingest import generate_sample_dataset
from src.data.preprocess import (
    preprocess_data,
    _cap_outliers_iqr,
    _encode_target,
    _encode_gender,
    _remap_binary_columns,
    _handle_missing_values,
)
from src.utils.config import TARGET_COLUMN, BINARY_COLUMNS


class TestEncodeTarget:
    """Tests for target encoding."""

    def test_encode_yes_no(self):
        """Should encode YES→1, NO→0."""
        df = pd.DataFrame({TARGET_COLUMN: ["YES", "NO", "YES", "NO"]})
        result = _encode_target(df)
        assert list(result[TARGET_COLUMN]) == [1, 0, 1, 0]

    def test_already_encoded(self):
        """Should not modify already-encoded target."""
        df = pd.DataFrame({TARGET_COLUMN: [1, 0, 1, 0]})
        result = _encode_target(df)
        assert list(result[TARGET_COLUMN]) == [1, 0, 1, 0]


class TestEncodeGender:
    """Tests for gender encoding."""

    def test_encode_mf(self):
        """Should encode M→1, F→0."""
        df = pd.DataFrame({"GENDER": ["M", "F", "M", "F"]})
        result = _encode_gender(df)
        assert list(result["GENDER"]) == [1, 0, 1, 0]


class TestRemapBinary:
    """Tests for binary column remapping."""

    def test_remap_1_2_to_0_1(self):
        """Should remap 1→0, 2→1."""
        df = pd.DataFrame({"SMOKING": [1, 2, 1, 2]})
        result = _remap_binary_columns(df)
        assert list(result["SMOKING"]) == [0, 1, 0, 1]


class TestCapOutliers:
    """Tests for IQR outlier capping."""

    def test_caps_extreme_values(self):
        """Should cap extreme values to IQR bounds."""
        data = pd.Series([20, 30, 40, 50, 60, 200], name="AGE")
        result = _cap_outliers_iqr(data)
        assert result.max() <= 200  # Should be capped
        assert result.min() >= data.min()

    def test_no_change_for_normal_data(self):
        """Should not modify data within IQR bounds."""
        data = pd.Series([30, 40, 50, 55, 60], name="AGE")
        result = _cap_outliers_iqr(data)
        pd.testing.assert_series_equal(data, result)


class TestHandleMissingValues:
    """Tests for missing value handling."""

    def test_fills_numerical_with_median(self):
        """Should fill numerical NaN with median."""
        df = pd.DataFrame({"AGE": [30, np.nan, 50, 60]})
        result = _handle_missing_values(df)
        assert result["AGE"].isnull().sum() == 0
        assert result["AGE"].iloc[1] == 50.0  # median of [30, 50, 60]

    def test_no_missing_passes_through(self):
        """Should pass through data with no missing values."""
        df = generate_sample_dataset(20)
        result = _handle_missing_values(df)
        assert result.isnull().sum().sum() == 0


class TestPreprocessData:
    """Integration tests for the full preprocessing pipeline."""

    def test_full_pipeline_output_shape(self, tmp_path):
        """Should return correct shapes after preprocessing."""
        df = generate_sample_dataset(100)
        X, y, preprocessor = preprocess_data(
            df,
            preprocessor_path=str(tmp_path / "test_preprocessor.pkl"),
        )

        assert len(X) == 100
        assert len(y) == 100
        assert TARGET_COLUMN not in X.columns

    def test_target_is_binary(self, tmp_path):
        """Target should be 0/1 after preprocessing."""
        df = generate_sample_dataset(50)
        X, y, _ = preprocess_data(
            df,
            preprocessor_path=str(tmp_path / "test_preprocessor.pkl"),
        )
        assert set(y.unique()).issubset({0, 1})

    def test_no_missing_after_preprocess(self, tmp_path):
        """Should have no missing values after preprocessing."""
        df = generate_sample_dataset(50)
        X, y, _ = preprocess_data(
            df,
            preprocessor_path=str(tmp_path / "test_preprocessor.pkl"),
        )
        assert X.isnull().sum().sum() == 0
        assert y.isnull().sum() == 0

    def test_preprocessor_saved(self, tmp_path):
        """Should save preprocessor to disk."""
        df = generate_sample_dataset(50)
        prep_path = tmp_path / "test_preprocessor.pkl"
        X, y, _ = preprocess_data(
            df,
            preprocessor_path=str(prep_path),
        )
        assert prep_path.exists()
