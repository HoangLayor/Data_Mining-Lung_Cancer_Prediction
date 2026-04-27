"""
Tests for the Data Ingestion Module.

Tests CSV loading, schema validation, and error handling.
"""

import sys
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ingest import (
    load_csv,
    validate_data,
    save_raw_data,
    load_from_api,
    generate_sample_dataset,
)
from src.utils.config import REQUIRED_COLUMNS, TARGET_COLUMN, BINARY_COLUMNS


class TestLoadCSV:
    """Tests for the load_csv function."""

    def test_load_csv_file_not_found(self, tmp_path):
        """Should raise FileNotFoundError for non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_csv(str(tmp_path / "nonexistent.csv"))

    def test_load_csv_empty_file(self, tmp_path):
        """Should raise EmptyDataError for empty CSV."""
        empty_file = tmp_path / "empty.csv"
        empty_file.write_text("")
        with pytest.raises(pd.errors.EmptyDataError):
            load_csv(str(empty_file))

    def test_load_csv_valid_file(self, tmp_path):
        """Should successfully load a valid CSV file."""
        df = generate_sample_dataset(50)
        csv_path = tmp_path / "test.csv"
        df.to_csv(csv_path, index=False)

        loaded = load_csv(str(csv_path))
        assert len(loaded) == 50
        assert list(loaded.columns) == REQUIRED_COLUMNS


class TestValidateData:
    """Tests for the validate_data function."""

    def test_validate_valid_data(self):
        """Should pass validation for properly formatted data."""
        df = generate_sample_dataset(20)
        assert validate_data(df) is True

    def test_validate_missing_columns(self):
        """Should raise ValueError when required columns are missing."""
        df = pd.DataFrame({"AGE": [50, 60], "GENDER": ["M", "F"]})
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_data(df)

    def test_validate_invalid_binary_values(self):
        """Should raise ValueError for invalid binary column values."""
        df = generate_sample_dataset(20)
        df["SMOKING"] = 5  # Invalid value
        with pytest.raises(ValueError, match="Data validation failed"):
            validate_data(df)

    def test_validate_all_columns_present(self):
        """Should verify all required columns are present."""
        df = generate_sample_dataset(10)
        for col in REQUIRED_COLUMNS:
            assert col in df.columns


class TestSaveRawData:
    """Tests for the save_raw_data function."""

    def test_save_and_reload(self, tmp_path):
        """Should save and reload data consistently."""
        df = generate_sample_dataset(30)
        output_path = tmp_path / "saved.csv"
        save_raw_data(df, str(output_path))

        assert output_path.exists()
        reloaded = pd.read_csv(output_path)
        assert len(reloaded) == 30
        assert list(reloaded.columns) == list(df.columns)


class TestLoadFromAPI:
    """Tests for the load_from_api mock function."""

    def test_api_returns_dataframe(self):
        """Should return a valid DataFrame."""
        df = load_from_api()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 100

    def test_api_data_has_required_columns(self):
        """Should include all required columns."""
        df = load_from_api()
        for col in REQUIRED_COLUMNS:
            assert col in df.columns


class TestGenerateSampleDataset:
    """Tests for the generate_sample_dataset function."""

    def test_generates_correct_size(self):
        """Should generate the requested number of samples."""
        df = generate_sample_dataset(100)
        assert len(df) == 100

    def test_generates_valid_data(self):
        """Generated data should pass validation."""
        df = generate_sample_dataset(50)
        assert validate_data(df) is True

    def test_reproducible_output(self):
        """Same seed should produce same output."""
        df1 = generate_sample_dataset(50)
        df2 = generate_sample_dataset(50)
        pd.testing.assert_frame_equal(df1, df2)
