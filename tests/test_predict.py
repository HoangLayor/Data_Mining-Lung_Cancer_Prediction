"""
Tests for the Prediction Module.

Tests input validation, preprocessing, and end-to-end prediction.
"""

import sys
import pytest
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.predict import (
    predict,
    _validate_input,
    _preprocess_input,
    reload_model,
)
from src.utils.config import REQUIRED_COLUMNS, TARGET_COLUMN


# ──────────────────────────────────────────────
# Sample test data
# ──────────────────────────────────────────────
VALID_PATIENT = {
    "GENDER": "M",
    "AGE": 65,
    "SMOKING": 2,
    "YELLOW_FINGERS": 2,
    "ANXIETY": 1,
    "PEER_PRESSURE": 1,
    "CHRONIC_DISEASE": 2,
    "FATIGUE": 2,
    "ALLERGY": 1,
    "WHEEZING": 2,
    "ALCOHOL_CONSUMING": 2,
    "COUGHING": 2,
    "SHORTNESS_OF_BREATH": 2,
    "SWALLOWING_DIFFICULTY": 1,
    "CHEST_PAIN": 2,
}


class TestValidateInput:
    """Tests for input validation."""

    def test_valid_input_passes(self):
        """Should pass for valid input data."""
        _validate_input(VALID_PATIENT)  # Should not raise

    def test_missing_fields_raises(self):
        """Should raise ValueError for missing required fields."""
        incomplete = {"GENDER": "M", "AGE": 50}
        with pytest.raises(ValueError, match="Missing required fields"):
            _validate_input(incomplete)

    def test_invalid_gender_raises(self):
        """Should raise ValueError for invalid gender."""
        invalid = VALID_PATIENT.copy()
        invalid["GENDER"] = "X"
        with pytest.raises(ValueError, match="Invalid GENDER"):
            _validate_input(invalid)

    def test_invalid_age_raises(self):
        """Should raise ValueError for invalid age."""
        invalid = VALID_PATIENT.copy()
        invalid["AGE"] = -5
        with pytest.raises(ValueError, match="Invalid AGE"):
            _validate_input(invalid)

    def test_invalid_age_too_high(self):
        """Should raise ValueError for unrealistic age."""
        invalid = VALID_PATIENT.copy()
        invalid["AGE"] = 200
        with pytest.raises(ValueError, match="Invalid AGE"):
            _validate_input(invalid)

    def test_invalid_binary_value_raises(self):
        """Should raise ValueError for invalid binary column values."""
        invalid = VALID_PATIENT.copy()
        invalid["SMOKING"] = 5
        with pytest.raises(ValueError, match="Invalid value for SMOKING"):
            _validate_input(invalid)


class TestPreprocessInput:
    """Tests for input preprocessing."""

    def test_gender_encoding(self):
        """Should encode GENDER M→1, F→0."""
        # Mock the preprocessor
        mock_scaler = MagicMock()
        mock_scaler.transform.return_value = np.array([[0.5]])

        with patch("src.models.predict._load_preprocessor", return_value=mock_scaler):
            with patch("src.models.predict.build_features", side_effect=lambda x: x):
                result = _preprocess_input(VALID_PATIENT)
                assert result["GENDER"].values[0] == 1

    def test_binary_remapping(self):
        """Should remap binary 1/2 to 0/1."""
        mock_scaler = MagicMock()
        mock_scaler.transform.return_value = np.array([[0.5]])

        with patch("src.models.predict._load_preprocessor", return_value=mock_scaler):
            with patch("src.models.predict.build_features", side_effect=lambda x: x):
                result = _preprocess_input(VALID_PATIENT)
                # SMOKING was 2 (yes), should be mapped to 1
                assert result["SMOKING"].values[0] == 1
                # ANXIETY was 1 (no), should be mapped to 0
                assert result["ANXIETY"].values[0] == 0


class TestPredict:
    """Integration tests for the predict function."""

    def test_predict_returns_expected_format(self):
        """Should return dict with prediction, probability, risk_level, label."""
        # Create a mock model
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([1])
        mock_model.predict_proba.return_value = np.array([[0.2, 0.8]])

        mock_scaler = MagicMock()
        mock_scaler.transform.return_value = np.array([[0.5]])

        with patch("src.models.predict._load_model", return_value=mock_model):
            with patch("src.models.predict._load_preprocessor", return_value=mock_scaler):
                with patch("src.models.predict.build_features", side_effect=lambda x: x):
                    result = predict(VALID_PATIENT)

                    assert "prediction" in result
                    assert "probability" in result
                    assert "risk_level" in result
                    assert "label" in result
                    assert result["prediction"] in [0, 1]
                    assert 0 <= result["probability"] <= 1

    def test_predict_high_risk(self):
        """Should classify high probability as High risk."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([1])
        mock_model.predict_proba.return_value = np.array([[0.1, 0.9]])

        mock_scaler = MagicMock()
        mock_scaler.transform.return_value = np.array([[0.5]])

        with patch("src.models.predict._load_model", return_value=mock_model):
            with patch("src.models.predict._load_preprocessor", return_value=mock_scaler):
                with patch("src.models.predict.build_features", side_effect=lambda x: x):
                    result = predict(VALID_PATIENT)
                    assert result["risk_level"] == "High"

    def test_predict_low_risk(self):
        """Should classify low probability as Low risk."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0])
        mock_model.predict_proba.return_value = np.array([[0.85, 0.15]])

        mock_scaler = MagicMock()
        mock_scaler.transform.return_value = np.array([[0.5]])

        with patch("src.models.predict._load_model", return_value=mock_model):
            with patch("src.models.predict._load_preprocessor", return_value=mock_scaler):
                with patch("src.models.predict.build_features", side_effect=lambda x: x):
                    result = predict(VALID_PATIENT)
                    assert result["risk_level"] == "Low"


class TestReloadModel:
    """Tests for model reload functionality."""

    def test_reload_clears_cache(self):
        """Should clear cached model and preprocessor."""
        import src.models.predict as predict_module
        predict_module._cached_model = "fake_model"
        predict_module._cached_preprocessor = "fake_preprocessor"

        reload_model()

        assert predict_module._cached_model is None
        assert predict_module._cached_preprocessor is None
