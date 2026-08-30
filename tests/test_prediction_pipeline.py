"""
tests/test_prediction_pipeline.py
===================================

PURPOSE:
    Automated unit test suite for the Phase 5 Step 1 production inference pipeline.

    Covers all 15 validation gates required by the Phase 5 Step 1 specification.

GATES:
    Gate  1: Model artifact exists on disk
    Gate  2: Preprocessor artifact exists on disk
    Gate  3: Metadata artifact exists on disk
    Gate  4: Model loads successfully
    Gate  5: Preprocessor loads successfully
    Gate  6: Model supports predict_proba()
    Gate  7: Feature dimensions match (preprocessor → model)
    Gate  8: Valid applicant prediction succeeds end-to-end
    Gate  9: PD is within [0, 1]
    Gate 10: Risk score is within [0, 100]
    Gate 11: Risk tier is valid ("LOW", "MEDIUM", or "HIGH")
    Gate 12: Decision is valid ("APPROVE", "MANUAL REVIEW", or "REJECT")
    Gate 13: Invalid categorical input is rejected
    Gate 14: NaN / Inf input is rejected
    Gate 15: Prediction path does not call .fit()

USAGE:
    Run from the project root directory:
        pytest tests/test_prediction_pipeline.py -v

    All 15 gates must pass.

    Prerequisites:
        Artifacts must exist. Run first:
            python scripts/build_model_artifacts.py

DATASET CONTEXT:
    All tests operate on the UCI Statlog German Credit benchmark model.
    Applicant fixtures in this file are ENTIRELY SYNTHETIC for testing purposes.

PHASE 5 — STEP 1
"""

import math
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Path setup — allow running from project root
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.model_artifacts import (  # noqa: E402
    METADATA_PATH,
    MODEL_PATH,
    PREPROCESSOR_PATH,
    ModelArtifacts,
    ArtifactError,
)
from src.prediction.predictor import CreditRiskPredictor  # noqa: E402
from src.prediction.risk_policy import (  # noqa: E402
    VALID_DECISIONS,
    VALID_TIERS,
    RiskPolicy,
)
from src.prediction.schemas import (  # noqa: E402
    ApplicantInput,
    InputValidationError,
    PredictionResult,
)


# ---------------------------------------------------------------------------
# Shared synthetic applicant fixture
# ---------------------------------------------------------------------------
# This is entirely fictional data for testing the UCI German Credit pipeline.

@pytest.fixture
def valid_applicant() -> dict:
    """Return a valid synthetic applicant dict with all 20 UCI features."""
    return {
        "status_existing_checking_account":       "A11",
        "duration_in_months":                     12,
        "credit_history":                         "A34",
        "purpose":                                "A43",
        "credit_amount":                          2000,
        "savings_account_bonds":                  "A61",
        "present_employment_since":               "A73",
        "installment_rate_pct_disposable_income": 2,
        "personal_status_sex":                    "A93",
        "other_debtors_guarantors":               "A101",
        "present_residence_since":                2,
        "property":                               "A121",
        "age_years":                              30,
        "other_installment_plans":                "A143",
        "housing":                                "A152",
        "existing_credits_count":                 1,
        "job":                                    "A173",
        "people_liable_maintenance":              1,
        "telephone":                              "A191",
        "foreign_worker":                         "A201",
    }


@pytest.fixture
def predictor() -> CreditRiskPredictor:
    """Return an initialized CreditRiskPredictor (artifacts must exist)."""
    return CreditRiskPredictor()


# ===========================================================================
# GATE 1 — Model artifact exists on disk
# ===========================================================================

class TestGate1ModelArtifactExists:
    """Gate 1: models/credit_risk_model.joblib must exist on disk."""

    def test_model_artifact_file_exists(self) -> None:
        """Gate 1: Verify credit_risk_model.joblib exists."""
        assert MODEL_PATH.exists(), (
            f"Gate 1 FAILED: Model artifact not found at {MODEL_PATH}\n"
            "Run: python scripts/build_model_artifacts.py"
        )


# ===========================================================================
# GATE 2 — Preprocessor artifact exists on disk
# ===========================================================================

class TestGate2PreprocessorArtifactExists:
    """Gate 2: models/preprocessing.joblib must exist on disk."""

    def test_preprocessor_artifact_file_exists(self) -> None:
        """Gate 2: Verify preprocessing.joblib exists."""
        assert PREPROCESSOR_PATH.exists(), (
            f"Gate 2 FAILED: Preprocessor artifact not found at {PREPROCESSOR_PATH}\n"
            "Run: python scripts/build_model_artifacts.py"
        )


# ===========================================================================
# GATE 3 — Metadata artifact exists on disk
# ===========================================================================

class TestGate3MetadataArtifactExists:
    """Gate 3: models/model_metadata.json must exist on disk."""

    def test_metadata_artifact_file_exists(self) -> None:
        """Gate 3: Verify model_metadata.json exists."""
        assert METADATA_PATH.exists(), (
            f"Gate 3 FAILED: Metadata artifact not found at {METADATA_PATH}\n"
            "Run: python scripts/build_model_artifacts.py"
        )


# ===========================================================================
# GATE 4 — Model loads successfully
# ===========================================================================

class TestGate4ModelLoadsSuccessfully:
    """Gate 4: Model deserializes without errors."""

    def test_model_loads_without_error(self) -> None:
        """Gate 4: ModelArtifacts.load_model() succeeds."""
        model = ModelArtifacts.load_model()
        assert model is not None, "Gate 4 FAILED: load_model() returned None."

    def test_model_is_sklearn_compatible(self) -> None:
        """Gate 4 (extended): Loaded model has standard sklearn attributes."""
        model = ModelArtifacts.load_model()
        assert hasattr(model, "fit"), "Model must have .fit() attribute."
        assert hasattr(model, "predict"), "Model must have .predict() attribute."


# ===========================================================================
# GATE 5 — Preprocessor loads successfully
# ===========================================================================

class TestGate5PreprocessorLoadsSuccessfully:
    """Gate 5: Preprocessor deserializes without errors."""

    def test_preprocessor_loads_without_error(self) -> None:
        """Gate 5: ModelArtifacts.load_preprocessor() succeeds."""
        preprocessor = ModelArtifacts.load_preprocessor()
        assert preprocessor is not None, "Gate 5 FAILED: load_preprocessor() returned None."

    def test_preprocessor_has_transform(self) -> None:
        """Gate 5 (extended): Preprocessor has .transform() method."""
        preprocessor = ModelArtifacts.load_preprocessor()
        assert hasattr(preprocessor, "transform"), (
            "Preprocessor must have .transform() method."
        )


# ===========================================================================
# GATE 6 — Model supports predict_proba()
# ===========================================================================

class TestGate6ModelSupportsPredictProba:
    """Gate 6: Loaded model must support predict_proba()."""

    def test_model_has_predict_proba(self) -> None:
        """Gate 6: Model has predict_proba() method."""
        model = ModelArtifacts.load_model()
        assert hasattr(model, "predict_proba"), (
            "Gate 6 FAILED: Model does not support predict_proba().\n"
            "Only probabilistic classifiers are accepted for PD estimation."
        )

    def test_predict_proba_is_callable(self) -> None:
        """Gate 6 (extended): predict_proba is callable."""
        model = ModelArtifacts.load_model()
        assert callable(getattr(model, "predict_proba", None)), (
            "predict_proba must be callable."
        )


# ===========================================================================
# GATE 7 — Feature dimensions match
# ===========================================================================

class TestGate7FeatureDimensionsMatch:
    """Gate 7: Preprocessor output dimensions must match model input dimensions."""

    def test_preprocessor_output_matches_model_input(self, valid_applicant) -> None:
        """Gate 7: Transform one valid applicant and confirm model accepts it."""
        import numpy as np
        import pandas as pd
        from src.prediction.schemas import UCI_ALL_FEATURES

        preprocessor = ModelArtifacts.load_preprocessor()
        model        = ModelArtifacts.load_model()

        # Transform a valid applicant
        df = pd.DataFrame([valid_applicant], columns=UCI_ALL_FEATURES)
        processed = preprocessor.transform(df)

        # Model must accept this shape
        proba = model.predict_proba(processed)
        assert proba.shape == (1, 2), (
            f"Gate 7 FAILED: Expected predict_proba shape (1, 2), "
            f"got {proba.shape}."
        )

    def test_model_n_features_matches_preprocessor_output(self, valid_applicant) -> None:
        """Gate 7 (extended): n_features_in_ of model matches processed feature count."""
        import pandas as pd
        from src.prediction.schemas import UCI_ALL_FEATURES

        preprocessor = ModelArtifacts.load_preprocessor()
        model        = ModelArtifacts.load_model()

        df = pd.DataFrame([valid_applicant], columns=UCI_ALL_FEATURES)
        processed = preprocessor.transform(df)
        n_processed = processed.shape[1]

        if hasattr(model, "n_features_in_"):
            assert model.n_features_in_ == n_processed, (
                f"Gate 7 FAILED: Model expects {model.n_features_in_} features "
                f"but preprocessor outputs {n_processed}."
            )


# ===========================================================================
# GATE 8 — Valid applicant prediction succeeds
# ===========================================================================

class TestGate8ValidPredictionSucceeds:
    """Gate 8: A valid applicant must produce a PredictionResult."""

    def test_valid_applicant_returns_result(
        self, predictor: CreditRiskPredictor, valid_applicant: dict
    ) -> None:
        """Gate 8: predictor.predict() returns a PredictionResult."""
        result = predictor.predict(valid_applicant)
        assert result is not None, "Gate 8 FAILED: predict() returned None."
        assert isinstance(result, PredictionResult), (
            f"Gate 8 FAILED: Expected PredictionResult, got {type(result).__name__}."
        )

    def test_valid_applicant_input_object_works(
        self, predictor: CreditRiskPredictor, valid_applicant: dict
    ) -> None:
        """Gate 8 (extended): ApplicantInput object also accepted."""
        applicant_obj = ApplicantInput.from_dict(valid_applicant)
        result = predictor.predict(applicant_obj)
        assert isinstance(result, PredictionResult)


# ===========================================================================
# GATE 9 — PD is within [0, 1]
# ===========================================================================

class TestGate9PDInRange:
    """Gate 9: estimated_pd must be in [0, 1]."""

    def test_pd_is_within_zero_to_one(
        self, predictor: CreditRiskPredictor, valid_applicant: dict
    ) -> None:
        """Gate 9: PD ∈ [0, 1]."""
        result = predictor.predict(valid_applicant)
        assert 0.0 <= result.estimated_pd <= 1.0, (
            f"Gate 9 FAILED: estimated_pd={result.estimated_pd} is outside [0, 1]."
        )

    def test_pd_is_not_nan(
        self, predictor: CreditRiskPredictor, valid_applicant: dict
    ) -> None:
        """Gate 9 (extended): PD must not be NaN."""
        result = predictor.predict(valid_applicant)
        assert not math.isnan(result.estimated_pd), "estimated_pd must not be NaN."

    def test_pd_is_not_infinite(
        self, predictor: CreditRiskPredictor, valid_applicant: dict
    ) -> None:
        """Gate 9 (extended): PD must not be infinite."""
        result = predictor.predict(valid_applicant)
        assert not math.isinf(result.estimated_pd), "estimated_pd must not be infinite."


# ===========================================================================
# GATE 10 — Risk score is within [0, 100]
# ===========================================================================

class TestGate10RiskScoreInRange:
    """Gate 10: risk_score must be in [0, 100]."""

    def test_risk_score_within_range(
        self, predictor: CreditRiskPredictor, valid_applicant: dict
    ) -> None:
        """Gate 10: risk_score ∈ [0, 100]."""
        result = predictor.predict(valid_applicant)
        assert 0.0 <= result.risk_score <= 100.0, (
            f"Gate 10 FAILED: risk_score={result.risk_score} is outside [0, 100]."
        )

    def test_risk_score_formula_consistency(
        self, predictor: CreditRiskPredictor, valid_applicant: dict
    ) -> None:
        """Gate 10 (extended): risk_score == (1 - estimated_pd) * 100 ± epsilon."""
        result = predictor.predict(valid_applicant)
        expected_score = (1.0 - result.estimated_pd) * 100.0
        assert abs(result.risk_score - expected_score) < 0.01, (
            f"Gate 10 FAILED: risk_score={result.risk_score} != "
            f"(1 - {result.estimated_pd}) * 100 = {expected_score:.4f}"
        )


# ===========================================================================
# GATE 11 — Risk tier is valid
# ===========================================================================

class TestGate11ValidRiskTier:
    """Gate 11: risk_tier must be LOW, MEDIUM, or HIGH."""

    def test_risk_tier_is_valid(
        self, predictor: CreditRiskPredictor, valid_applicant: dict
    ) -> None:
        """Gate 11: risk_tier ∈ {"LOW", "MEDIUM", "HIGH"}."""
        result = predictor.predict(valid_applicant)
        assert result.risk_tier in VALID_TIERS, (
            f"Gate 11 FAILED: risk_tier='{result.risk_tier}' not in {VALID_TIERS}."
        )

    def test_tier_boundary_low(self) -> None:
        """Gate 11 (extended): PD < 0.20 → LOW tier."""
        policy = RiskPolicy()
        assert policy.assign_tier(0.19) == "LOW"
        assert policy.assign_tier(0.00) == "LOW"

    def test_tier_boundary_medium(self) -> None:
        """Gate 11 (extended): 0.20 ≤ PD < 0.45 → MEDIUM tier."""
        policy = RiskPolicy()
        assert policy.assign_tier(0.20) == "MEDIUM"
        assert policy.assign_tier(0.30) == "MEDIUM"
        assert policy.assign_tier(0.44) == "MEDIUM"

    def test_tier_boundary_high(self) -> None:
        """Gate 11 (extended): PD ≥ 0.45 → HIGH tier."""
        policy = RiskPolicy()
        assert policy.assign_tier(0.45) == "HIGH"
        assert policy.assign_tier(1.00) == "HIGH"


# ===========================================================================
# GATE 12 — Decision is valid
# ===========================================================================

class TestGate12ValidDecision:
    """Gate 12: decision must be APPROVE, MANUAL REVIEW, or REJECT."""

    def test_decision_is_valid(
        self, predictor: CreditRiskPredictor, valid_applicant: dict
    ) -> None:
        """Gate 12: decision ∈ {"APPROVE", "MANUAL REVIEW", "REJECT"}."""
        result = predictor.predict(valid_applicant)
        assert result.decision in VALID_DECISIONS, (
            f"Gate 12 FAILED: decision='{result.decision}' not in {VALID_DECISIONS}."
        )

    def test_decision_mapping_low_to_approve(self) -> None:
        """Gate 12 (extended): LOW → APPROVE."""
        policy = RiskPolicy()
        assert policy.assign_decision("LOW") == "APPROVE"

    def test_decision_mapping_medium_to_manual_review(self) -> None:
        """Gate 12 (extended): MEDIUM → MANUAL REVIEW."""
        policy = RiskPolicy()
        assert policy.assign_decision("MEDIUM") == "MANUAL REVIEW"

    def test_decision_mapping_high_to_reject(self) -> None:
        """Gate 12 (extended): HIGH → REJECT."""
        policy = RiskPolicy()
        assert policy.assign_decision("HIGH") == "REJECT"


# ===========================================================================
# GATE 13 — Invalid input is rejected
# ===========================================================================

class TestGate13InvalidInputRejected:
    """Gate 13: Missing or invalid inputs must raise InputValidationError."""

    def test_missing_feature_raises_error(self, valid_applicant: dict) -> None:
        """Gate 13: Missing a required field raises InputValidationError."""
        incomplete = {k: v for k, v in valid_applicant.items()
                      if k != "age_years"}
        with pytest.raises(InputValidationError) as exc_info:
            ApplicantInput.from_dict(incomplete)
        assert "age_years" in str(exc_info.value), (
            "Error message must identify the missing field."
        )

    def test_invalid_categorical_raises_error(self, valid_applicant: dict) -> None:
        """Gate 13: Invalid categorical code raises InputValidationError."""
        bad_input = dict(valid_applicant)
        bad_input["credit_history"] = "INVALID_CODE"
        with pytest.raises(InputValidationError) as exc_info:
            ApplicantInput.from_dict(bad_input)
        assert "credit_history" in str(exc_info.value), (
            "Error message must identify the invalid field."
        )

    def test_invalid_checking_account_raises_error(self, valid_applicant: dict) -> None:
        """Gate 13 (extended): Invalid checking account code rejected."""
        bad_input = dict(valid_applicant)
        bad_input["status_existing_checking_account"] = "A99"
        with pytest.raises(InputValidationError):
            ApplicantInput.from_dict(bad_input)

    def test_negative_credit_amount_raises_error(self, valid_applicant: dict) -> None:
        """Gate 13 (extended): Credit amount outside valid range rejected."""
        bad_input = dict(valid_applicant)
        bad_input["credit_amount"] = -1000
        with pytest.raises(InputValidationError) as exc_info:
            ApplicantInput.from_dict(bad_input)
        assert "credit_amount" in str(exc_info.value)

    def test_age_below_range_raises_error(self, valid_applicant: dict) -> None:
        """Gate 13 (extended): Age below valid range rejected."""
        bad_input = dict(valid_applicant)
        bad_input["age_years"] = 10  # Below minimum of 18
        with pytest.raises(InputValidationError) as exc_info:
            ApplicantInput.from_dict(bad_input)
        assert "age_years" in str(exc_info.value)


# ===========================================================================
# GATE 14 — NaN / Inf input is rejected
# ===========================================================================

class TestGate14NaNInfRejected:
    """Gate 14: NaN and infinite values must be rejected cleanly."""

    def test_nan_in_numerical_raises_error(self, valid_applicant: dict) -> None:
        """Gate 14: NaN in a numerical field raises InputValidationError."""
        import math
        nan_input = dict(valid_applicant)
        nan_input["duration_in_months"] = math.nan
        with pytest.raises(InputValidationError) as exc_info:
            ApplicantInput.from_dict(nan_input)
        assert "duration_in_months" in str(exc_info.value)
        assert "NaN" in str(exc_info.value)

    def test_positive_inf_in_numerical_raises_error(self, valid_applicant: dict) -> None:
        """Gate 14: +Inf in a numerical field raises InputValidationError."""
        import math
        inf_input = dict(valid_applicant)
        inf_input["credit_amount"] = math.inf
        with pytest.raises(InputValidationError) as exc_info:
            ApplicantInput.from_dict(inf_input)
        assert "credit_amount" in str(exc_info.value)
        assert "infinite" in str(exc_info.value)

    def test_negative_inf_in_numerical_raises_error(self, valid_applicant: dict) -> None:
        """Gate 14: -Inf in a numerical field raises InputValidationError."""
        import math
        neg_inf_input = dict(valid_applicant)
        neg_inf_input["age_years"] = -math.inf
        with pytest.raises(InputValidationError) as exc_info:
            ApplicantInput.from_dict(neg_inf_input)
        assert "age_years" in str(exc_info.value)

    def test_nan_as_float_nan_raises_error(self, valid_applicant: dict) -> None:
        """Gate 14 (extended): float('nan') raises InputValidationError."""
        nan_input = dict(valid_applicant)
        nan_input["existing_credits_count"] = float("nan")
        with pytest.raises(InputValidationError):
            ApplicantInput.from_dict(nan_input)

    def test_inf_as_float_inf_raises_error(self, valid_applicant: dict) -> None:
        """Gate 14 (extended): float('inf') raises InputValidationError."""
        inf_input = dict(valid_applicant)
        inf_input["installment_rate_pct_disposable_income"] = float("inf")
        with pytest.raises(InputValidationError):
            ApplicantInput.from_dict(inf_input)


# ===========================================================================
# GATE 15 — Prediction does not refit model
# ===========================================================================

class TestGate15PredictionDoesNotRefit:
    """Gate 15: predict() must never call .fit() or .fit_transform()."""

    def test_predict_does_not_call_fit_on_model(
        self, predictor: CreditRiskPredictor, valid_applicant: dict
    ) -> None:
        """Gate 15: Monkey-patch model.fit() to detect if it is called."""
        fit_called = []

        original_fit = predictor._model.fit

        def spy_fit(*args, **kwargs):
            fit_called.append("FIT_CALLED")
            return original_fit(*args, **kwargs)

        predictor._model.fit = spy_fit

        try:
            predictor.predict(valid_applicant)
        finally:
            # Restore original
            predictor._model.fit = original_fit

        assert len(fit_called) == 0, (
            "Gate 15 FAILED: model.fit() was called during prediction. "
            "The prediction path must NEVER refit the model."
        )

    def test_predict_does_not_call_fit_on_preprocessor(
        self, predictor: CreditRiskPredictor, valid_applicant: dict
    ) -> None:
        """Gate 15 (extended): Preprocessor.fit() is never called during prediction."""
        fit_called = []

        original_fit = predictor._preprocessor.fit

        def spy_fit(*args, **kwargs):
            fit_called.append("FIT_CALLED")
            return original_fit(*args, **kwargs)

        predictor._preprocessor.fit = spy_fit

        try:
            predictor.predict(valid_applicant)
        finally:
            predictor._preprocessor.fit = original_fit

        assert len(fit_called) == 0, (
            "Gate 15 FAILED: preprocessor.fit() was called during prediction. "
            "Only .transform() is allowed on the prediction path."
        )

    def test_predict_does_not_call_fit_transform_on_preprocessor(
        self, predictor: CreditRiskPredictor, valid_applicant: dict
    ) -> None:
        """Gate 15 (extended): Preprocessor.fit_transform() never called during prediction."""
        fit_transform_called = []

        original = predictor._preprocessor.fit_transform

        def spy(*args, **kwargs):
            fit_transform_called.append("FIT_TRANSFORM_CALLED")
            return original(*args, **kwargs)

        predictor._preprocessor.fit_transform = spy

        try:
            predictor.predict(valid_applicant)
        finally:
            predictor._preprocessor.fit_transform = original

        assert len(fit_transform_called) == 0, (
            "Gate 15 FAILED: preprocessor.fit_transform() was called during prediction."
        )


# ===========================================================================
# Additional — Prediction output structure completeness
# ===========================================================================

class TestPredictionOutputStructure:
    """Verify the prediction result contains all required fields."""

    def test_result_has_estimated_pd(
        self, predictor: CreditRiskPredictor, valid_applicant: dict
    ) -> None:
        """Result must have estimated_pd."""
        result = predictor.predict(valid_applicant)
        assert hasattr(result, "estimated_pd")

    def test_result_has_risk_score(
        self, predictor: CreditRiskPredictor, valid_applicant: dict
    ) -> None:
        """Result must have risk_score."""
        result = predictor.predict(valid_applicant)
        assert hasattr(result, "risk_score")

    def test_result_has_risk_tier(
        self, predictor: CreditRiskPredictor, valid_applicant: dict
    ) -> None:
        """Result must have risk_tier."""
        result = predictor.predict(valid_applicant)
        assert hasattr(result, "risk_tier")

    def test_result_has_decision(
        self, predictor: CreditRiskPredictor, valid_applicant: dict
    ) -> None:
        """Result must have decision."""
        result = predictor.predict(valid_applicant)
        assert hasattr(result, "decision")

    def test_result_to_dict_contains_all_keys(
        self, predictor: CreditRiskPredictor, valid_applicant: dict
    ) -> None:
        """to_dict() must contain all 6 expected keys."""
        result = predictor.predict(valid_applicant)
        d = result.to_dict()
        required_keys = {
            "estimated_pd", "risk_score", "risk_tier",
            "decision", "model_name", "dataset"
        }
        assert required_keys.issubset(d.keys()), (
            f"Missing keys: {required_keys - set(d.keys())}"
        )
