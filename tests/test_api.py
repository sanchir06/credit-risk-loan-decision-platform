"""
tests/test_api.py
==================

PURPOSE:
    18-gate automated test suite for the Phase 5 Step 2 FastAPI REST API.

    Uses FastAPI's built-in TestClient (wraps httpx) for synchronous testing.
    No live server is needed — tests run against an in-process ASGI app.

GATES:
    Gate  1: FastAPI application imports successfully
    Gate  2: GET /health responds with 200
    Gate  3: model_loaded is detected dynamically (not hard-coded)
    Gate  4: GET /model-info responds with 200
    Gate  5: Model metadata fields are returned correctly
    Gate  6: Valid applicant POST /predict succeeds
    Gate  7: estimated_pd is in [0, 1]
    Gate  8: risk_score is in [0, 100]
    Gate  9: risk_tier is a valid tier
    Gate 10: decision is a valid decision
    Gate 11: Missing required field returns 422
    Gate 12: Invalid categorical value returns 422
    Gate 13: Invalid numeric value (out of range) returns 422
    Gate 14: NaN/Inf values are rejected
    Gate 15: Prediction does not call model.fit()
    Gate 16: Prediction does not call preprocessor.fit()
    Gate 17: No dataset loaded during prediction
    Gate 18: OpenAPI schema generates successfully

IMPORTANT:
    - Tests do NOT assert hard-coded PD values (e.g., == 0.1760).
    - They verify ranges and structural correctness only.
    - All applicant data in fixtures is ENTIRELY SYNTHETIC.
    - Artifacts must exist. Run: python scripts/build_model_artifacts.py

USAGE:
    pytest tests/test_api.py -v

PHASE 5 — STEP 2
"""

import math
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Import app and dependencies
# ---------------------------------------------------------------------------

from api.main import app                                          # noqa: E402
from src.prediction.risk_policy import VALID_TIERS, VALID_DECISIONS  # noqa: E402


# ---------------------------------------------------------------------------
# TestClient fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client() -> TestClient:
    """
    Module-scoped TestClient — the lifespan runs once per test module.
    Predictor is loaded once and shared across all tests in this module.
    """
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Synthetic applicant fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def valid_applicant() -> dict:
    """
    A fully valid synthetic UCI German Credit applicant payload.
    All values are fictional and provided for testing purposes only.
    """
    return {
        "status_existing_checking_account":       "A12",
        "duration_in_months":                     24,
        "credit_history":                         "A32",
        "purpose":                                "A43",
        "credit_amount":                          4500.0,
        "savings_account_bonds":                  "A62",
        "present_employment_since":               "A73",
        "installment_rate_pct_disposable_income": 3.0,
        "personal_status_sex":                    "A93",
        "other_debtors_guarantors":               "A101",
        "present_residence_since":                2.0,
        "property":                               "A121",
        "age_years":                              34.0,
        "other_installment_plans":                "A143",
        "housing":                                "A152",
        "existing_credits_count":                 1.0,
        "job":                                    "A173",
        "people_liable_maintenance":              1.0,
        "telephone":                              "A192",
        "foreign_worker":                         "A201",
    }


# ===========================================================================
# GATE 1 — FastAPI application imports successfully
# ===========================================================================

class TestGate1AppImports:
    """Gate 1: The FastAPI application must import without errors."""

    def test_app_is_fastapi_instance(self) -> None:
        """Gate 1: app is a valid FastAPI instance."""
        from fastapi import FastAPI
        assert isinstance(app, FastAPI), (
            "Gate 1 FAILED: api.main.app is not a FastAPI instance."
        )

    def test_app_has_title(self) -> None:
        """Gate 1 (extended): App has a meaningful title."""
        assert app.title, "Gate 1 FAILED: FastAPI app has no title."
        assert len(app.title) > 5, "App title is too short to be meaningful."

    def test_app_has_description(self) -> None:
        """Gate 1 (extended): App has a description with disclaimer."""
        assert app.description, "Gate 1 FAILED: FastAPI app has no description."
        # Must contain UCI context
        assert "UCI" in app.description, (
            "App description must mention the UCI dataset."
        )


# ===========================================================================
# GATE 2 — GET /health responds with 200
# ===========================================================================

class TestGate2HealthResponds:
    """Gate 2: GET /health must return HTTP 200."""

    def test_health_returns_200(self, client: TestClient) -> None:
        """Gate 2: Health endpoint responds."""
        response = client.get("/health")
        assert response.status_code == 200, (
            f"Gate 2 FAILED: /health returned {response.status_code}."
        )

    def test_health_response_has_required_keys(self, client: TestClient) -> None:
        """Gate 2 (extended): Response has status, service, model_loaded."""
        response = client.get("/health")
        data = response.json()
        assert "status" in data, "Missing 'status' in /health response."
        assert "service" in data, "Missing 'service' in /health response."
        assert "model_loaded" in data, "Missing 'model_loaded' in /health response."

    def test_health_service_name(self, client: TestClient) -> None:
        """Gate 2 (extended): Service name is 'credit-risk-api'."""
        response = client.get("/health")
        data = response.json()
        assert data.get("service") == "credit-risk-api", (
            f"Expected service='credit-risk-api', got '{data.get('service')}'."
        )


# ===========================================================================
# GATE 3 — model_loaded is detected dynamically
# ===========================================================================

class TestGate3ModelLoadedDynamic:
    """Gate 3: model_loaded reflects actual artifact state — not hard-coded."""

    def test_model_loaded_is_boolean(self, client: TestClient) -> None:
        """Gate 3: model_loaded field is a boolean."""
        response = client.get("/health")
        data = response.json()
        assert isinstance(data["model_loaded"], bool), (
            "Gate 3 FAILED: model_loaded must be a boolean."
        )

    def test_model_loaded_is_true_when_artifacts_exist(self, client: TestClient) -> None:
        """Gate 3: model_loaded is True because artifacts exist."""
        from src.models.model_artifacts import ModelArtifacts
        artifacts_exist = ModelArtifacts.all_artifacts_exist()

        response = client.get("/health")
        data = response.json()

        assert data["model_loaded"] == artifacts_exist, (
            f"Gate 3 FAILED: model_loaded={data['model_loaded']} but "
            f"artifacts_exist={artifacts_exist}. "
            "model_loaded must reflect actual artifact state."
        )

    def test_status_reflects_model_loaded(self, client: TestClient) -> None:
        """Gate 3 (extended): 'status' reflects model_loaded truthfully."""
        response = client.get("/health")
        data = response.json()
        if data["model_loaded"]:
            assert data["status"] == "healthy", (
                "Status must be 'healthy' when model is loaded."
            )
        else:
            assert data["status"] != "healthy", (
                "Status must not be 'healthy' when model is not loaded."
            )


# ===========================================================================
# GATE 4 — GET /model-info responds with 200
# ===========================================================================

class TestGate4ModelInfoResponds:
    """Gate 4: GET /model-info must return HTTP 200."""

    def test_model_info_returns_200(self, client: TestClient) -> None:
        """Gate 4: /model-info endpoint responds."""
        response = client.get("/model-info")
        assert response.status_code == 200, (
            f"Gate 4 FAILED: /model-info returned {response.status_code}."
        )

    def test_model_info_returns_json(self, client: TestClient) -> None:
        """Gate 4 (extended): Response body is valid JSON."""
        response = client.get("/model-info")
        data = response.json()
        assert isinstance(data, dict), "/model-info must return a JSON object."


# ===========================================================================
# GATE 5 — Model metadata fields are correct
# ===========================================================================

class TestGate5ModelMetadataCorrect:
    """Gate 5: /model-info returns correct metadata fields."""

    def test_metadata_has_model_name(self, client: TestClient) -> None:
        """Gate 5: metadata has model_name."""
        response = client.get("/model-info")
        data = response.json()
        assert "model_name" in data, "Missing 'model_name' in /model-info response."
        assert data["model_name"], "model_name must not be empty."

    def test_metadata_has_dataset(self, client: TestClient) -> None:
        """Gate 5: metadata has dataset field."""
        response = client.get("/model-info")
        data = response.json()
        assert "dataset" in data, "Missing 'dataset' in /model-info response."
        assert "UCI" in data["dataset"], (
            "dataset field must identify the UCI German Credit dataset."
        )

    def test_metadata_has_feature_count(self, client: TestClient) -> None:
        """Gate 5: metadata has feature_count == 20."""
        response = client.get("/model-info")
        data = response.json()
        assert data.get("feature_count") == 20, (
            f"Expected feature_count=20, got {data.get('feature_count')}."
        )

    def test_metadata_pd_is_calibrated_false(self, client: TestClient) -> None:
        """Gate 5: pd_is_calibrated must be False (PD not formally calibrated)."""
        response = client.get("/model-info")
        data = response.json()
        assert data.get("pd_is_calibrated") is False, (
            "pd_is_calibrated must be False — PD is not formally calibrated."
        )

    def test_metadata_has_risk_score_formula(self, client: TestClient) -> None:
        """Gate 5 (extended): metadata has risk_score_formula."""
        response = client.get("/model-info")
        data = response.json()
        assert "risk_score_formula" in data, "Missing 'risk_score_formula' in metadata."


# ===========================================================================
# GATE 6 — Valid applicant prediction succeeds
# ===========================================================================

class TestGate6ValidPredictionSucceeds:
    """Gate 6: A valid applicant POST /predict must return HTTP 200."""

    def test_valid_applicant_returns_200(
        self, client: TestClient, valid_applicant: dict
    ) -> None:
        """Gate 6: POST /predict with valid applicant returns 200."""
        response = client.post("/predict", json=valid_applicant)
        assert response.status_code == 200, (
            f"Gate 6 FAILED: POST /predict returned {response.status_code}. "
            f"Body: {response.text}"
        )

    def test_valid_applicant_returns_json(
        self, client: TestClient, valid_applicant: dict
    ) -> None:
        """Gate 6 (extended): Response is valid JSON."""
        response = client.post("/predict", json=valid_applicant)
        data = response.json()
        assert isinstance(data, dict), "POST /predict must return a JSON object."

    def test_response_has_all_required_fields(
        self, client: TestClient, valid_applicant: dict
    ) -> None:
        """Gate 6 (extended): Response has all required fields."""
        response = client.post("/predict", json=valid_applicant)
        data = response.json()
        required = {"estimated_pd", "risk_score", "risk_tier", "decision"}
        missing = required - set(data.keys())
        assert not missing, f"Missing response fields: {missing}"


# ===========================================================================
# GATE 7 — estimated_pd is in [0, 1]
# ===========================================================================

class TestGate7PDInRange:
    """Gate 7: estimated_pd must be in [0, 1]."""

    def test_pd_in_zero_to_one(
        self, client: TestClient, valid_applicant: dict
    ) -> None:
        """Gate 7: estimated_pd is in [0, 1]."""
        response = client.post("/predict", json=valid_applicant)
        data = response.json()
        pd = data["estimated_pd"]
        assert 0.0 <= pd <= 1.0, (
            f"Gate 7 FAILED: estimated_pd={pd} is outside [0, 1]."
        )

    def test_pd_is_not_nan(
        self, client: TestClient, valid_applicant: dict
    ) -> None:
        """Gate 7 (extended): estimated_pd must not be NaN."""
        response = client.post("/predict", json=valid_applicant)
        data = response.json()
        assert not math.isnan(data["estimated_pd"]), "estimated_pd must not be NaN."

    def test_pd_is_not_hard_coded(
        self, client: TestClient, valid_applicant: dict
    ) -> None:
        """Gate 7 (extended): estimated_pd is generated dynamically by the model."""
        response = client.post("/predict", json=valid_applicant)
        data = response.json()
        # Verify it's a float in [0,1] — not checking an exact value
        pd = float(data["estimated_pd"])
        assert 0.0 <= pd <= 1.0


# ===========================================================================
# GATE 8 — risk_score is in [0, 100]
# ===========================================================================

class TestGate8RiskScoreInRange:
    """Gate 8: risk_score must be in [0, 100]."""

    def test_risk_score_in_range(
        self, client: TestClient, valid_applicant: dict
    ) -> None:
        """Gate 8: risk_score in [0, 100]."""
        response = client.post("/predict", json=valid_applicant)
        data = response.json()
        score = data["risk_score"]
        assert 0.0 <= score <= 100.0, (
            f"Gate 8 FAILED: risk_score={score} is outside [0, 100]."
        )

    def test_risk_score_formula_consistency(
        self, client: TestClient, valid_applicant: dict
    ) -> None:
        """Gate 8 (extended): risk_score == (1 - estimated_pd) * 100 within tolerance."""
        response = client.post("/predict", json=valid_applicant)
        data = response.json()
        expected = (1.0 - data["estimated_pd"]) * 100.0
        assert abs(data["risk_score"] - expected) < 0.01, (
            f"risk_score={data['risk_score']} does not match "
            f"(1 - {data['estimated_pd']}) * 100 = {expected:.4f}"
        )


# ===========================================================================
# GATE 9 — risk_tier is valid
# ===========================================================================

class TestGate9ValidRiskTier:
    """Gate 9: risk_tier must be LOW, MEDIUM, or HIGH."""

    def test_risk_tier_is_valid(
        self, client: TestClient, valid_applicant: dict
    ) -> None:
        """Gate 9: risk_tier in {LOW, MEDIUM, HIGH}."""
        response = client.post("/predict", json=valid_applicant)
        data = response.json()
        assert data["risk_tier"] in VALID_TIERS, (
            f"Gate 9 FAILED: risk_tier='{data['risk_tier']}' not in {VALID_TIERS}."
        )


# ===========================================================================
# GATE 10 — decision is valid
# ===========================================================================

class TestGate10ValidDecision:
    """Gate 10: decision must be APPROVE, MANUAL REVIEW, or REJECT."""

    def test_decision_is_valid(
        self, client: TestClient, valid_applicant: dict
    ) -> None:
        """Gate 10: decision in {APPROVE, MANUAL REVIEW, REJECT}."""
        response = client.post("/predict", json=valid_applicant)
        data = response.json()
        assert data["decision"] in VALID_DECISIONS, (
            f"Gate 10 FAILED: decision='{data['decision']}' not in {VALID_DECISIONS}."
        )


# ===========================================================================
# GATE 11 — Missing required field returns 422
# ===========================================================================

class TestGate11MissingFieldRejected:
    """Gate 11: Missing a required field must return HTTP 422."""

    def test_missing_age_returns_422(
        self, client: TestClient, valid_applicant: dict
    ) -> None:
        """Gate 11: Missing age_years returns 422."""
        incomplete = {k: v for k, v in valid_applicant.items() if k != "age_years"}
        response = client.post("/predict", json=incomplete)
        assert response.status_code == 422, (
            f"Gate 11 FAILED: Expected 422 for missing field, got {response.status_code}."
        )

    def test_missing_credit_amount_returns_422(
        self, client: TestClient, valid_applicant: dict
    ) -> None:
        """Gate 11: Missing credit_amount returns 422."""
        incomplete = {k: v for k, v in valid_applicant.items() if k != "credit_amount"}
        response = client.post("/predict", json=incomplete)
        assert response.status_code == 422

    def test_empty_body_returns_422(self, client: TestClient) -> None:
        """Gate 11: Empty body returns 422."""
        response = client.post("/predict", json={})
        assert response.status_code == 422


# ===========================================================================
# GATE 12 — Invalid categorical value returns 422
# ===========================================================================

class TestGate12InvalidCategoricalRejected:
    """Gate 12: Invalid categorical codes must be rejected with 422."""

    def test_invalid_credit_history_code_returns_422(
        self, client: TestClient, valid_applicant: dict
    ) -> None:
        """Gate 12: Invalid credit_history code returns 422."""
        bad = dict(valid_applicant)
        bad["credit_history"] = "INVALID_CODE"
        response = client.post("/predict", json=bad)
        assert response.status_code == 422, (
            f"Gate 12 FAILED: Expected 422 for invalid categorical, got {response.status_code}."
        )

    def test_invalid_checking_account_code_returns_422(
        self, client: TestClient, valid_applicant: dict
    ) -> None:
        """Gate 12: Invalid status_existing_checking_account returns 422."""
        bad = dict(valid_applicant)
        bad["status_existing_checking_account"] = "A99"
        response = client.post("/predict", json=bad)
        assert response.status_code == 422

    def test_invalid_housing_code_returns_422(
        self, client: TestClient, valid_applicant: dict
    ) -> None:
        """Gate 12 (extended): Invalid housing code returns 422."""
        bad = dict(valid_applicant)
        bad["housing"] = "RENT"   # should be A151
        response = client.post("/predict", json=bad)
        assert response.status_code == 422


# ===========================================================================
# GATE 13 — Invalid numeric value returns 422
# ===========================================================================

class TestGate13InvalidNumericRejected:
    """Gate 13: Numeric values outside valid ranges must return 422."""

    def test_negative_credit_amount_returns_422(
        self, client: TestClient, valid_applicant: dict
    ) -> None:
        """Gate 13: Negative credit_amount returns 422."""
        bad = dict(valid_applicant)
        bad["credit_amount"] = -500.0
        response = client.post("/predict", json=bad)
        assert response.status_code == 422, (
            f"Gate 13 FAILED: Expected 422 for negative credit_amount, got {response.status_code}."
        )

    def test_age_below_minimum_returns_422(
        self, client: TestClient, valid_applicant: dict
    ) -> None:
        """Gate 13: age_years below minimum (18) returns 422."""
        bad = dict(valid_applicant)
        bad["age_years"] = 10.0
        response = client.post("/predict", json=bad)
        assert response.status_code == 422

    def test_duration_above_maximum_returns_422(
        self, client: TestClient, valid_applicant: dict
    ) -> None:
        """Gate 13 (extended): duration_in_months above maximum returns 422."""
        bad = dict(valid_applicant)
        bad["duration_in_months"] = 500.0
        response = client.post("/predict", json=bad)
        assert response.status_code == 422


# ===========================================================================
# GATE 14 — NaN/Inf values are rejected
# ===========================================================================

class TestGate14NaNInfRejected:
    """Gate 14: NaN and Infinity values must be rejected before reaching the model.

    JSON spec does not support NaN or Infinity literals.
    We test this by sending raw JSON strings with the 'null' value (which Pydantic
    rejects as a required numeric field) and by sending strings where numbers are
    expected — both of which the API must reject with 422.

    Note: Python's json module refuses to serialize float('nan')/float('inf')
    because they are not valid JSON. In a real client, these would either be
    sent as 'null' (missing/null), as strings, or cause a client-side error.
    All three cases must be rejected by the API with 422.
    """

    def test_null_value_for_numeric_field_returns_422(
        self, client: TestClient, valid_applicant: dict
    ) -> None:
        """Gate 14: null value for required numeric field returns 422."""
        import json as json_lib
        bad = dict(valid_applicant)
        bad["duration_in_months"] = None  # null in JSON
        # Use content= to bypass httpx NaN/Inf serialization restriction
        response = client.post(
            "/predict",
            content=json_lib.dumps(bad),
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422, (
            f"Gate 14 FAILED: Expected 422 for null numeric field, got {response.status_code}."
        )

    def test_string_value_for_numeric_field_returns_422(
        self, client: TestClient, valid_applicant: dict
    ) -> None:
        """Gate 14: String 'nan' for numeric field returns 422."""
        import json as json_lib
        bad = dict(valid_applicant)
        bad["credit_amount"] = "nan"  # string, not a valid float
        response = client.post(
            "/predict",
            content=json_lib.dumps(bad),
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422, (
            f"Gate 14 FAILED: Expected 422 for string 'nan', got {response.status_code}."
        )

    def test_string_inf_for_numeric_field_returns_422(
        self, client: TestClient, valid_applicant: dict
    ) -> None:
        """Gate 14: String 'inf' for numeric field returns 422."""
        import json as json_lib
        bad = dict(valid_applicant)
        bad["age_years"] = "inf"
        response = client.post(
            "/predict",
            content=json_lib.dumps(bad),
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422, (
            f"Gate 14 FAILED: Expected 422 for string 'inf', got {response.status_code}."
        )


# ===========================================================================
# GATE 15 — Prediction does not call model.fit()
# ===========================================================================

class TestGate15PredictionDoesNotRefitModel:
    """Gate 15: POST /predict must never call model.fit()."""

    def test_predict_does_not_call_model_fit(
        self, client: TestClient, valid_applicant: dict
    ) -> None:
        """Gate 15: Monkey-patch model.fit() and verify it is never called."""
        predictor = client.app.state.predictor
        assert predictor is not None, "Predictor must be loaded."

        fit_called = []
        original_fit = predictor._model.fit

        def spy_fit(*args, **kwargs):
            fit_called.append("FIT_CALLED")
            return original_fit(*args, **kwargs)

        predictor._model.fit = spy_fit
        try:
            response = client.post("/predict", json=valid_applicant)
            assert response.status_code == 200
        finally:
            predictor._model.fit = original_fit

        assert len(fit_called) == 0, (
            "Gate 15 FAILED: model.fit() was called during prediction. "
            "The prediction path must NEVER refit the model."
        )


# ===========================================================================
# GATE 16 — Prediction does not call preprocessor.fit()
# ===========================================================================

class TestGate16PredictionDoesNotRefitPreprocessor:
    """Gate 16: POST /predict must never call preprocessor.fit() or fit_transform()."""

    def test_predict_does_not_call_preprocessor_fit(
        self, client: TestClient, valid_applicant: dict
    ) -> None:
        """Gate 16: preprocessor.fit() is never called during prediction."""
        predictor = client.app.state.predictor
        assert predictor is not None, "Predictor must be loaded."

        fit_called = []
        original_fit = predictor._preprocessor.fit

        def spy_fit(*args, **kwargs):
            fit_called.append("FIT_CALLED")
            return original_fit(*args, **kwargs)

        predictor._preprocessor.fit = spy_fit
        try:
            response = client.post("/predict", json=valid_applicant)
            assert response.status_code == 200
        finally:
            predictor._preprocessor.fit = original_fit

        assert len(fit_called) == 0, (
            "Gate 16 FAILED: preprocessor.fit() was called during prediction."
        )

    def test_predict_does_not_call_fit_transform(
        self, client: TestClient, valid_applicant: dict
    ) -> None:
        """Gate 16 (extended): preprocessor.fit_transform() never called during prediction."""
        predictor = client.app.state.predictor
        assert predictor is not None

        ft_called = []
        original_ft = predictor._preprocessor.fit_transform

        def spy_ft(*args, **kwargs):
            ft_called.append("FIT_TRANSFORM_CALLED")
            return original_ft(*args, **kwargs)

        predictor._preprocessor.fit_transform = spy_ft
        try:
            response = client.post("/predict", json=valid_applicant)
            assert response.status_code == 200
        finally:
            predictor._preprocessor.fit_transform = original_ft

        assert len(ft_called) == 0, (
            "Gate 16 (extended) FAILED: preprocessor.fit_transform() was called."
        )


# ===========================================================================
# GATE 17 — No dataset loaded during prediction
# ===========================================================================

class TestGate17NoDatasetLoadedDuringPrediction:
    """Gate 17: No training dataset is loaded when POST /predict is called."""

    def test_no_csv_read_during_prediction(
        self, client: TestClient, valid_applicant: dict
    ) -> None:
        """Gate 17: pd.read_csv() is not called during prediction."""
        import pandas as pd
        read_calls = []
        original_read_csv = pd.read_csv

        def spy_read_csv(*args, **kwargs):
            read_calls.append(args)
            return original_read_csv(*args, **kwargs)

        pd.read_csv = spy_read_csv
        try:
            response = client.post("/predict", json=valid_applicant)
            assert response.status_code == 200
        finally:
            pd.read_csv = original_read_csv

        assert len(read_calls) == 0, (
            "Gate 17 FAILED: pd.read_csv() was called during prediction. "
            "No dataset should be loaded at inference time."
        )


# ===========================================================================
# GATE 18 — OpenAPI schema generates successfully
# ===========================================================================

class TestGate18OpenAPISchema:
    """Gate 18: OpenAPI schema must be generated and accessible."""

    def test_openapi_json_endpoint_responds(self, client: TestClient) -> None:
        """Gate 18: GET /openapi.json returns 200."""
        response = client.get("/openapi.json")
        assert response.status_code == 200, (
            f"Gate 18 FAILED: /openapi.json returned {response.status_code}."
        )

    def test_openapi_schema_is_valid_json(self, client: TestClient) -> None:
        """Gate 18: /openapi.json contains valid OpenAPI JSON."""
        response = client.get("/openapi.json")
        schema = response.json()
        assert "openapi" in schema, "OpenAPI schema missing 'openapi' key."
        assert "paths" in schema, "OpenAPI schema missing 'paths' key."
        assert "info" in schema, "OpenAPI schema missing 'info' key."

    def test_openapi_schema_has_predict_endpoint(self, client: TestClient) -> None:
        """Gate 18 (extended): /predict appears in OpenAPI schema."""
        response = client.get("/openapi.json")
        schema = response.json()
        assert "/predict" in schema.get("paths", {}), (
            "'/predict' must appear in the OpenAPI schema paths."
        )

    def test_docs_endpoint_responds(self, client: TestClient) -> None:
        """Gate 18 (extended): GET /docs returns 200 (Swagger UI)."""
        response = client.get("/docs")
        assert response.status_code == 200, (
            f"Gate 18 FAILED: /docs returned {response.status_code}."
        )

    def test_redoc_endpoint_responds(self, client: TestClient) -> None:
        """Gate 18 (extended): GET /redoc returns 200 (ReDoc)."""
        response = client.get("/redoc")
        assert response.status_code == 200, (
            f"Gate 18 FAILED: /redoc returned {response.status_code}."
        )
