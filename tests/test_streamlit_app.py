"""
tests/test_streamlit_app.py
============================

PURPOSE:
    16-test validation suite for Phase 5 Step 3 — Streamlit Dashboard.

    Tests verify:
        - The streamlit_app module imports successfully
        - API URL configuration exists and is correct
        - Integration logic calls the correct API endpoints
        - The 20-feature UCI payload is built correctly
        - No training code (no .fit() calls) exists
        - No CSV dataset loading occurs
        - HTTP 200, 422, 503 responses are handled correctly
        - Connection errors are handled gracefully
        - Prediction response fields are processed correctly
        - Disclaimer text exists and contains the required statements
        - Indian-specific fields are absent

APPROACH:
    All tests run without a live FastAPI server or Streamlit session.
    The `streamlit` module is mocked before import so that st.* calls
    in streamlit_app.py do not fail or render anything.
    HTTP calls (httpx) are mocked using unittest.mock.patch.

USAGE:
    pytest tests/test_streamlit_app.py -v

PHASE 5 — STEP 3
"""

import sys
import importlib
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
import pytest

# ---------------------------------------------------------------------------
# Path setup — ensure project root on sys.path
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Mock streamlit BEFORE importing streamlit_app
# ---------------------------------------------------------------------------

def _install_streamlit_mock() -> MagicMock:
    """
    Install a MagicMock for the streamlit module so that streamlit_app.py
    can be imported without an active Streamlit server.

    The mock is stored in sys.modules['streamlit'] before the import.
    st.session_state is a real dict-like object to allow session_state access.
    """
    mock_st = MagicMock()

    # session_state must behave like a dict for 'key in st.session_state'
    class _FakeSessionState(dict):
        def __getattr__(self, name):
            return self.get(name)

        def __setattr__(self, name, value):
            if name.startswith("_"):
                super().__setattr__(name, value)
            else:
                self[name] = value

    mock_st.session_state = _FakeSessionState()

    # st.columns returns a list of context managers — size matches argument
    mock_st.columns.side_effect = lambda n, **kw: [MagicMock() for _ in range(n if isinstance(n, int) else len(n))]

    # st.expander returns a context manager
    mock_expander = MagicMock()
    mock_expander.__enter__ = MagicMock(return_value=MagicMock())
    mock_expander.__exit__ = MagicMock(return_value=False)
    mock_st.expander.return_value = mock_expander

    # st.sidebar is also a context manager
    mock_sidebar = MagicMock()
    mock_sidebar.__enter__ = MagicMock(return_value=MagicMock())
    mock_sidebar.__exit__ = MagicMock(return_value=False)
    mock_st.sidebar = mock_sidebar

    # st.spinner is a context manager
    mock_spinner = MagicMock()
    mock_spinner.__enter__ = MagicMock(return_value=None)
    mock_spinner.__exit__ = MagicMock(return_value=False)
    mock_st.spinner.return_value = mock_spinner

    # st.button returns False by default (not clicked)
    mock_st.button.return_value = False

    # st.text_input returns the default value
    mock_st.text_input.return_value = "http://127.0.0.1:8000"

    # st.number_input returns the default value
    mock_st.number_input.return_value = 1.0

    # st.selectbox returns the first option
    mock_st.selectbox.return_value = None  # tests set return_value as needed

    sys.modules["streamlit"] = mock_st
    return mock_st


# Install mock before any import of streamlit_app
_mock_st = _install_streamlit_mock()

# Now import the module under test
import streamlit_app  # noqa: E402


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

VALID_FORM_VALUES = {
    "status_existing_checking_account":       "A12",
    "duration_in_months":                     24.0,
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

MOCK_HEALTH_RESPONSE = {
    "status": "healthy",
    "service": "credit-risk-api",
    "model_loaded": True,
}

MOCK_MODEL_INFO = {
    "model_name": "Logistic Regression",
    "model_version": "1.0.0",
    "dataset": "UCI Statlog German Credit Data",
    "feature_count": 20,
    "processed_feature_count": 61,
    "training_records": 800,
    "risk_score_formula": "(1 - estimated_pd) * 100",
    "pd_is_calibrated": False,
    "purpose": "historical benchmark demonstration",
    "context_note": "Historical European banking benchmark.",
}

MOCK_PREDICTION_RESPONSE = {
    "estimated_pd": 0.1760,
    "risk_score":   82.40,
    "risk_tier":    "LOW",
    "decision":     "APPROVE",
    "model_name":   "Logistic Regression",
    "dataset":      "UCI Statlog German Credit Data",
}


def _make_mock_response(status_code: int, json_data: dict) -> MagicMock:
    """Create a mock httpx.Response with the given status_code and JSON."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.text = str(json_data)
    return resp


# ===========================================================================
# TEST 1 — Streamlit module imports successfully
# ===========================================================================

class TestGate1Import:
    """Gate 1: streamlit_app must import without error (with mocked streamlit)."""

    def test_streamlit_app_imports_successfully(self) -> None:
        """Gate 1: Module import succeeds."""
        assert streamlit_app is not None, (
            "Gate 1 FAILED: streamlit_app failed to import."
        )

    def test_module_is_a_module(self) -> None:
        """Gate 1 (extended): streamlit_app is a proper Python module."""
        import types
        assert isinstance(streamlit_app, types.ModuleType), (
            "streamlit_app must be a Python module."
        )


# ===========================================================================
# TEST 2 — API URL configuration exists
# ===========================================================================

class TestGate2APIURLConfiguration:
    """Gate 2: API_DEFAULT_URL constant must exist and have a correct default."""

    def test_api_default_url_constant_exists(self) -> None:
        """Gate 2: API_DEFAULT_URL constant is defined."""
        assert hasattr(streamlit_app, "API_DEFAULT_URL"), (
            "Gate 2 FAILED: API_DEFAULT_URL not found in streamlit_app."
        )

    def test_api_default_url_value(self) -> None:
        """Gate 2: Default URL points to localhost:8000."""
        assert "127.0.0.1:8000" in streamlit_app.API_DEFAULT_URL or \
               "localhost:8000" in streamlit_app.API_DEFAULT_URL, (
            f"Gate 2 FAILED: Unexpected default URL '{streamlit_app.API_DEFAULT_URL}'."
        )

    def test_api_default_url_is_string(self) -> None:
        """Gate 2 (extended): API_DEFAULT_URL is a string."""
        assert isinstance(streamlit_app.API_DEFAULT_URL, str)


# ===========================================================================
# TEST 3 — /health endpoint integration
# ===========================================================================

class TestGate3HealthEndpointIntegration:
    """Gate 3: check_api_health() must exist and call GET /health."""

    def test_check_api_health_function_exists(self) -> None:
        """Gate 3: check_api_health is defined."""
        assert hasattr(streamlit_app, "check_api_health"), (
            "Gate 3 FAILED: check_api_health() not found in streamlit_app."
        )
        assert callable(streamlit_app.check_api_health)

    def test_check_api_health_calls_health_endpoint(self) -> None:
        """Gate 3: check_api_health() calls GET .../health."""
        mock_resp = _make_mock_response(200, MOCK_HEALTH_RESPONSE)
        with patch("streamlit_app.httpx.get", return_value=mock_resp) as mock_get:
            result = streamlit_app.check_api_health("http://127.0.0.1:8000")
        call_url = mock_get.call_args[0][0]
        assert "/health" in call_url, (
            f"Gate 3 FAILED: Expected /health in URL, got '{call_url}'."
        )

    def test_check_api_health_returns_dict(self) -> None:
        """Gate 3 (extended): Returns a dict with model_loaded field."""
        mock_resp = _make_mock_response(200, MOCK_HEALTH_RESPONSE)
        with patch("streamlit_app.httpx.get", return_value=mock_resp):
            result = streamlit_app.check_api_health("http://127.0.0.1:8000")
        assert isinstance(result, dict)
        assert "model_loaded" in result


# ===========================================================================
# TEST 4 — /model-info endpoint integration
# ===========================================================================

class TestGate4ModelInfoEndpointIntegration:
    """Gate 4: get_model_info() must exist and call GET /model-info."""

    def test_get_model_info_function_exists(self) -> None:
        """Gate 4: get_model_info is defined."""
        assert hasattr(streamlit_app, "get_model_info"), (
            "Gate 4 FAILED: get_model_info() not found in streamlit_app."
        )

    def test_get_model_info_calls_model_info_endpoint(self) -> None:
        """Gate 4: get_model_info() calls GET .../model-info."""
        mock_resp = _make_mock_response(200, MOCK_MODEL_INFO)
        with patch("streamlit_app.httpx.get", return_value=mock_resp) as mock_get:
            result = streamlit_app.get_model_info("http://127.0.0.1:8000")
        call_url = mock_get.call_args[0][0]
        assert "/model-info" in call_url, (
            f"Gate 4 FAILED: Expected /model-info in URL, got '{call_url}'."
        )


# ===========================================================================
# TEST 5 — /predict endpoint integration
# ===========================================================================

class TestGate5PredictEndpointIntegration:
    """Gate 5: send_prediction() must exist and call POST /predict."""

    def test_send_prediction_function_exists(self) -> None:
        """Gate 5: send_prediction is defined."""
        assert hasattr(streamlit_app, "send_prediction"), (
            "Gate 5 FAILED: send_prediction() not found in streamlit_app."
        )

    def test_send_prediction_calls_predict_endpoint(self) -> None:
        """Gate 5: send_prediction() calls POST .../predict."""
        mock_resp = _make_mock_response(200, MOCK_PREDICTION_RESPONSE)
        with patch("streamlit_app.httpx.post", return_value=mock_resp) as mock_post:
            result = streamlit_app.send_prediction(
                "http://127.0.0.1:8000",
                VALID_FORM_VALUES,
            )
        call_url = mock_post.call_args[0][0]
        assert "/predict" in call_url, (
            f"Gate 5 FAILED: Expected /predict in URL, got '{call_url}'."
        )


# ===========================================================================
# TEST 6 — Payload has exactly 20 UCI features
# ===========================================================================

class TestGate6PayloadHas20Features:
    """Gate 6: build_applicant_payload() returns exactly 20 features."""

    def test_build_applicant_payload_exists(self) -> None:
        """Gate 6: build_applicant_payload is defined."""
        assert hasattr(streamlit_app, "build_applicant_payload"), (
            "Gate 6 FAILED: build_applicant_payload() not found."
        )

    def test_payload_has_exactly_20_features(self) -> None:
        """Gate 6: Payload has exactly 20 keys."""
        payload = streamlit_app.build_applicant_payload(VALID_FORM_VALUES)
        assert len(payload) == 20, (
            f"Gate 6 FAILED: Expected 20 features, got {len(payload)}."
        )

    def test_required_uci_features_constant_has_20_items(self) -> None:
        """Gate 6 (extended): REQUIRED_UCI_FEATURES constant has 20 entries."""
        assert len(streamlit_app.REQUIRED_UCI_FEATURES) == 20, (
            f"Expected 20 UCI features, got {len(streamlit_app.REQUIRED_UCI_FEATURES)}."
        )


# ===========================================================================
# TEST 7 — Payload has correct UCI feature names
# ===========================================================================

class TestGate7PayloadHasCorrectFeatureNames:
    """Gate 7: Payload keys must exactly match the 20 UCI feature names."""

    EXPECTED_FEATURES = {
        "status_existing_checking_account",
        "duration_in_months",
        "credit_history",
        "purpose",
        "credit_amount",
        "savings_account_bonds",
        "present_employment_since",
        "installment_rate_pct_disposable_income",
        "personal_status_sex",
        "other_debtors_guarantors",
        "present_residence_since",
        "property",
        "age_years",
        "other_installment_plans",
        "housing",
        "existing_credits_count",
        "job",
        "people_liable_maintenance",
        "telephone",
        "foreign_worker",
    }

    def test_payload_keys_match_uci_features(self) -> None:
        """Gate 7: All payload keys are valid UCI feature names."""
        payload = streamlit_app.build_applicant_payload(VALID_FORM_VALUES)
        payload_keys = set(payload.keys())
        assert payload_keys == self.EXPECTED_FEATURES, (
            f"Gate 7 FAILED: Payload key mismatch.\n"
            f"  Extra:   {payload_keys - self.EXPECTED_FEATURES}\n"
            f"  Missing: {self.EXPECTED_FEATURES - payload_keys}"
        )


# ===========================================================================
# TEST 8 — No training operations in source
# ===========================================================================

class TestGate8NoTrainingInSource:
    """Gate 8: streamlit_app.py must not contain any model training calls."""

    @pytest.fixture(scope="class")
    def source_code(self, tmp_path_factory) -> str:
        source_path = PROJECT_ROOT / "streamlit_app.py"
        return source_path.read_text(encoding="utf-8")

    def test_no_fit_call_in_source(self, source_code: str) -> None:
        """Gate 8: No .fit( call in streamlit_app.py."""
        # Exclude comments and docstrings by checking non-comment lines
        code_lines = [
            line for line in source_code.splitlines()
            if not line.strip().startswith("#") and ".fit(" in line
        ]
        assert len(code_lines) == 0, (
            f"Gate 8 FAILED: Found .fit() call(s) in streamlit_app.py:\n"
            + "\n".join(code_lines)
        )

    def test_no_fit_transform_in_source(self, source_code: str) -> None:
        """Gate 8 (extended): No .fit_transform( in streamlit_app.py."""
        lines = [l for l in source_code.splitlines() if ".fit_transform(" in l]
        assert not lines, f"Found .fit_transform() in source: {lines}"


# ===========================================================================
# TEST 9 — No CSV dataset loading in source
# ===========================================================================

class TestGate9NoCsvLoadingInSource:
    """Gate 9: streamlit_app.py must not call pd.read_csv() or load any dataset."""

    @pytest.fixture(scope="class")
    def source_code(self) -> str:
        return (PROJECT_ROOT / "streamlit_app.py").read_text(encoding="utf-8")

    def test_no_read_csv_in_source(self, source_code: str) -> None:
        """Gate 9: No pd.read_csv( in streamlit_app.py."""
        lines = [l for l in source_code.splitlines() if "read_csv" in l]
        assert not lines, (
            f"Gate 9 FAILED: Found read_csv() in streamlit_app.py: {lines}"
        )

    def test_no_raw_data_path_in_source(self, source_code: str) -> None:
        """Gate 9 (extended): No raw data file paths referenced in source."""
        assert "german_credit_1000.csv" not in source_code, (
            "Raw dataset path found in streamlit_app.py."
        )
        assert "credit_risk_50.csv" not in source_code, (
            "Raw dataset path found in streamlit_app.py."
        )


# ===========================================================================
# TEST 10 — HTTP 200 response handled correctly
# ===========================================================================

class TestGate10Http200Handled:
    """Gate 10: send_prediction() must correctly parse a 200 response."""

    def test_http_200_returns_parsed_dict(self) -> None:
        """Gate 10: HTTP 200 returns the parsed prediction dict."""
        mock_resp = _make_mock_response(200, MOCK_PREDICTION_RESPONSE)
        with patch("streamlit_app.httpx.post", return_value=mock_resp):
            result = streamlit_app.send_prediction(
                "http://127.0.0.1:8000",
                VALID_FORM_VALUES,
            )
        assert result == MOCK_PREDICTION_RESPONSE, (
            f"Gate 10 FAILED: Expected {MOCK_PREDICTION_RESPONSE}, got {result}."
        )

    def test_http_200_result_has_required_keys(self) -> None:
        """Gate 10 (extended): 200 response includes all required keys."""
        mock_resp = _make_mock_response(200, MOCK_PREDICTION_RESPONSE)
        with patch("streamlit_app.httpx.post", return_value=mock_resp):
            result = streamlit_app.send_prediction(
                "http://127.0.0.1:8000",
                VALID_FORM_VALUES,
            )
        required = {"estimated_pd", "risk_score", "risk_tier", "decision"}
        assert required.issubset(set(result.keys())), (
            f"Missing keys: {required - set(result.keys())}"
        )


# ===========================================================================
# TEST 11 — HTTP 422 handled correctly
# ===========================================================================

class TestGate11Http422Handled:
    """Gate 11: send_prediction() must raise ValueError on HTTP 422."""

    def test_http_422_raises_value_error(self) -> None:
        """Gate 11: HTTP 422 raises ValueError."""
        mock_resp = _make_mock_response(422, {
            "error": "input_validation_error",
            "detail": "Missing required field: 'age_years'.",
        })
        with patch("streamlit_app.httpx.post", return_value=mock_resp):
            with pytest.raises(ValueError) as exc_info:
                streamlit_app.send_prediction(
                    "http://127.0.0.1:8000",
                    VALID_FORM_VALUES,
                )
        assert "invalid" in str(exc_info.value).lower() or \
               "check" in str(exc_info.value).lower(), (
            "Gate 11 FAILED: ValueError message does not describe invalid input."
        )


# ===========================================================================
# TEST 12 — HTTP 503 handled correctly
# ===========================================================================

class TestGate12Http503Handled:
    """Gate 12: send_prediction() must raise RuntimeError on HTTP 503."""

    def test_http_503_raises_runtime_error(self) -> None:
        """Gate 12: HTTP 503 raises RuntimeError with user-friendly message."""
        mock_resp = _make_mock_response(503, {
            "error": "model_unavailable",
            "detail": "Model artifacts not found.",
        })
        with patch("streamlit_app.httpx.post", return_value=mock_resp):
            with pytest.raises(RuntimeError) as exc_info:
                streamlit_app.send_prediction(
                    "http://127.0.0.1:8000",
                    VALID_FORM_VALUES,
                )
        error_msg = str(exc_info.value).lower()
        assert "unavailable" in error_msg or "artifact" in error_msg, (
            "Gate 12 FAILED: RuntimeError message does not describe unavailable service."
        )


# ===========================================================================
# TEST 13 — Connection failure handled
# ===========================================================================

class TestGate13ConnectionFailureHandled:
    """Gate 13: httpx.ConnectError must propagate (caller handles it cleanly)."""

    def test_connect_error_propagates_from_send_prediction(self) -> None:
        """Gate 13: ConnectError is not swallowed silently."""
        with patch(
            "streamlit_app.httpx.post",
            side_effect=streamlit_app.httpx.ConnectError("refused"),
        ):
            with pytest.raises(streamlit_app.httpx.ConnectError):
                streamlit_app.send_prediction(
                    "http://127.0.0.1:8000",
                    VALID_FORM_VALUES,
                )

    def test_connect_error_propagates_from_health_check(self) -> None:
        """Gate 13 (extended): check_api_health propagates ConnectError."""
        with patch(
            "streamlit_app.httpx.get",
            side_effect=streamlit_app.httpx.ConnectError("refused"),
        ):
            with pytest.raises(streamlit_app.httpx.ConnectError):
                streamlit_app.check_api_health("http://127.0.0.1:8000")


# ===========================================================================
# TEST 14 — Prediction response fields processed correctly
# ===========================================================================

class TestGate14PredictionResponseFields:
    """Gate 14: Prediction response must include estimated_pd, risk_score, tier, decision."""

    def test_estimated_pd_in_response(self) -> None:
        """Gate 14: estimated_pd is present and in [0, 1]."""
        mock_resp = _make_mock_response(200, MOCK_PREDICTION_RESPONSE)
        with patch("streamlit_app.httpx.post", return_value=mock_resp):
            result = streamlit_app.send_prediction(
                "http://127.0.0.1:8000",
                VALID_FORM_VALUES,
            )
        pd_val = result["estimated_pd"]
        assert 0.0 <= pd_val <= 1.0, f"estimated_pd={pd_val} outside [0, 1]."

    def test_risk_score_in_response(self) -> None:
        """Gate 14: risk_score is present and in [0, 100]."""
        mock_resp = _make_mock_response(200, MOCK_PREDICTION_RESPONSE)
        with patch("streamlit_app.httpx.post", return_value=mock_resp):
            result = streamlit_app.send_prediction(
                "http://127.0.0.1:8000",
                VALID_FORM_VALUES,
            )
        score = result["risk_score"]
        assert 0.0 <= score <= 100.0, f"risk_score={score} outside [0, 100]."

    def test_risk_tier_in_response(self) -> None:
        """Gate 14: risk_tier is a valid tier."""
        mock_resp = _make_mock_response(200, MOCK_PREDICTION_RESPONSE)
        with patch("streamlit_app.httpx.post", return_value=mock_resp):
            result = streamlit_app.send_prediction(
                "http://127.0.0.1:8000",
                VALID_FORM_VALUES,
            )
        assert result["risk_tier"] in {"LOW", "MEDIUM", "HIGH"}, (
            f"Unexpected risk_tier: {result['risk_tier']}"
        )

    def test_decision_in_response(self) -> None:
        """Gate 14: decision is a valid decision."""
        mock_resp = _make_mock_response(200, MOCK_PREDICTION_RESPONSE)
        with patch("streamlit_app.httpx.post", return_value=mock_resp):
            result = streamlit_app.send_prediction(
                "http://127.0.0.1:8000",
                VALID_FORM_VALUES,
            )
        assert result["decision"] in {"APPROVE", "MANUAL REVIEW", "REJECT"}, (
            f"Unexpected decision: {result['decision']}"
        )


# ===========================================================================
# TEST 15 — Disclaimer text exists
# ===========================================================================

class TestGate15DisclaimerExists:
    """Gate 15: DISCLAIMER constant must exist and contain required statements."""

    def test_disclaimer_constant_exists(self) -> None:
        """Gate 15: DISCLAIMER is defined in streamlit_app."""
        assert hasattr(streamlit_app, "DISCLAIMER"), (
            "Gate 15 FAILED: DISCLAIMER constant not found in streamlit_app."
        )

    def test_disclaimer_mentions_uci_dataset(self) -> None:
        """Gate 15: Disclaimer mentions the UCI dataset."""
        assert "UCI" in streamlit_app.DISCLAIMER, (
            "Disclaimer must reference the UCI dataset."
        )

    def test_disclaimer_states_not_indian_banking(self) -> None:
        """Gate 15: Disclaimer explicitly states this is NOT Indian banking."""
        disc_lower = streamlit_app.DISCLAIMER.lower()
        assert "not" in disc_lower and ("indian" in disc_lower or "india" in disc_lower), (
            "Disclaimer must explicitly state this is NOT an Indian banking system."
        )

    def test_disclaimer_mentions_demonstration(self) -> None:
        """Gate 15 (extended): Disclaimer uses the word 'demonstration'."""
        assert "demonstration" in streamlit_app.DISCLAIMER.lower() or \
               "demo" in streamlit_app.DISCLAIMER.lower(), (
            "Disclaimer must indicate this is a demonstration."
        )


# ===========================================================================
# TEST 16 — Indian-specific fields are absent
# ===========================================================================

class TestGate16NoIndianSpecificFields:
    """Gate 16: Indian lending fields must not appear in the UCI payload or code."""

    @pytest.fixture(scope="class")
    def source_code(self, tmp_path_factory) -> str:
        return (PROJECT_ROOT / "streamlit_app.py").read_text(encoding="utf-8")

    @pytest.fixture(scope="class")
    def payload_keys(self, tmp_path_factory) -> set:
        return set(streamlit_app.REQUIRED_UCI_FEATURES)

    def test_no_pan_in_payload(self, payload_keys: set) -> None:
        """Gate 16: 'pan' not in REQUIRED_UCI_FEATURES."""
        assert "pan" not in payload_keys and "pan_verified" not in payload_keys, (
            "Gate 16 FAILED: PAN field found in UCI payload."
        )

    def test_no_aadhaar_in_payload(self, payload_keys: set) -> None:
        """Gate 16: 'aadhaar' not in REQUIRED_UCI_FEATURES."""
        assert not any("aadhaar" in k for k in payload_keys), (
            "Gate 16 FAILED: Aadhaar field found in UCI payload."
        )

    def test_no_cibil_in_payload(self, payload_keys: set) -> None:
        """Gate 16: 'cibil' not in REQUIRED_UCI_FEATURES."""
        assert not any("cibil" in k for k in payload_keys), (
            "Gate 16 FAILED: CIBIL field found in UCI payload."
        )

    def test_no_indian_fields_in_source(self, source_code: str) -> None:
        """Gate 16 (extended): No Indian field names appear as payload fields in source."""
        indian_keywords = ["pan_verified", "aadhaar_kyc", "cibil_score", "rbi_"]
        found = [kw for kw in indian_keywords if kw in source_code.lower()]
        assert not found, (
            f"Gate 16 FAILED: Indian-specific field references found: {found}"
        )
