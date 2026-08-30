"""
api/main.py
============

PURPOSE:
    FastAPI application for the Phase 5 Step 2 REST API layer.

    Exposes the existing Phase 5 Step 1 CreditRiskPredictor through three endpoints:
        GET  /health       — Dynamic health check (model_loaded reflects actual artifact state)
        GET  /model-info   — Model metadata from models/model_metadata.json
        POST /predict      — Single applicant credit risk prediction

ARCHITECTURE:
    HTTP JSON
        -> Pydantic validation (api/schemas.py)
        -> CreditRiskPredictor.predict() (src/prediction/predictor.py)
        -> src/models/model_artifacts.py (loads saved artifacts)
        -> preprocessor.transform() + model.predict_proba()
        -> PredictResponse (api/schemas.py)

LEAKAGE PREVENTION:
    This API NEVER calls:
        - model.fit()
        - preprocessor.fit()
        - preprocessor.fit_transform()
    Only .transform() and .predict_proba() are called during prediction.

PREDICTOR LIFECYCLE:
    CreditRiskPredictor is instantiated ONCE at application startup via
    the FastAPI lifespan context manager and stored in app.state.predictor.
    It is shared across all requests (thread-safe: no mutable state after init).

ERROR HANDLING:
    InputValidationError  -> HTTP 422 (invalid UCI applicant data)
    ArtifactError         -> HTTP 503 (missing or corrupt model artifacts)
    Unexpected exceptions -> HTTP 500 (no internal details exposed)

SECURITY BASICS:
    - No filesystem paths in error responses
    - No model internals exposed in error messages
    - No training, dataset upload, or code execution endpoints
    - All request data validated before reaching the predictor
    - Deterministic prediction (no randomness post-training)

DISCLAIMER:
    This API serves the UCI Statlog German Credit Data benchmark model.
    UCI dataset is historical European banking data (1994, Deutsche Mark).
    This API is NOT an Indian banking underwriting system.
    PD estimates are NOT formally calibrated.
    Risk thresholds are DEMONSTRATION POLICY ASSUMPTIONS ONLY.

RUNNING:
    From the project root directory:
        python -m uvicorn api.main:app --reload

    API available at: http://127.0.0.1:8000
    Swagger UI:       http://127.0.0.1:8000/docs
    ReDoc:            http://127.0.0.1:8000/redoc

PHASE 5 — STEP 2
"""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

# ---------------------------------------------------------------------------
# Path setup — allow running from project root
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Internal imports (Phase 5 Step 1 modules — never modified here)
# ---------------------------------------------------------------------------

from src.models.model_artifacts import ModelArtifacts, ArtifactError  # noqa: E402
from src.prediction.predictor import CreditRiskPredictor              # noqa: E402
from src.prediction.schemas import InputValidationError               # noqa: E402
from src.prediction.risk_policy import VALID_TIERS, VALID_DECISIONS   # noqa: E402
from api.schemas import (                                              # noqa: E402
    PredictRequest,
    PredictResponse,
    HealthResponse,
    ModelInfoResponse,
    ErrorResponse,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan — predictor loaded once at startup, shared across all requests
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.

    Startup: Load CreditRiskPredictor and store in app.state.
             If artifacts are missing, store None and log a warning.
             The API will still start but predictions will fail with 503.

    Shutdown: Clean up (no-op for this predictor).
    """
    logger.info("API startup: initializing CreditRiskPredictor...")
    try:
        app.state.predictor = CreditRiskPredictor()
        logger.info("CreditRiskPredictor loaded successfully.")
    except ArtifactError as exc:
        # Start the API anyway — /health will report unavailable
        app.state.predictor = None
        logger.warning("Model artifacts unavailable at startup: %s", exc)
    except Exception as exc:
        app.state.predictor = None
        logger.error("Unexpected error loading predictor: %s", exc)

    yield

    logger.info("API shutdown complete.")


# ---------------------------------------------------------------------------
# FastAPI application instance
# ---------------------------------------------------------------------------

_DESCRIPTION = """
## Credit Risk & Loan Decision API

**Inference-only REST API** for the UCI Statlog German Credit benchmark model.

### What this API does
Accepts the 20 UCI German Credit features for a single applicant and returns:
- **Estimated Probability of Default (PD)** — model's raw `predict_proba()` output
- **Internal Risk Score** — `(1 - estimated_pd) * 100` — range: 0 to 100
- **Risk Tier** — `LOW`, `MEDIUM`, or `HIGH`
- **Loan Decision** — `APPROVE`, `MANUAL REVIEW`, or `REJECT`

### Important disclaimers

> ⚠️ **This is a demonstration inference API, not a production lending system.**

- The model was trained on the **UCI Statlog German Credit Data** (Prof. Dr. Hans Hofmann, 1994).
- The dataset represents **historical German credit data** collected in Deutsche Mark.
- It is **NOT** an Indian banking underwriting model.
- It does **NOT** use CIBIL scores, RBI regulations, or Indian KYC data.
- Estimated PD is **not formally calibrated**.
- Risk thresholds (LOW < 0.20, MEDIUM < 0.45, HIGH ≥ 0.45) are **demonstration policy assumptions only**.
- Do **not** interpret this API as a real lending approval system.

### Running the API
```bash
python -m uvicorn api.main:app --reload
```

### Rebuilding model artifacts
```bash
python scripts/build_model_artifacts.py
```
"""

app = FastAPI(
    title="Credit Risk & Loan Decision API",
    description=_DESCRIPTION,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------

@app.exception_handler(InputValidationError)
async def input_validation_error_handler(
    request: Request, exc: InputValidationError
) -> JSONResponse:
    """
    Handle UCI input validation errors raised by ApplicantInput.from_dict().
    Returns HTTP 422 with a clean, client-safe error message.
    No internal paths or stack traces exposed.
    """
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "input_validation_error",
            "detail": str(exc),
        },
    )


@app.exception_handler(ArtifactError)
async def artifact_error_handler(
    request: Request, exc: ArtifactError
) -> JSONResponse:
    """
    Handle model artifact errors.
    Returns HTTP 503 — service not ready.
    No internal paths or stack traces exposed.
    """
    logger.error("ArtifactError during request: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "error": "model_unavailable",
            "detail": (
                "The credit risk model is currently unavailable. "
                "Ensure model artifacts are built by running: "
                "python scripts/build_model_artifacts.py"
            ),
        },
    )


@app.exception_handler(Exception)
async def generic_error_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """
    Catch-all handler for unexpected errors.
    Returns HTTP 500. No internal details exposed to clients.
    """
    logger.exception("Unexpected error processing request: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "internal_server_error",
            "detail": "An unexpected error occurred. Please try again.",
        },
    )


# ---------------------------------------------------------------------------
# Helper — resolve predictor from app state
# ---------------------------------------------------------------------------

def _get_predictor(request: Request) -> CreditRiskPredictor:
    """
    Retrieve the predictor from application state.

    Raises:
        ArtifactError: If the predictor was not loaded at startup (artifacts missing).
    """
    predictor = request.app.state.predictor
    if predictor is None:
        raise ArtifactError(
            "Model predictor is not available. "
            "Run: python scripts/build_model_artifacts.py"
        )
    return predictor


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description=(
        "Returns the current health status of the API service. "
        "The `model_loaded` field is determined dynamically by checking "
        "whether all required model artifact files exist on disk."
    ),
    tags=["System"],
)
async def health(request: Request) -> HealthResponse:
    """
    GET /health

    Returns service health status.

    The `model_loaded` value reflects the ACTUAL state of model artifacts.
    It is NOT hard-coded to `true`.

    If artifacts are missing, the response will indicate the service is not ready.
    """
    predictor_loaded = request.app.state.predictor is not None
    artifacts_exist  = ModelArtifacts.all_artifacts_exist()
    model_loaded     = predictor_loaded and artifacts_exist

    return HealthResponse(
        status="healthy" if model_loaded else "unavailable",
        service="credit-risk-api",
        model_loaded=model_loaded,
    )


@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
    summary="Model Information",
    description=(
        "Returns metadata about the currently loaded model. "
        "Values are read dynamically from `models/model_metadata.json`. "
        "No metadata is hard-coded in the application."
    ),
    tags=["Model"],
    responses={
        503: {"model": ErrorResponse, "description": "Model artifacts unavailable."},
    },
)
async def model_info(request: Request) -> ModelInfoResponse:
    """
    GET /model-info

    Returns metadata about the production model loaded at startup.

    Data is read from the predictor's metadata property (which loads
    models/model_metadata.json). No values are hard-coded.
    """
    predictor = _get_predictor(request)
    meta = predictor.metadata

    return ModelInfoResponse(
        model_name=meta.get("model_name", "unknown"),
        model_version=meta.get("model_version", "unknown"),
        dataset=meta.get("dataset", "unknown"),
        feature_count=meta.get("feature_count", 0),
        processed_feature_count=meta.get("processed_feature_count", 0),
        training_records=meta.get("training_records", 0),
        risk_score_formula=meta.get("risk_score_formula", "(1 - estimated_pd) * 100"),
        pd_is_calibrated=meta.get("pd_is_calibrated", False),
        purpose=meta.get("purpose", "historical benchmark demonstration"),
        context_note=meta.get("context_note", ""),
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Credit Risk Prediction",
    description=(
        "Submit a UCI German Credit applicant profile and receive a credit risk assessment. "
        "All 20 UCI feature fields are required. "
        "Invalid inputs are rejected with HTTP 422 before reaching the model. "
        "The model is NEVER retrained during this request."
    ),
    tags=["Prediction"],
    responses={
        422: {"model": ErrorResponse, "description": "Invalid applicant input (missing fields, invalid categoricals, NaN/Inf, out-of-range values)."},
        503: {"model": ErrorResponse, "description": "Model artifacts unavailable."},
    },
)
async def predict(
    request: Request,
    predict_request: PredictRequest,
) -> PredictResponse:
    """
    POST /predict

    Accept a UCI German Credit applicant profile and return a credit risk prediction.

    PIPELINE (inference only — no model retraining):
        1. Pydantic validates types and numeric ranges (HTTP layer)
        2. Request converted to dict via .model_dump()
        3. Dict passed to CreditRiskPredictor.predict()
        4. Inside predictor: ApplicantInput.from_dict() validates UCI categories
        5. preprocessor.transform() applied (NO .fit())
        6. model.predict_proba() called (NO .fit())
        7. Risk policy applied: score, tier, decision
        8. PredictResponse returned

    Returns:
        PredictResponse with estimated_pd, risk_score, risk_tier, decision, model_name, dataset.
    """
    predictor = _get_predictor(request)

    # Convert Pydantic model to plain dict (matches UCI feature name keys)
    applicant_dict = predict_request.model_dump()

    # Delegate entirely to the existing production predictor
    # This is where all remaining validation and inference happens
    result = predictor.predict(applicant_dict)

    return PredictResponse(
        estimated_pd=result.estimated_pd,
        risk_score=result.risk_score,
        risk_tier=result.risk_tier,
        decision=result.decision,
        model_name=result.model_name,
        dataset=result.dataset,
    )
