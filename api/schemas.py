"""
api/schemas.py
===============

PURPOSE:
    Pydantic request and response models for the Phase 5 Step 2 FastAPI layer.

    These models serve two purposes:
        1. HTTP-layer type coercion — FastAPI uses them to parse/validate JSON bodies
           and generate OpenAPI (Swagger/ReDoc) documentation.
        2. Response serialization — structured JSON output from endpoints.

DESIGN — TWO-TIER VALIDATION:
    HTTP JSON
        -> Pydantic (type coercion + OpenAPI docs generation)
        -> dict  (.model_dump())
        -> ApplicantInput.from_dict() (domain validation: UCI whitelists, NaN, ranges)
        -> CreditRiskPredictor.predict()

    This avoids duplicating the UCI categorical whitelist logic.
    Pydantic handles type coercion; domain validation remains in src/prediction/schemas.py.

UCI FEATURES:
    The 20 UCI German Credit features are divided into:
        - 13 categorical (str) — validated against UCI whitelists in src/prediction/schemas.py
        - 7 numerical (float) — range-validated in src/prediction/schemas.py

IMPORTANT — SCHEMA ALIGNMENT:
    Field names here MUST exactly match the UCI column names in:
        src/prediction/schemas.py -> UCI_ALL_FEATURES

    Do NOT introduce Indian lending fields (PAN, Aadhaar, CIBIL, etc.).
    This API is an inference wrapper for the UCI German Credit benchmark only.

DATASET CONTEXT:
    UCI Statlog German Credit Data (1994) — historical European banking benchmark.
    NOT Indian borrower data. NOT CIBIL. NOT RBI-regulated.

PHASE 5 — STEP 2
"""

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request Schema — 20 UCI features
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    """
    Request body for POST /predict.

    Contains all 20 UCI German Credit features required by the inference pipeline.
    Categorical values must match the UCI coding scheme (e.g., 'A11', 'A32').
    Numerical values must be finite and within plausible ranges.

    Note: All field names exactly match the UCI dataset column headers.
    """

    # --- Categorical features (13) ---

    status_existing_checking_account: str = Field(
        ...,
        description=(
            "Status of the existing checking account. "
            "UCI codes: A11 (< 0 DM), A12 (0-200 DM), A13 (>= 200 DM), A14 (no account)."
        ),
        examples=["A11"],
    )

    credit_history: str = Field(
        ...,
        description=(
            "Credit history. "
            "UCI codes: A30 (no credits taken), A31 (all credits paid duly), "
            "A32 (existing credits paid duly), A33 (delay in past), A34 (critical account)."
        ),
        examples=["A32"],
    )

    purpose: str = Field(
        ...,
        description=(
            "Purpose of the credit. "
            "UCI codes: A40 (car new), A41 (car used), A42 (furniture/equipment), "
            "A43 (radio/TV), A44 (domestic appliances), A45 (repairs), A46 (education), "
            "A47 (vacation), A48 (retraining), A49 (business), A410 (others)."
        ),
        examples=["A43"],
    )

    savings_account_bonds: str = Field(
        ...,
        description=(
            "Savings account / bonds balance. "
            "UCI codes: A61 (< 100 DM), A62 (100-499 DM), A63 (500-999 DM), "
            "A64 (>= 1000 DM), A65 (unknown / no savings)."
        ),
        examples=["A62"],
    )

    present_employment_since: str = Field(
        ...,
        description=(
            "Present employment duration. "
            "UCI codes: A71 (unemployed), A72 (< 1 year), A73 (1-3 years), "
            "A74 (4-6 years), A75 (>= 7 years)."
        ),
        examples=["A73"],
    )

    personal_status_sex: str = Field(
        ...,
        description=(
            "Personal status and sex. "
            "UCI codes: A91 (male, divorced/separated), A92 (female, divorced/separated/married), "
            "A93 (male, single), A94 (male, married/widowed), A95 (female, single)."
        ),
        examples=["A93"],
    )

    other_debtors_guarantors: str = Field(
        ...,
        description=(
            "Other debtors / guarantors. "
            "UCI codes: A101 (none), A102 (co-applicant), A103 (guarantor)."
        ),
        examples=["A101"],
    )

    property: str = Field(
        ...,
        description=(
            "Property type. "
            "UCI codes: A121 (real estate), A122 (building society savings/life insurance), "
            "A123 (car or other), A124 (unknown / no property)."
        ),
        examples=["A121"],
    )

    other_installment_plans: str = Field(
        ...,
        description=(
            "Other installment plans. "
            "UCI codes: A141 (bank), A142 (stores), A143 (none)."
        ),
        examples=["A143"],
    )

    housing: str = Field(
        ...,
        description=(
            "Housing type. "
            "UCI codes: A151 (rent), A152 (own), A153 (for free)."
        ),
        examples=["A152"],
    )

    job: str = Field(
        ...,
        description=(
            "Job type. "
            "UCI codes: A171 (unemployed / unskilled non-resident), "
            "A172 (unskilled resident), A173 (skilled employee / official), "
            "A174 (management / self-employed / highly qualified)."
        ),
        examples=["A173"],
    )

    telephone: str = Field(
        ...,
        description=(
            "Telephone registered under applicant name. "
            "UCI codes: A191 (none), A192 (yes, registered)."
        ),
        examples=["A192"],
    )

    foreign_worker: str = Field(
        ...,
        description=(
            "Foreign worker status. "
            "UCI codes: A201 (yes), A202 (no)."
        ),
        examples=["A201"],
    )

    # --- Numerical features (7) ---

    duration_in_months: float = Field(
        ...,
        ge=1,
        le=120,
        description="Credit duration in months. Valid range: 1–120.",
        examples=[24],
    )

    credit_amount: float = Field(
        ...,
        ge=100,
        le=200_000,
        description="Credit amount in Deutsche Mark (historical dataset). Valid range: 100–200,000.",
        examples=[4500],
    )

    installment_rate_pct_disposable_income: float = Field(
        ...,
        ge=1,
        le=4,
        description="Installment rate as percentage of disposable income. Valid range: 1–4.",
        examples=[3],
    )

    present_residence_since: float = Field(
        ...,
        ge=1,
        le=4,
        description="Number of years at present residence. Valid range: 1–4.",
        examples=[2],
    )

    age_years: float = Field(
        ...,
        ge=18,
        le=100,
        description="Applicant age in years. Valid range: 18–100.",
        examples=[34],
    )

    existing_credits_count: float = Field(
        ...,
        ge=1,
        le=4,
        description="Number of existing credits at this bank. Valid range: 1–4.",
        examples=[1],
    )

    people_liable_maintenance: float = Field(
        ...,
        ge=1,
        le=2,
        description="Number of people being liable to provide maintenance. Valid range: 1–2.",
        examples=[1],
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "status_existing_checking_account": "A12",
                "duration_in_months": 24,
                "credit_history": "A32",
                "purpose": "A43",
                "credit_amount": 4500,
                "savings_account_bonds": "A62",
                "present_employment_since": "A73",
                "installment_rate_pct_disposable_income": 3,
                "personal_status_sex": "A93",
                "other_debtors_guarantors": "A101",
                "present_residence_since": 2,
                "property": "A121",
                "age_years": 34,
                "other_installment_plans": "A143",
                "housing": "A152",
                "existing_credits_count": 1,
                "job": "A173",
                "people_liable_maintenance": 1,
                "telephone": "A192",
                "foreign_worker": "A201",
            }
        }
    }


# ---------------------------------------------------------------------------
# Response Schemas
# ---------------------------------------------------------------------------

class PredictResponse(BaseModel):
    """
    Response body for POST /predict.

    Contains the structured prediction result from CreditRiskPredictor.

    IMPORTANT:
        - estimated_pd is the model's raw predict_proba() output for the default class.
        - It is NOT formally calibrated.
        - risk_tier and decision are demonstration policy assumptions only.
        - This is NOT an Indian banking credit decision.
    """

    estimated_pd: float = Field(
        ...,
        description=(
            "Estimated Probability of Default in [0, 1]. "
            "Derived from model.predict_proba(). NOT formally calibrated."
        ),
        examples=[0.1760],
    )
    risk_score: float = Field(
        ...,
        description=(
            "Internal demonstration risk score in [0, 100]. "
            "Formula: (1 - estimated_pd) * 100. Higher score = lower estimated risk."
        ),
        examples=[82.40],
    )
    risk_tier: str = Field(
        ...,
        description=(
            "Risk tier based on demonstration policy thresholds. "
            "One of: 'LOW' (PD < 0.20), 'MEDIUM' (0.20 <= PD < 0.45), 'HIGH' (PD >= 0.45)."
        ),
        examples=["LOW"],
    )
    decision: str = Field(
        ...,
        description=(
            "Demonstration loan decision. "
            "One of: 'APPROVE' (LOW), 'MANUAL REVIEW' (MEDIUM), 'REJECT' (HIGH). "
            "Based on demonstration policy assumptions only."
        ),
        examples=["APPROVE"],
    )
    model_name: str = Field(
        ...,
        description="Name of the model that produced this prediction.",
        examples=["Logistic Regression"],
    )
    dataset: str = Field(
        ...,
        description="Dataset context for this prediction.",
        examples=["UCI Statlog German Credit Data"],
    )


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: str = Field(
        ...,
        description="Service status. 'healthy' if ready, 'unavailable' if artifacts missing.",
        examples=["healthy"],
    )
    service: str = Field(
        ...,
        description="Service name.",
        examples=["credit-risk-api"],
    )
    model_loaded: bool = Field(
        ...,
        description=(
            "Whether the model artifacts are present and loaded. "
            "Determined dynamically — not hard-coded."
        ),
        examples=[True],
    )


class ModelInfoResponse(BaseModel):
    """Response body for GET /model-info."""

    model_name: str = Field(..., description="Name of the production model.", examples=["Logistic Regression"])
    model_version: str = Field(..., description="Model version string.", examples=["1.0.0"])
    dataset: str = Field(..., description="Training dataset name.", examples=["UCI Statlog German Credit Data"])
    feature_count: int = Field(..., description="Number of raw input features.", examples=[20])
    processed_feature_count: int = Field(..., description="Number of features after preprocessing.", examples=[61])
    training_records: int = Field(..., description="Number of records used for training.", examples=[800])
    risk_score_formula: str = Field(..., description="Formula for internal risk score.", examples=["(1 - estimated_pd) * 100"])
    pd_is_calibrated: bool = Field(..., description="Whether the PD estimate is formally calibrated.", examples=[False])
    purpose: str = Field(..., description="Intended purpose of this model.", examples=["historical benchmark demonstration"])
    context_note: str = Field(..., description="Dataset context and disclaimer.")


class ErrorResponse(BaseModel):
    """Standard error response body."""

    error: str = Field(..., description="Error type identifier.")
    detail: str = Field(..., description="Human-readable error description.")
