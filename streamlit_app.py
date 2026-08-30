"""
streamlit_app.py
=================

PURPOSE:
    Phase 5 — Step 3: Streamlit presentation layer for the Credit Risk &
    Loan Decision Intelligence Platform.

    This application is a THIN CLIENT over the Phase 5 Step 2 FastAPI REST API.
    It does NOT:
        - Load CSV datasets
        - Load model artifacts
        - Train or retrain any model
        - Invoke preprocessor training or model training operations
        - Duplicate prediction or risk-policy logic

    All inference is performed by the existing FastAPI service:
        POST http://127.0.0.1:8000/predict

ARCHITECTURE:
    Streamlit UI
        ↓ HTTP (httpx)
    FastAPI REST API  (api/main.py)
        ↓
    CreditRiskPredictor  (src/prediction/predictor.py)
        ↓
    preprocessing.joblib  +  credit_risk_model.joblib
        ↓
    Risk Policy
        ↓
    JSON Response
        ↓
    Streamlit Result UI

RUNNING:
    Terminal 1 (API):
        python -m uvicorn api.main:app --reload

    Terminal 2 (Dashboard):
        streamlit run streamlit_app.py

DATASET CONTEXT:
    UCI Statlog German Credit Data (1994) — historical European banking benchmark.
    NOT Indian borrower data. NOT CIBIL. NOT RBI-regulated.

PHASE 5 — STEP 3
"""

import sys
from typing import Any, Dict, Optional, Tuple

import httpx
import streamlit as st


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

API_DEFAULT_URL: str = "http://127.0.0.1:8000"

REQUEST_TIMEOUT: float = 10.0
HEALTH_TIMEOUT: float  = 5.0

DISCLAIMER: str = (
    "Demonstration only: This assessment uses the historical UCI Statlog German Credit "
    "benchmark dataset (1994, Prof. Dr. Hans Hofmann). It is NOT an Indian banking "
    "approval decision, does NOT represent Indian borrowers, and should NOT be used "
    "for real lending decisions. Risk policy thresholds are demonstration assumptions only."
)

# The 20 UCI feature names in exact training order
REQUIRED_UCI_FEATURES = [
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
]

# Human-readable labels for categorical codes (code → label)
UCI_CATEGORICAL_OPTIONS: Dict[str, Dict[str, str]] = {
    "status_existing_checking_account": {
        "A11": "< 0 DM  (account overdrawn)",
        "A12": "0 to 200 DM",
        "A13": ">= 200 DM",
        "A14": "No checking account",
    },
    "credit_history": {
        "A30": "No credits taken / all paid back duly",
        "A31": "All credits at this bank paid duly",
        "A32": "Existing credits paid duly so far",
        "A33": "Delay in paying off in the past",
        "A34": "Critical account / other credits existing",
    },
    "purpose": {
        "A40":  "Car (new)",
        "A41":  "Car (used)",
        "A42":  "Furniture / Equipment",
        "A43":  "Radio / TV",
        "A44":  "Domestic Appliances",
        "A45":  "Repairs",
        "A46":  "Education",
        "A47":  "Vacation",
        "A48":  "Retraining",
        "A49":  "Business",
        "A410": "Other",
    },
    "savings_account_bonds": {
        "A61": "< 100 DM",
        "A62": "100 to 499 DM",
        "A63": "500 to 999 DM",
        "A64": ">= 1000 DM",
        "A65": "Unknown / No savings account",
    },
    "present_employment_since": {
        "A71": "Unemployed",
        "A72": "< 1 year",
        "A73": "1 to 3 years",
        "A74": "4 to 6 years",
        "A75": ">= 7 years",
    },
    "personal_status_sex": {
        "A91": "Male — Divorced / Separated",
        "A92": "Female — Divorced / Separated / Married",
        "A93": "Male — Single",
        "A94": "Male — Married / Widowed",
        "A95": "Female — Single",
    },
    "other_debtors_guarantors": {
        "A101": "None",
        "A102": "Co-applicant",
        "A103": "Guarantor",
    },
    "property": {
        "A121": "Real estate",
        "A122": "Building society savings / life insurance",
        "A123": "Car or other",
        "A124": "Unknown / No property",
    },
    "other_installment_plans": {
        "A141": "Bank",
        "A142": "Stores",
        "A143": "None",
    },
    "housing": {
        "A151": "Rent",
        "A152": "Own",
        "A153": "For free",
    },
    "job": {
        "A171": "Unemployed / Unskilled — Non-resident",
        "A172": "Unskilled — Resident",
        "A173": "Skilled Employee / Official",
        "A174": "Management / Self-employed / Highly Qualified",
    },
    "telephone": {
        "A191": "None",
        "A192": "Yes — Registered under applicant's name",
    },
    "foreign_worker": {
        "A201": "Yes",
        "A202": "No",
    },
}

# Numerical feature configuration: feature → {label, min, max, step, default, help}
UCI_NUMERICAL_CONFIG: Dict[str, Dict[str, Any]] = {
    "duration_in_months": {
        "label":   "Credit Duration (months)",
        "min":     1,
        "max":     120,
        "step":    1,
        "default": 24,
        "help":    "Duration of the credit in months.",
    },
    "credit_amount": {
        "label":   "Credit Amount (Deutsche Mark)",
        "min":     100,
        "max":     200_000,
        "step":    100,
        "default": 4_500,
        "help":    "Historical dataset currency: Deutsche Mark (DM).",
    },
    "installment_rate_pct_disposable_income": {
        "label":   "Installment Rate (% of disposable income)",
        "min":     1,
        "max":     4,
        "step":    1,
        "default": 3,
        "help":    "Installment rate as percentage of disposable income.",
    },
    "present_residence_since": {
        "label":   "Years at Present Residence",
        "min":     1,
        "max":     4,
        "step":    1,
        "default": 2,
        "help":    "Number of years at current address.",
    },
    "age_years": {
        "label":   "Applicant Age (years)",
        "min":     18,
        "max":     100,
        "step":    1,
        "default": 34,
        "help":    "Applicant age in years.",
    },
    "existing_credits_count": {
        "label":   "Existing Credits at This Bank",
        "min":     1,
        "max":     4,
        "step":    1,
        "default": 1,
        "help":    "Number of existing credits at this bank.",
    },
    "people_liable_maintenance": {
        "label":   "Dependants (liable for maintenance)",
        "min":     1,
        "max":     2,
        "step":    1,
        "default": 1,
        "help":    "Number of people the applicant is liable to provide maintenance for.",
    },
}

# Risk tier → display color (CSS)
TIER_COLORS: Dict[str, str] = {
    "LOW":    "#0d7a0d",   # dark green
    "MEDIUM": "#b07800",   # amber
    "HIGH":   "#c0392b",   # red
}

# Decision → display color (CSS)
DECISION_COLORS: Dict[str, str] = {
    "APPROVE":       "#0d7a0d",
    "MANUAL REVIEW": "#b07800",
    "REJECT":        "#c0392b",
}


# ---------------------------------------------------------------------------
# Pure helper functions (no st.* calls — fully testable without Streamlit)
# ---------------------------------------------------------------------------

def check_api_health(api_url: str, timeout: float = HEALTH_TIMEOUT) -> Dict[str, Any]:
    """
    Call GET /health and return the parsed JSON response.

    Args:
        api_url: Base API URL (e.g., "http://127.0.0.1:8000").
        timeout: Request timeout in seconds.

    Returns:
        dict with at minimum {"status": str, "model_loaded": bool}

    Raises:
        httpx.ConnectError: If the API is not running.
        httpx.TimeoutException: If the request times out.
        RuntimeError: If the API returns a non-200 status.
    """
    url = f"{api_url.rstrip('/')}/health"
    response = httpx.get(url, timeout=timeout)
    if response.status_code != 200:
        raise RuntimeError(
            f"Health check returned HTTP {response.status_code}."
        )
    return response.json()


def get_model_info(api_url: str, timeout: float = REQUEST_TIMEOUT) -> Dict[str, Any]:
    """
    Call GET /model-info and return the parsed JSON metadata.

    Args:
        api_url: Base API URL.
        timeout: Request timeout in seconds.

    Returns:
        dict with model metadata fields.

    Raises:
        httpx.ConnectError: If the API is not running.
        RuntimeError: If the API returns a non-200 status.
    """
    url = f"{api_url.rstrip('/')}/model-info"
    response = httpx.get(url, timeout=timeout)
    if response.status_code != 200:
        raise RuntimeError(
            f"Model info returned HTTP {response.status_code}."
        )
    return response.json()


def send_prediction(
    api_url: str,
    payload: Dict[str, Any],
    timeout: float = REQUEST_TIMEOUT,
) -> Dict[str, Any]:
    """
    Call POST /predict with the applicant payload and return parsed result.

    Args:
        api_url: Base API URL.
        payload: Dict of 20 UCI feature values.
        timeout: Request timeout in seconds.

    Returns:
        dict with estimated_pd, risk_score, risk_tier, decision, model_name, dataset.

    Raises:
        httpx.ConnectError: API not running.
        ValueError: HTTP 422 (invalid input).
        RuntimeError: HTTP 503 or 500.
    """
    url = f"{api_url.rstrip('/')}/predict"
    response = httpx.post(url, json=payload, timeout=timeout)

    if response.status_code == 200:
        return response.json()

    if response.status_code == 422:
        detail = response.json().get("detail", str(response.text))
        raise ValueError(
            f"Please check the applicant information. One or more values are invalid.\n"
            f"Details: {detail}"
        )

    if response.status_code == 503:
        raise RuntimeError(
            "The prediction service is currently unavailable because model "
            "artifacts could not be loaded. Run: python scripts/build_model_artifacts.py"
        )

    raise RuntimeError(
        f"An internal prediction service error occurred (HTTP {response.status_code})."
    )


def build_applicant_payload(form_values: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build the exact 20-feature UCI payload from form values.

    Only the 20 required UCI features are included. Any extra keys in
    form_values are excluded. This ensures the API payload is always clean.

    Args:
        form_values: Dict containing at least all 20 REQUIRED_UCI_FEATURES keys.

    Returns:
        Dict with exactly 20 UCI feature key-value pairs.

    Raises:
        KeyError: If a required feature is missing from form_values.
    """
    return {feature: form_values[feature] for feature in REQUIRED_UCI_FEATURES}


def format_tier_badge(risk_tier: str) -> str:
    """
    Return an HTML badge string for the given risk tier.

    Args:
        risk_tier: One of "LOW", "MEDIUM", "HIGH".

    Returns:
        HTML string.
    """
    color = TIER_COLORS.get(risk_tier, "#555555")
    return (
        f'<span style="background:{color};color:#fff;padding:4px 16px;'
        f'border-radius:20px;font-weight:600;font-size:1rem;">{risk_tier}</span>'
    )


def format_decision_badge(decision: str) -> str:
    """
    Return an HTML badge string for the given loan decision.

    Args:
        decision: One of "APPROVE", "MANUAL REVIEW", "REJECT".

    Returns:
        HTML string.
    """
    color = DECISION_COLORS.get(decision, "#555555")
    return (
        f'<span style="background:{color};color:#fff;padding:4px 16px;'
        f'border-radius:20px;font-weight:600;font-size:1rem;">{decision}</span>'
    )


def kpi_card_html(title: str, value: str, subtitle: str = "") -> str:
    """
    Return an HTML KPI card for display via st.markdown(..., unsafe_allow_html=True).

    Args:
        title:    Card header label.
        value:    Primary displayed value.
        subtitle: Optional secondary line.

    Returns:
        HTML string for an individual KPI card.
    """
    sub_html = (
        f'<div style="font-size:0.75rem;color:#888;margin-top:2px;">{subtitle}</div>'
        if subtitle else ""
    )
    return f"""
    <div style="
        background: #1e2130;
        border: 1px solid #2e3250;
        border-radius: 10px;
        padding: 1.2rem 1rem;
        text-align: center;
        min-height: 100px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    ">
        <div style="font-size:0.75rem;color:#9ba3c5;text-transform:uppercase;
                    letter-spacing:0.08em;margin-bottom:0.4rem;">{title}</div>
        <div style="font-size:1.6rem;font-weight:700;color:#e8eaf0;">{value}</div>
        {sub_html}
    </div>
    """


# ---------------------------------------------------------------------------
# Streamlit UI — sidebar
# ---------------------------------------------------------------------------

def _render_sidebar() -> str:
    """Render sidebar and return the current API URL."""
    with st.sidebar:
        st.markdown("## Credit Risk Platform")
        st.markdown("---")

        api_url = st.text_input(
            "API Base URL",
            value=st.session_state.get("api_url", API_DEFAULT_URL),
            key="api_url_input",
            help="FastAPI REST API base URL (Phase 5 Step 2).",
        )
        st.session_state["api_url"] = api_url

        st.markdown("---")

        if st.button("Check API Health", key="btn_health_check"):
            with st.spinner("Checking API..."):
                try:
                    health = check_api_health(api_url)
                    if health.get("model_loaded"):
                        st.success("API Status: Connected")
                        st.success("Model Status: Loaded")
                    else:
                        st.warning("API reachable but model not loaded.")
                        st.caption("Run: python scripts/build_model_artifacts.py")
                    st.session_state["api_healthy"] = health.get("model_loaded", False)
                except (httpx.ConnectError, httpx.ConnectTimeout):
                    st.error("API Status: Offline")
                    st.caption(
                        f"FastAPI is not running. Start the API:\n"
                        f"`python -m uvicorn api.main:app --reload`"
                    )
                    st.session_state["api_healthy"] = False
                except Exception:
                    st.error("Health check failed.")
                    st.session_state["api_healthy"] = False

        current_health = st.session_state.get("api_healthy")
        if current_health is True:
            st.markdown(
                '<span style="color:#2ecc71;">&#9679; API Ready</span>',
                unsafe_allow_html=True,
            )
        elif current_health is False:
            st.markdown(
                '<span style="color:#e74c3c;">&#9679; API Offline</span>',
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # Model info section
        if st.button("Load Model Info", key="btn_model_info"):
            with st.spinner("Loading model information..."):
                try:
                    meta = get_model_info(api_url)
                    st.markdown("#### Model Information")
                    st.markdown(f"**Model:** {meta.get('model_name', 'N/A')}")
                    st.markdown(f"**Dataset:** {meta.get('dataset', 'N/A')}")
                    st.markdown(f"**Features:** {meta.get('feature_count', 'N/A')} raw → {meta.get('processed_feature_count', 'N/A')} processed")
                    st.markdown(f"**Training records:** {meta.get('training_records', 'N/A')}")
                    cal = meta.get("pd_is_calibrated", False)
                    st.markdown(f"**PD Calibrated:** {'Yes' if cal else 'No'}")
                    st.markdown(f"**Score formula:** `{meta.get('risk_score_formula', 'N/A')}`")
                    st.session_state["model_meta"] = meta
                except (httpx.ConnectError, httpx.ConnectTimeout):
                    st.error("Cannot reach API.")
                except Exception:
                    st.error("Failed to load model information.")

        st.markdown("---")
        st.caption(
            "Phase 5 Step 3 — Streamlit Client\n\n"
            "UCI German Credit Dataset (1994)\n\n"
            "Historical European benchmark only.\n"
            "Not Indian banking data."
        )

    return api_url


# ---------------------------------------------------------------------------
# Streamlit UI — applicant input form
# ---------------------------------------------------------------------------

def _render_form() -> Dict[str, Any]:
    """
    Render the 20-feature UCI applicant input form.

    Returns:
        dict with all 20 UCI feature values, ready for build_applicant_payload().
    """
    form_values: Dict[str, Any] = {}

    def _categorical_select(feature: str, label: str, help_text: str = "") -> str:
        """Render a selectbox returning the UCI code."""
        options = list(UCI_CATEGORICAL_OPTIONS[feature].keys())
        return st.selectbox(
            label,
            options=options,
            format_func=lambda code: f"{code} — {UCI_CATEGORICAL_OPTIONS[feature][code]}",
            help=help_text,
            key=f"form_{feature}",
        )

    # ---- A. Financial / Credit Information --------------------------------
    st.markdown("#### A. Financial / Credit Information")
    col1, col2 = st.columns(2)

    with col1:
        form_values["status_existing_checking_account"] = _categorical_select(
            "status_existing_checking_account",
            "Checking Account Status",
            "Status of the existing checking account.",
        )
        form_values["credit_history"] = _categorical_select(
            "credit_history",
            "Credit History",
            "Record of past credit repayment.",
        )
        cfg = UCI_NUMERICAL_CONFIG["credit_amount"]
        form_values["credit_amount"] = float(st.number_input(
            cfg["label"],
            min_value=float(cfg["min"]),
            max_value=float(cfg["max"]),
            value=float(cfg["default"]),
            step=float(cfg["step"]),
            help=cfg["help"],
            key="form_credit_amount",
        ))
    with col2:
        form_values["savings_account_bonds"] = _categorical_select(
            "savings_account_bonds",
            "Savings Account / Bonds",
            "Balance in savings account or bonds.",
        )
        cfg = UCI_NUMERICAL_CONFIG["installment_rate_pct_disposable_income"]
        form_values["installment_rate_pct_disposable_income"] = float(st.number_input(
            cfg["label"],
            min_value=float(cfg["min"]),
            max_value=float(cfg["max"]),
            value=float(cfg["default"]),
            step=float(cfg["step"]),
            help=cfg["help"],
            key="form_installment_rate",
        ))
        cfg = UCI_NUMERICAL_CONFIG["existing_credits_count"]
        form_values["existing_credits_count"] = float(st.number_input(
            cfg["label"],
            min_value=float(cfg["min"]),
            max_value=float(cfg["max"]),
            value=float(cfg["default"]),
            step=float(cfg["step"]),
            help=cfg["help"],
            key="form_existing_credits",
        ))

    st.markdown("")

    # ---- B. Loan Information -----------------------------------------------
    st.markdown("#### B. Loan Information")
    col3, col4 = st.columns(2)

    with col3:
        cfg = UCI_NUMERICAL_CONFIG["duration_in_months"]
        form_values["duration_in_months"] = float(st.number_input(
            cfg["label"],
            min_value=float(cfg["min"]),
            max_value=float(cfg["max"]),
            value=float(cfg["default"]),
            step=float(cfg["step"]),
            help=cfg["help"],
            key="form_duration",
        ))
        form_values["purpose"] = _categorical_select(
            "purpose",
            "Credit Purpose",
            "Purpose for which the credit is requested.",
        )
    with col4:
        form_values["other_installment_plans"] = _categorical_select(
            "other_installment_plans",
            "Other Installment Plans",
            "Whether the applicant has other installment plans.",
        )

    st.markdown("")

    # ---- C. Employment / Residence ----------------------------------------
    st.markdown("#### C. Employment & Residence")
    col5, col6 = st.columns(2)

    with col5:
        form_values["present_employment_since"] = _categorical_select(
            "present_employment_since",
            "Employment Duration",
            "Duration of present employment.",
        )
        cfg = UCI_NUMERICAL_CONFIG["present_residence_since"]
        form_values["present_residence_since"] = float(st.number_input(
            cfg["label"],
            min_value=float(cfg["min"]),
            max_value=float(cfg["max"]),
            value=float(cfg["default"]),
            step=float(cfg["step"]),
            help=cfg["help"],
            key="form_residence_since",
        ))
    with col6:
        form_values["housing"] = _categorical_select(
            "housing",
            "Housing Type",
            "Whether the applicant rents, owns, or lives for free.",
        )
        form_values["job"] = _categorical_select(
            "job",
            "Job Type",
            "Employment category.",
        )

    st.markdown("")

    # ---- D. Personal Information ------------------------------------------
    st.markdown("#### D. Personal Information")
    col7, col8 = st.columns(2)

    with col7:
        cfg = UCI_NUMERICAL_CONFIG["age_years"]
        form_values["age_years"] = float(st.number_input(
            cfg["label"],
            min_value=float(cfg["min"]),
            max_value=float(cfg["max"]),
            value=float(cfg["default"]),
            step=float(cfg["step"]),
            help=cfg["help"],
            key="form_age",
        ))
        form_values["personal_status_sex"] = _categorical_select(
            "personal_status_sex",
            "Personal Status & Sex",
            "Marital status and sex category.",
        )
        cfg = UCI_NUMERICAL_CONFIG["people_liable_maintenance"]
        form_values["people_liable_maintenance"] = float(st.number_input(
            cfg["label"],
            min_value=float(cfg["min"]),
            max_value=float(cfg["max"]),
            value=float(cfg["default"]),
            step=float(cfg["step"]),
            help=cfg["help"],
            key="form_people_liable",
        ))
    with col8:
        form_values["telephone"] = _categorical_select(
            "telephone",
            "Telephone",
            "Whether the applicant has a registered telephone.",
        )
        form_values["foreign_worker"] = _categorical_select(
            "foreign_worker",
            "Foreign Worker",
            "Whether the applicant is a foreign worker.",
        )
        form_values["other_debtors_guarantors"] = _categorical_select(
            "other_debtors_guarantors",
            "Other Debtors / Guarantors",
            "Whether there are co-applicants or guarantors.",
        )
        form_values["property"] = _categorical_select(
            "property",
            "Property Type",
            "Type of property owned by the applicant.",
        )

    return form_values


# ---------------------------------------------------------------------------
# Streamlit UI — prediction result display
# ---------------------------------------------------------------------------

def _render_prediction_result(result: Dict[str, Any]) -> None:
    """
    Render a professional KPI result panel from a prediction response dict.

    Args:
        result: Dict with estimated_pd, risk_score, risk_tier, decision, etc.
    """
    st.markdown("---")
    st.markdown("### Credit Risk Assessment Result")

    pd_val      = result.get("estimated_pd", 0.0)
    score_val   = result.get("risk_score",   0.0)
    tier        = result.get("risk_tier",    "N/A")
    decision    = result.get("decision",     "N/A")
    model_name  = result.get("model_name",   "Logistic Regression")
    dataset     = result.get("dataset",      "UCI Statlog German Credit Data")

    # KPI cards
    cols = st.columns(4)
    with cols[0]:
        st.markdown(
            kpi_card_html(
                "Estimated PD",
                f"{pd_val * 100:.2f}%",
                "Probability of Default",
            ),
            unsafe_allow_html=True,
        )
    with cols[1]:
        st.markdown(
            kpi_card_html(
                "Internal Risk Score",
                f"{score_val:.1f} / 100",
                "Higher = Lower Risk",
            ),
            unsafe_allow_html=True,
        )
    with cols[2]:
        tier_color = TIER_COLORS.get(tier, "#555")
        st.markdown(
            kpi_card_html("Risk Tier", f'<span style="color:{tier_color};">{tier}</span>'),
            unsafe_allow_html=True,
        )
    with cols[3]:
        dec_color = DECISION_COLORS.get(decision, "#555")
        st.markdown(
            kpi_card_html(
                "Loan Decision",
                f'<span style="color:{dec_color};">{decision}</span>',
            ),
            unsafe_allow_html=True,
        )

    st.markdown("")

    # Explanation panel
    with st.expander("Risk Assessment Explanation", expanded=True):
        st.markdown(f"""
**Estimated Probability of Default (PD): {pd_val * 100:.2f}%**

The model estimates a **{pd_val * 100:.2f}%** probability that this applicant will default on the credit.
This is the raw output of `model.predict_proba()` applied to the preprocessed feature vector.
It is **not formally calibrated** and should not be treated as a true frequency estimate.

**Internal Risk Score: {score_val:.2f} / 100**

Calculated as `(1 − PD) × 100`. A higher score indicates lower estimated risk.

**Risk Tier: {tier}** | **Decision: {decision}**

Risk tiers and decisions are based on **demonstration policy thresholds only**:
- **LOW** (PD < 20%) → APPROVE
- **MEDIUM** (20% ≤ PD < 45%) → MANUAL REVIEW
- **HIGH** (PD ≥ 45%) → REJECT

These thresholds are NOT RBI guidelines, industry standards, or calibrated lending policy.

⚠️ **Model attribution does not imply causality.** No individual feature caused this applicant's risk tier. The model identifies statistical associations in historical data only.

*Model: {model_name} | Dataset: {dataset}*
        """.strip())

    # Disclaimer
    st.info(f"ℹ️ {DISCLAIMER}")


# ---------------------------------------------------------------------------
# Streamlit UI — main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Main Streamlit application entry point.

    Renders the full dashboard: sidebar → header → form → prediction → result.
    All API calls go through check_api_health(), get_model_info(), send_prediction().
    No model training, dataset loading, or notebook execution occurs here.
    """
    st.set_page_config(
        page_title="Credit Risk Platform",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS — minimal professional fintech style
    st.markdown("""
    <style>
    /* Global */
    html, body, [class*="css"] {
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }
    /* Main header */
    h1 { color: #e8eaf0; }
    h2, h3, h4 { color: #c5c9e0; }
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #131622;
        border-right: 1px solid #1e2130;
    }
    /* Streamlit button accent */
    div.stButton > button {
        background: #3b4cc0;
        color: #fff;
        border: none;
        border-radius: 6px;
        font-weight: 600;
        transition: background 0.2s;
    }
    div.stButton > button:hover {
        background: #4d5fd4;
    }
    /* Info box */
    div[data-testid="stInfo"] {
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Render sidebar and get active API URL
    api_url = _render_sidebar()

    # ---- Page header -------------------------------------------------------
    st.markdown("# 📊 Credit Risk & Loan Decision Intelligence Platform")
    st.markdown(
        "Machine-learning-based credit risk assessment using a historical UCI benchmark model"
    )
    st.markdown(
        f'<p style="font-size:0.82rem;color:#e74c3c;font-weight:500;">'
        f'⚠️ {DISCLAIMER}</p>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ---- Applicant input form ----------------------------------------------
    st.markdown("## Applicant Profile")
    st.caption(
        "Enter the applicant's 20 UCI German Credit features below. "
        "Use the sidebar to verify API connectivity before submitting."
    )
    st.markdown("")

    form_values = _render_form()

    st.markdown("")
    st.markdown("---")

    # ---- Prediction button -------------------------------------------------
    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        assess_clicked = st.button(
            "Assess Credit Risk",
            key="btn_assess",
            use_container_width=True,
        )
    with col_info:
        st.caption(
            "Clicking 'Assess Credit Risk' sends the applicant profile to the "
            "FastAPI inference service. No data is stored. "
            "The model is NOT retrained."
        )

    # ---- Handle prediction -------------------------------------------------
    if assess_clicked:
        # Build payload from form values
        try:
            payload = build_applicant_payload(form_values)
        except KeyError as exc:
            st.error(f"Internal form error — missing feature: {exc}")
            return

        # Call the API
        with st.spinner("Sending request to FastAPI inference service..."):
            try:
                result = send_prediction(api_url, payload)
                st.session_state["last_result"] = result
            except httpx.ConnectError:
                st.error(
                    f"FastAPI is not running. Start the API at `{api_url}` and try again.\n\n"
                    f"Run: `python -m uvicorn api.main:app --reload`"
                )
                return
            except httpx.TimeoutException:
                st.error("Request timed out. Please check that the API is responding.")
                return
            except ValueError as exc:
                st.error(str(exc))
                return
            except RuntimeError as exc:
                st.error(str(exc))
                return
            except Exception:
                st.error(
                    "An unexpected error occurred while contacting the prediction service."
                )
                return

        _render_prediction_result(result)

    # Show last result if present (persists across Streamlit reruns from other interactions)
    elif "last_result" in st.session_state and not assess_clicked:
        st.markdown("")
        st.caption("Showing last prediction result. Click 'Assess Credit Risk' to update.")
        _render_prediction_result(st.session_state["last_result"])


# ---------------------------------------------------------------------------
# Entry point — called by Streamlit runner
# ---------------------------------------------------------------------------

main()
