"""
examples/run_api_prediction.py
================================

PURPOSE:
    Demonstrates calling POST /predict on the locally running FastAPI REST API.

    This script sends one synthetic/fictional UCI German Credit applicant to the
    API and prints the structured prediction result.

IMPORTANT:
    The applicant data below is ENTIRELY SYNTHETIC AND FICTIONAL.
    It is provided for architectural demonstration only.
    It does NOT represent a real person, real bank, or real loan application.

    This is a demonstration using the historical UCI German Credit benchmark
    model and is NOT an Indian banking approval decision.

PREREQUISITES:
    1. Model artifacts must be built:
           python scripts/build_model_artifacts.py

    2. The API must be running locally:
           python -m uvicorn api.main:app --reload

    Then run:
           python examples/run_api_prediction.py

PHASE 5 — STEP 2
"""

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import httpx
except ImportError:
    print("[ERROR] httpx is not installed.")
    print("Run: pip install httpx")
    sys.exit(1)


# ---------------------------------------------------------------------------
# API configuration
# ---------------------------------------------------------------------------

API_BASE_URL = "http://127.0.0.1:8000"
PREDICT_URL  = f"{API_BASE_URL}/predict"
HEALTH_URL   = f"{API_BASE_URL}/health"

# ---------------------------------------------------------------------------
# Synthetic demonstration applicant
# ---------------------------------------------------------------------------
# All values below are purely fictional for demonstration purposes.
# Category codes follow the UCI German Credit Data coding scheme.

DEMO_APPLICANT = {
    "status_existing_checking_account":       "A12",   # 0 <= balance < 200 DM
    "duration_in_months":                     24,
    "credit_history":                         "A32",   # existing credits paid back duly
    "purpose":                                "A43",   # furniture/equipment
    "credit_amount":                          4500.0,  # Deutsche Mark (historical dataset)
    "savings_account_bonds":                  "A62",   # 100 <= savings < 500 DM
    "present_employment_since":               "A73",   # 1 <= employment < 4 years
    "installment_rate_pct_disposable_income": 3.0,
    "personal_status_sex":                    "A93",   # male, single
    "other_debtors_guarantors":               "A101",  # none
    "present_residence_since":                2.0,
    "property":                               "A121",  # real estate
    "age_years":                              34.0,
    "other_installment_plans":                "A143",  # none
    "housing":                                "A152",  # own
    "existing_credits_count":                 1.0,
    "job":                                    "A173",  # skilled employee / official
    "people_liable_maintenance":              1.0,
    "telephone":                              "A192",  # yes, registered
    "foreign_worker":                         "A201",  # yes
}


# ---------------------------------------------------------------------------
# Separator helper
# ---------------------------------------------------------------------------

def _sep(char: str = "=", width: int = 60) -> None:
    print(char * width)


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def run_demo() -> None:
    """
    Check API health, then submit the demo applicant and print results.
    """
    _sep()
    print("CREDIT RISK API PREDICTION EXAMPLE")
    _sep()
    print()
    print(f"API URL: {API_BASE_URL}")
    print()

    # ------------------------------------------------------------------
    # Step 1: Health check
    # ------------------------------------------------------------------
    print("Checking API health...")
    try:
        health_resp = httpx.get(HEALTH_URL, timeout=10.0)
    except httpx.ConnectError:
        print()
        print("[ERROR] Cannot connect to the API.")
        print("Make sure the API is running with:")
        print("  python -m uvicorn api.main:app --reload")
        print()
        sys.exit(1)

    health_data = health_resp.json()
    print(f"  Status:       {health_data.get('status', 'unknown')}")
    print(f"  Model loaded: {health_data.get('model_loaded', False)}")
    print()

    if not health_data.get("model_loaded"):
        print("[ERROR] Model artifacts are not loaded.")
        print("Run first:")
        print("  python scripts/build_model_artifacts.py")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 2: Submit prediction
    # ------------------------------------------------------------------
    print("Submitting synthetic applicant to POST /predict...")
    print()
    print("Synthetic Applicant Profile:")
    print(f"  Duration:       {DEMO_APPLICANT['duration_in_months']} months")
    print(f"  Credit Amount:  {DEMO_APPLICANT['credit_amount']:,.0f} DM (historical dataset)")
    print(f"  Age:            {DEMO_APPLICANT['age_years']:.0f} years")
    print(f"  Purpose:        Furniture/Equipment (A43)")
    print(f"  Checking Acct:  0 to 200 DM balance (A12)")
    print()
    _sep("-")

    try:
        pred_resp = httpx.post(PREDICT_URL, json=DEMO_APPLICANT, timeout=30.0)
    except httpx.ConnectError:
        print("[ERROR] Cannot connect to the API.")
        sys.exit(1)

    if pred_resp.status_code != 200:
        print(f"[ERROR] POST /predict returned HTTP {pred_resp.status_code}.")
        print(f"Response: {pred_resp.text}")
        sys.exit(1)

    result = pred_resp.json()

    # ------------------------------------------------------------------
    # Step 3: Print results
    # ------------------------------------------------------------------
    print()
    print("Estimated PD:")
    pd_val = result.get("estimated_pd", 0)
    print(f"  {pd_val:.4f}  ({pd_val * 100:.2f}%)")
    print()
    print("Internal Risk Score:")
    print(f"  {result.get('risk_score', 0):.2f} / 100")
    print()
    print("Risk Tier:")
    print(f"  {result.get('risk_tier', 'unknown')}")
    print()
    print("Loan Decision:")
    print(f"  {result.get('decision', 'unknown')}")
    print()
    _sep("-")
    print()
    print("NOTE:")
    print("  This is a demonstration using the historical UCI German Credit")
    print("  benchmark model and is NOT an Indian banking approval decision.")
    print("  Estimated PD is NOT formally calibrated.")
    print("  Risk thresholds are demonstration policy assumptions only.")
    print()
    _sep()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_demo()
