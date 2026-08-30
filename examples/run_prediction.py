"""
examples/run_prediction.py
===========================

PURPOSE:
    Demonstrates the Phase 5 Step 1 production inference pipeline using
    one clearly synthetic applicant.

    This example shows how CreditRiskPredictor can be used without
    opening any Jupyter notebook.

IMPORTANT:
    The applicant data below is ENTIRELY SYNTHETIC and FICTIONAL.
    It is provided for architectural demonstration only.
    It does NOT represent a real person, real bank, or real loan application.

    The UCI Statlog German Credit Data model is a historical European
    credit-risk benchmark (1994). Its predictions are NOT:
        - Indian banking decisions
        - CIBIL-based scoring
        - RBI-regulated lending assessments
        - Calibrated real-world credit scores

USAGE:
    Run from the project root directory:
        python examples/run_prediction.py

    If artifacts have not been generated yet, run first:
        python scripts/build_model_artifacts.py

PHASE 5 — STEP 1
"""

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — allow running from project root
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.model_artifacts import ArtifactError  # noqa: E402
from src.prediction.predictor import CreditRiskPredictor  # noqa: E402
from src.prediction.schemas import InputValidationError  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic demonstration applicant
# ---------------------------------------------------------------------------
# All values below are purely fictional for demonstration purposes.
# Category codes follow the UCI German Credit Data coding scheme.

DEMO_APPLICANT = {
    # Checking account status
    "status_existing_checking_account":   "A12",   # 0 <= balance < 200 DM

    # Loan duration
    "duration_in_months":                 24,

    # Credit history
    "credit_history":                     "A32",   # existing credits paid back duly

    # Loan purpose
    "purpose":                            "A43",   # furniture/equipment

    # Loan credit amount (Deutsche Mark — historical dataset)
    "credit_amount":                      4500,

    # Savings / bonds
    "savings_account_bonds":              "A62",   # 100 <= savings < 500 DM

    # Employment duration
    "present_employment_since":           "A73",   # 1 <= employment < 4 years

    # Installment rate (% of disposable income)
    "installment_rate_pct_disposable_income": 3,

    # Personal status and sex
    "personal_status_sex":                "A93",   # male, single

    # Other debtors / guarantors
    "other_debtors_guarantors":           "A101",  # none

    # Present residence duration
    "present_residence_since":            2,

    # Property
    "property":                           "A121",  # real estate

    # Age (years)
    "age_years":                          34,

    # Other installment plans
    "other_installment_plans":            "A143",  # none

    # Housing
    "housing":                            "A152",  # own

    # Number of existing credits
    "existing_credits_count":             1,

    # Job type
    "job":                                "A173",  # skilled employee / official

    # Number of people liable for maintenance
    "people_liable_maintenance":          1,

    # Telephone
    "telephone":                          "A192",  # yes, registered

    # Foreign worker
    "foreign_worker":                     "A201",  # yes
}


# ---------------------------------------------------------------------------
# Separator helper
# ---------------------------------------------------------------------------

def _sep(char: str = "=", width: int = 60) -> None:
    print(char * width)


# ---------------------------------------------------------------------------
# Main demonstration
# ---------------------------------------------------------------------------

def run_demonstration() -> None:
    """
    Load the predictor and run one demonstration prediction.
    """
    _sep()
    print("CREDIT RISK PREDICTION DEMONSTRATION")
    _sep()
    print()
    print("Model: UCI Statlog German Credit Benchmark")
    print("Applicant: Synthetic/fictional demo only")
    print()

    # Initialize predictor (loads saved artifacts)
    try:
        predictor = CreditRiskPredictor()
    except ArtifactError as exc:
        print(f"[ERROR] Could not load model artifacts:\n{exc}")
        print()
        print("Run this first:")
        print("  python scripts/build_model_artifacts.py")
        sys.exit(1)

    # Run prediction
    try:
        result = predictor.predict(DEMO_APPLICANT)
    except InputValidationError as exc:
        print(f"[ERROR] Input validation failed:\n{exc}")
        sys.exit(1)

    # Print result
    print("Synthetic Applicant Profile:")
    print(f"  Duration:     {DEMO_APPLICANT['duration_in_months']} months")
    print(f"  Credit Amount: {DEMO_APPLICANT['credit_amount']:,} DM (historical)")
    print(f"  Age:          {DEMO_APPLICANT['age_years']} years")
    print(f"  Purpose:      Furniture/Equipment (A43)")
    print()
    _sep("-")

    print()
    print("Estimated PD:")
    print(f"  {result.estimated_pd:.4f}  ({result.estimated_pd * 100:.2f}%)")
    print()
    print("Internal Risk Score:")
    print(f"  {result.risk_score:.2f} / 100")
    print()
    print("Risk Tier:")
    print(f"  {result.risk_tier}")
    print()
    print("Loan Decision:")
    print(f"  {result.decision}")
    print()
    _sep("-")
    print()
    print("NOTE:")
    print("  This is a demonstration prediction using the historical UCI")
    print("  benchmark model. It is NOT an Indian banking approval decision.")
    print("  Estimated PD is NOT formally calibrated.")
    print()
    _sep()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_demonstration()
