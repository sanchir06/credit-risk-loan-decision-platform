"""
src/features/feature_engineering.py
======================================

PURPOSE:
    Takes a LoanApplicant object (with raw fields already filled in by the
    verification and financial layers) and computes all the ENGINEERED FEATURES
    that the ML model will actually use.

    This is the BRIDGE between raw data and machine learning.

THE PIPELINE POSITION:
    Verification Layer (pan, kyc, mobile, identity checks)
        ↓
    Financial Analysis (income, bank statements, debt)
        ↓
    Credit Profile (bureau data)
        ↓
    Feature Engineering  ← THIS FILE
        ↓
    Feature Vector (a row of numbers ready for the ML model)
        ↓
    Credit Risk ML Model (Logistic Regression / XGBoost)
        ↓
    Probability of Default

FEATURE TAXONOMY (3 types for the CREDIT RISK ML model):
    ┌─────────────────────┬──────────────────────────────────────────────────┐
    │ Type                │ Description                                      │
    ├─────────────────────┼──────────────────────────────────────────────────┤
    │ 1. RAW              │ Direct from the applicant — no transformation    │
    │                     │ e.g., monthly_income, loan_amount_requested      │
    ├─────────────────────┼──────────────────────────────────────────────────┤
    │ 2. CALCULATED       │ Derived mathematically from raw fields           │
    │                     │ e.g., debt_to_income_ratio, age                  │
    ├─────────────────────┼──────────────────────────────────────────────────┤
    │ 3. AGGREGATED       │ Summaries from multi-record data (bank stmts)    │
    │                     │ e.g., avg_monthly_balance, bounce_count          │
    └─────────────────────┴──────────────────────────────────────────────────┘

    IMPORTANT — VERIFICATION SIGNALS ARE NOT CREDIT RISK FEATURES:
        pan_verified, kyc_verified, mobile_verified, identity_match_status
        are WORKFLOW/ELIGIBILITY signals, not credit default risk features.
        They live in the applicant object and drive the product/review workflow.
        They are NOT included in the credit risk ML feature vector by default.
        See: IDENTITY RISK vs CREDIT RISK section at the bottom of this file.

IS THIS DATA SCIENCE?
    YES — Feature Engineering is core Data Science work.
    "The quality of your features determines the quality of your model."
    A model with good features and a simple algorithm often beats
    a complex model with poor features.

STATUS:
    Skeleton — feature groups documented, extraction function written.
    No model training yet. Features are defined and documented.
"""

import datetime
from typing import Optional, Dict, Any

# We import LoanApplicant to get type hints — not creating circular dependencies
# because feature_engineering.py doesn't modify applicant.py
from src.application.applicant import LoanApplicant


# ===========================================================================
# CREDIT RISK FEATURE GROUPS (what goes INTO the ML model)
# ===========================================================================
#
# GROUP 1: APPLICANT FEATURES
# ──────────────────────────────────────────────
# Feature                       Type          Source
# age                           CALCULATED    date_of_birth → compute age
# employment_type               RAW           application form
# employment_length_years       RAW           application form
#
# GROUP 2: INCOME FEATURES
# ──────────────────────────────────────────────
# Feature                       Type          Source
# monthly_income                RAW           application form
# income_stability_score        AGGREGATED    bank_statement_analyzer.py
# declared_vs_detected_diff_pct CALCULATED    income_verification.py
#
# GROUP 3: DEBT FEATURES
# ──────────────────────────────────────────────
# Feature                       Type          Source
# existing_emi_monthly          RAW           application form
# debt_to_income_ratio          CALCULATED    debt_analyzer.py
# loan_to_income_ratio          CALCULATED    debt_analyzer.py
# existing_loan_count           RAW           application form
# loan_amount_requested         RAW           application form
#
# GROUP 4: CREDIT FEATURES (from bureau)
# ──────────────────────────────────────────────
# Feature                       Type          Source
# credit_score                  RAW           credit bureau (demo: credit_profile.py)
# credit_utilization_ratio      RAW           credit bureau
# late_payment_count            RAW           credit bureau
# previous_defaults             RAW           credit bureau
# credit_history_length_months  RAW           credit bureau
# recent_credit_enquiries       RAW           credit bureau
#
# GROUP 5: BANKING BEHAVIOUR FEATURES
# ──────────────────────────────────────────────
# Feature                       Type          Source
# avg_monthly_balance           AGGREGATED    bank_statement_analyzer.py
# avg_monthly_cashflow          AGGREGATED    bank_statement_analyzer.py
# expense_to_income_ratio       AGGREGATED    bank_statement_analyzer.py
# negative_balance_frequency    AGGREGATED    bank_statement_analyzer.py
# bounce_count                  AGGREGATED    bank_statement_analyzer.py
#
# ===========================================================================
# VERIFICATION SIGNALS (NOT in the credit risk ML model by default)
# ===========================================================================
#
# These fields live on the LoanApplicant object and drive the APPLICATION
# WORKFLOW and ELIGIBILITY LAYER — not the credit default risk model.
#
# Signal                        Source               Used by
# pan_verified                  pan_verification.py  Eligibility gate
# kyc_verified                  aadhaar_kyc.py       Eligibility gate (mandatory)
# mobile_verified               mobile_verification  Eligibility gate
# identity_match_status         identity_checks.py   Manual review trigger
#
# WHY NOT IN THE ML MODEL?
#   - Failed PAN/KYC means the application is REJECTED or QUEUED FOR REVIEW
#     at the workflow level — BEFORE the ML model even runs.
#   - This is an IDENTITY RISK problem, not a CREDIT DEFAULT RISK problem.
#   - "PAN failed" ≠ "this person is likely to default on a loan."
#   - Mixing these two risks in a single model would be conceptually wrong.
#
# FUTURE CONSIDERATION:
#   If we ever have a sufficiently large dataset with verified/unverified outcomes
#   and a justified hypothesis, we can evaluate whether any of these signals
#   have residual predictive value for default — but this requires evidence, not assumption.
#
# ===========================================================================


# ---------------------------------------------------------------------------
# FUNCTION 1: Calculate Age from Date of Birth
# ---------------------------------------------------------------------------

def calculate_age(date_of_birth: str) -> Optional[int]:
    """
    Calculate current age in years from a date of birth string.

    WHY THIS IS ENGINEERED (not raw):
        We store DOB (raw) and calculate age at the time of application.
        This way, even if the model is retrained next year,
        we can recalculate age correctly from the stored DOB.
        If we stored age directly, it would go stale.

    ARGS:
        date_of_birth (str): DOB in "YYYY-MM-DD" format.

    RETURNS:
        int: Age in years. Returns None if DOB is empty or invalid.

    EXAMPLE:
        calculate_age("1995-03-15") → 31  (in 2026)
    """
    if not date_of_birth:
        return None
    try:
        dob = datetime.datetime.strptime(date_of_birth, "%Y-%m-%d").date()
        today = datetime.date.today()
        age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
        return age
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# NOTE: encode_identity_match() has been intentionally removed.
# ---------------------------------------------------------------------------
# Identity match status (PASS / MANUAL_REVIEW / FAIL) is an ELIGIBILITY SIGNAL.
# It is used by the application workflow and manual review layer.
# It is NOT encoded as a credit risk ML feature by default.
# See the VERIFICATION SIGNALS section above for the design rationale.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# FUNCTION 3: Build the Feature Vector (main function)
# ---------------------------------------------------------------------------

def build_feature_vector(applicant: LoanApplicant) -> Dict[str, Any]:
    """
    Takes a fully populated LoanApplicant object and extracts the CREDIT RISK
    feature vector — the set of signals used by the ML model.

    WHAT IS INCLUDED:
        Applicant, income, debt, credit history, and banking behaviour features.
        These are the signals that relate to CREDIT DEFAULT RISK.

    WHAT IS NOT INCLUDED:
        pan_verified, kyc_verified, mobile_verified, identity_match_status.
        These are VERIFICATION / ELIGIBILITY signals.
        They gate the application BEFORE this function is called.
        Failed verification → application does not reach this function.
        See: VERIFICATION SIGNALS section above for full rationale.

    IMPORTANT:
        - Missing values (None) are preserved — the ML pipeline will handle them.
        - We do NOT do imputation here — that happens in the preprocessing notebook.

    ARGS:
        applicant (LoanApplicant): A fully populated applicant object.

    RETURNS:
        Dict[str, Any]: A flat dictionary of feature_name → value.
                        This can be converted to a pandas DataFrame row.

    EXAMPLE:
        features = build_feature_vector(applicant)
        # features["debt_to_income_ratio"] → 0.4375
        # features["credit_score"] → 720
        # features["bounce_count"] → 0
        # NOTE: pan_verified is NOT in this dict — it is a workflow signal.

    NEXT STEP:
        In future notebooks (03_feature_engineering.ipynb), we will call this
        function on every row of the dataset and build a features DataFrame.
    """

    features: Dict[str, Any] = {}

    # -----------------------------------------------------------------------
    # GROUP 1: APPLICANT FEATURES
    # -----------------------------------------------------------------------

    # age: CALCULATED from date_of_birth
    features["age"] = (
        calculate_age(applicant.date_of_birth)
        if applicant.date_of_birth
        else applicant.age  # Use pre-computed if available
    )

    # employment_type: RAW — but we note that encoding (e.g., one-hot) happens later
    features["employment_type"] = applicant.employment_type
    # NOTE: This will be encoded in the preprocessing notebook.
    # "salaried" → [1, 0, 0], "self_employed" → [0, 1, 0], etc.

    features["employment_length_years"] = applicant.employment_length_years

    # -----------------------------------------------------------------------
    # GROUP 2: INCOME FEATURES
    # -----------------------------------------------------------------------

    features["monthly_income"] = applicant.monthly_income_declared
    features["income_stability_score"] = applicant.income_consistency_score
    features["declared_vs_detected_income_diff_pct"] = (
        applicant.declared_vs_detected_income_diff_pct
    )

    # -----------------------------------------------------------------------
    # GROUP 3: DEBT FEATURES
    # -----------------------------------------------------------------------

    features["existing_emi_monthly"] = applicant.existing_emi_monthly
    features["debt_to_income_ratio"] = applicant.debt_to_income_ratio
    features["loan_to_income_ratio"] = applicant.loan_to_income_ratio
    features["existing_loan_count"] = applicant.existing_loan_count
    features["loan_amount_requested"] = applicant.loan_amount_requested

    # -----------------------------------------------------------------------
    # GROUP 4: CREDIT FEATURES (from bureau)
    # -----------------------------------------------------------------------

    features["credit_score"] = applicant.credit_score
    features["credit_utilization_ratio"] = applicant.credit_utilization_ratio
    features["late_payment_count"] = applicant.late_payment_count
    features["previous_defaults"] = applicant.previous_defaults
    features["credit_history_length_months"] = applicant.credit_history_length_months
    features["recent_credit_enquiries"] = applicant.recent_credit_enquiries

    # -----------------------------------------------------------------------
    # GROUP 5: BANKING BEHAVIOUR FEATURES
    # -----------------------------------------------------------------------

    features["avg_monthly_balance"] = applicant.avg_monthly_balance
    features["avg_monthly_cashflow"] = applicant.monthly_cashflow
    features["expense_to_income_ratio"] = applicant.expense_to_income_ratio
    features["negative_balance_frequency"] = applicant.negative_balance_frequency
    features["bounce_count"] = applicant.bounce_count

    # -----------------------------------------------------------------------
    # VERIFICATION SIGNALS — NOT included in this feature vector
    # -----------------------------------------------------------------------
    # pan_verified, kyc_verified, mobile_verified, identity_match_status
    # are available on the applicant object for the product/workflow layer.
    # They do NOT belong in the credit risk ML feature vector by default.
    # IDENTITY RISK ≠ CREDIT DEFAULT RISK.
    # -----------------------------------------------------------------------

    return features


# ---------------------------------------------------------------------------
# WHAT HAPPENS TO THESE FEATURES NEXT?
# ---------------------------------------------------------------------------
#
# Step 1 (This file): Features are extracted into a dict/DataFrame row.
#
# Step 2 (Preprocessing notebook - Phase 2):
#   - Handle missing values (imputation strategy)
#   - Encode categorical variables (employment_type → one-hot or ordinal)
#   - Scale numeric features (StandardScaler or MinMaxScaler)
#   - Remove highly correlated features
#
# Step 3 (Modeling notebook - Phase 3):
#   - The preprocessed feature matrix X is fed to the ML model
#   - model.fit(X_train, y_train)  where y_train is 0/1 (defaulted)
#
# Step 4 (Prediction):
#   - model.predict_proba(X_new) → [0.28, 0.72]
#   - [prob_repay, prob_default] → Probability of Default = 0.72
#
# ---------------------------------------------------------------------------
# IDENTITY RISK vs CREDIT RISK — THE KEY DISTINCTION
# ---------------------------------------------------------------------------
#
# IDENTITY RISK asks: "Is this person who they say they are?"
#   Signals: PAN verified, KYC verified, Mobile verified, identity match
#   Handled by: src/verification/ layer + identity_checks.py
#   Decision: eligible to continue / manual review / rejected at workflow level
#
# CREDIT DEFAULT RISK asks: "How likely is this borrower to fail to repay?"
#   Signals: credit score, DTI, payment history, bank behaviour, income stability
#   Handled by: THIS FILE → ML model
#   Output: Probability of Default
#
# A failed PAN check is NOT evidence that someone will default on a loan.
# A passed KYC check does NOT guarantee someone will repay a loan.
# These are DIFFERENT questions answered by DIFFERENT layers.
#
# ---------------------------------------------------------------------------
# FEATURE IMPORTANCE NOTE:
# ---------------------------------------------------------------------------
# After training the model, we will use SHAP to understand which features
# mattered most. Based on typical credit risk models, we expect:
#
#   HIGH IMPORTANCE (usually):
#     - credit_score
#     - previous_defaults
#     - debt_to_income_ratio
#     - late_payment_count
#
#   MEDIUM IMPORTANCE:
#     - credit_utilization_ratio
#     - income_stability_score
#     - bounce_count
#
#   USEFUL SIGNALS:
#     - employment_type
#     - negative_balance_frequency
#     - declared_vs_detected_income_diff_pct
#
# We will VERIFY this through EDA and SHAP, not assume it.
# ---------------------------------------------------------------------------
