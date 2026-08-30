"""
src/application/applicant.py
=============================

PURPOSE:
    Defines what information belongs to a single loan application.
    This is the central data model of the entire system.
    Every other module (verification, financial, credit, features) reads
    from or writes to this structure.

THINK OF IT AS:
    A form that a customer fills out when applying for a loan.
    Some fields are filled by the customer directly (RAW INPUTS).
    Other fields are calculated or verified later (ENGINEERED / DERIVED).

LAYER IN THE WORKFLOW:
    Customer → Loan Application ← THIS FILE
    Then: Verification → Financial → Credit → Features → ML Model

STATUS:
    Skeleton — structure defined, no logic implemented yet.
"""

# ===========================================================================
# RAW vs ENGINEERED — THE MOST IMPORTANT DISTINCTION IN THIS PROJECT
# ===========================================================================
#
# RAW INPUTS:
#   Information the customer provides directly, or that we collect at source.
#   Examples:
#     - monthly_income       → customer declares ₹80,000
#     - existing_emi         → customer declares ₹20,000
#     - loan_amount_requested → customer wants ₹5,00,000
#     - date_of_birth        → customer provides their DOB
#
# ENGINEERED FEATURES:
#   Values we CALCULATE from raw inputs using logic or domain knowledge.
#   The ML model learns from these — not from raw inputs directly.
#   Examples:
#     - debt_to_income_ratio = existing_emi / monthly_income   ← calculated
#     - age = today's date - date_of_birth                     ← calculated
#     - loan_to_income_ratio = loan_amount / annual_income     ← calculated
#
# VERIFICATION-DERIVED FIELDS:
#   Information that comes from external checks (PAN, KYC, Mobile OTP).
#   Examples:
#     - pan_verified: True/False   ← comes from PAN verification step
#     - kyc_verified: True/False   ← comes from Aadhaar/KYC step
#     - identity_match: "PASS"/"FAIL"/"MANUAL_REVIEW" ← from identity_checks
#
# WHY THIS MATTERS FOR DATA SCIENCE:
#   - The ML model should receive FEATURES, not raw IDs or sensitive strings.
#   - We separate raw storage from model-ready features intentionally.
#   - This is called the "feature store" concept in production systems.
#
# ===========================================================================


# ---------------------------------------------------------------------------
# LOAN APPLICANT DATA MODEL
# ---------------------------------------------------------------------------
# We use a Python dataclass here.
# A dataclass is a clean way to define a data structure with named fields.
# It is simpler than a regular class for holding data.
#
# TO LEARN: What is a Python dataclass?
#   https://docs.python.org/3/library/dataclasses.html
#   Short answer: @dataclass automatically creates __init__, __repr__, etc.
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LoanApplicant:
    """
    Represents one complete loan application.

    Fields are grouped into logical sections so the structure
    is easy to understand and explain in an interview.

    Each field is annotated with:
        [RAW]         → provided directly by the applicant or collected at source
        [ENGINEERED]  → calculated from other fields
        [VERIFIED]    → derived from a verification step (PAN, KYC, OTP, etc.)
        [TARGET]      → what we are trying to predict (for model training only)

    USAGE EXAMPLE:
        applicant = LoanApplicant(
            applicant_id="APP001",
            full_name="Priya Sharma",
            age=29,
            monthly_income=80000.0,
            loan_amount_requested=500000.0,
            existing_emi=20000.0,
        )
    """

    # -----------------------------------------------------------------------
    # SECTION 1: Identity & Personal Information
    # [RAW] — collected from the application form
    # -----------------------------------------------------------------------

    applicant_id: str = ""
    # Unique ID for this application. Not a sensitive field — just a reference.

    full_name: str = ""
    # [RAW] Customer's full name as declared on the application.
    # Used for: identity matching against PAN/KYC name.

    date_of_birth: str = ""
    # [RAW] Format: "YYYY-MM-DD". Used to calculate age (an ENGINEERED feature).
    # We store DOB, not age, because age changes every year.

    gender: Optional[str] = None
    # [RAW] Optional — not all lenders collect this. Values: "M", "F", "Other".

    # -----------------------------------------------------------------------
    # SECTION 2: Loan Information
    # [RAW] — what the customer is asking for
    # -----------------------------------------------------------------------

    loan_amount_requested: float = 0.0
    # [RAW] How much money the customer wants to borrow, in Indian Rupees (₹).

    loan_purpose: str = ""
    # [RAW] Why do they need the loan? e.g., "home renovation", "medical", "education"
    # Some lenders price risk differently based on loan purpose.

    loan_tenure_months: int = 0
    # [RAW] How many months the customer wants to repay over. e.g., 24, 36, 60.

    # -----------------------------------------------------------------------
    # SECTION 3: Employment Information
    # [RAW] — collected from the application form
    # -----------------------------------------------------------------------

    employment_type: str = ""
    # [RAW] e.g., "salaried", "self_employed", "business_owner", "freelancer"
    # This is one of the most important risk signals — salaried is more stable.

    employer_name: str = ""
    # [RAW] Name of the company or business.

    employment_length_years: Optional[float] = None
    # [RAW] How long have they been at their current job?
    # Longer employment = more stability = lower risk (generally).

    # -----------------------------------------------------------------------
    # SECTION 4: Income Information
    # [RAW] — declared by the customer
    # -----------------------------------------------------------------------

    monthly_income_declared: float = 0.0
    # [RAW] Income the customer claims on the application form, in ₹/month.
    # "Declared" = what they say. This may be verified later.

    # -----------------------------------------------------------------------
    # SECTION 5: Existing Debt Information
    # [RAW] — declared by the customer
    # -----------------------------------------------------------------------

    existing_emi_monthly: float = 0.0
    # [RAW] Total monthly EMI obligations declared by the customer.
    # This is their existing debt payment — before this new loan.
    # Example: Home loan EMI ₹20,000 + bike loan EMI ₹8,000 = ₹28,000

    existing_loan_count: int = 0
    # [RAW] How many active loans does the customer currently have?

    # -----------------------------------------------------------------------
    # SECTION 6: Credit Information
    # [RAW/VERIFIED] — sourced from credit bureau (simulated in this project)
    # -----------------------------------------------------------------------

    credit_score: Optional[int] = None
    # [VERIFIED] Credit score from a bureau (CIBIL, Experian, etc.)
    # In this project: DEMO/SIMULATED — see src/credit/credit_profile.py
    # Range in India: typically 300–900. Above 750 is considered good.

    credit_history_length_months: Optional[int] = None
    # [VERIFIED] How many months of credit history does the bureau have?

    late_payment_count: Optional[int] = None
    # [VERIFIED] Number of late payments in the past 12 months.

    previous_defaults: Optional[int] = None
    # [VERIFIED] Number of times the customer has defaulted on a loan historically.

    credit_utilization_ratio: Optional[float] = None
    # [VERIFIED] What fraction of their available credit limit are they using?
    # e.g., 0.75 = using 75% of their credit limit. High utilization = risk signal.

    recent_credit_enquiries: Optional[int] = None
    # [VERIFIED] How many times has someone checked this person's credit recently?
    # Many recent enquiries can signal financial stress.

    # -----------------------------------------------------------------------
    # SECTION 7: Verification Status
    # [VERIFIED] — set by the verification layer (src/verification/)
    # These are NOT set by the customer — they are set by our verification code.
    # -----------------------------------------------------------------------

    pan_verified: Optional[bool] = None
    # [VERIFIED] Did PAN format check and mock verification pass?
    # Set by: src/verification/pan_verification.py

    kyc_verified: Optional[bool] = None
    # [VERIFIED] Did the Aadhaar/KYC check pass?
    # Set by: src/verification/aadhaar_kyc.py

    mobile_verified: Optional[bool] = None
    # [VERIFIED] Did the OTP verification pass?
    # Set by: src/verification/mobile_verification.py

    identity_match_status: Optional[str] = None
    # [VERIFIED] Cross-check result: "PASS", "FAIL", or "MANUAL_REVIEW"
    # Set by: src/verification/identity_checks.py
    # Example: "MANUAL_REVIEW — PAN name and bank name mismatch"

    # -----------------------------------------------------------------------
    # SECTION 8: Banking Behaviour
    # [VERIFIED/DERIVED] — sourced from bank statement analysis
    # Set by: src/financial/bank_statement_analyzer.py
    # -----------------------------------------------------------------------

    avg_monthly_balance: Optional[float] = None
    # [DERIVED] Average bank account balance over past 3–6 months.

    monthly_cashflow: Optional[float] = None
    # [DERIVED] Average (Income - Expenses) per month from bank statements.

    bounce_count: Optional[int] = None
    # [DERIVED] Number of bounced cheques or failed EMI payments.

    negative_balance_frequency: Optional[int] = None
    # [DERIVED] How many days/months did the account go into negative balance?

    income_detected_from_bank: Optional[float] = None
    # [DERIVED] Salary/income amount detected from bank credit patterns.
    # Compared against monthly_income_declared in income_verification.py

    # -----------------------------------------------------------------------
    # SECTION 9: ENGINEERED FEATURES
    # [ENGINEERED] — calculated from the raw fields above
    # Set by: src/features/feature_engineering.py
    # These are the features the ML model will actually receive.
    # -----------------------------------------------------------------------

    age: Optional[int] = None
    # [ENGINEERED] Calculated from date_of_birth at the time of application.
    # Formula: today's year - birth year (simplified)

    debt_to_income_ratio: Optional[float] = None
    # [ENGINEERED] = existing_emi_monthly / monthly_income_declared
    # One of the most important features in credit risk models.
    # Example: ₹28,000 / ₹80,000 = 0.35 → 35% of income goes to debt.

    loan_to_income_ratio: Optional[float] = None
    # [ENGINEERED] = loan_amount_requested / (monthly_income_declared * 12)
    # How big is the requested loan relative to annual income?

    income_consistency_score: Optional[float] = None
    # [ENGINEERED] How consistent is income month-to-month in bank statements?
    # Calculated by: src/financial/income_verification.py

    expense_to_income_ratio: Optional[float] = None
    # [ENGINEERED] = monthly_expenses / monthly_income
    # What fraction of income is being spent every month?

    declared_vs_detected_income_diff_pct: Optional[float] = None
    # [ENGINEERED] % difference between declared income and bank-detected income.
    # If very high, it may indicate misrepresentation.

    # -----------------------------------------------------------------------
    # SECTION 10: TARGET VARIABLE (for supervised ML training only)
    # [TARGET] — this field only exists in the training dataset.
    # In real production: we don't know this at application time!
    # -----------------------------------------------------------------------

    defaulted: Optional[int] = None
    # [TARGET] 1 = customer eventually defaulted on the loan, 0 = repaid.
    # This is what the ML model is trained to predict.
    # In the real world, we only know this AFTER the loan lifecycle ends.
    # In our training dataset, we use historical data where we know the outcome.


# ===========================================================================
# QUICK REFERENCE: WHO SETS WHAT
# ===========================================================================
#
# SECTION          → SET BY
# ─────────────────────────────────────────────────────
# Personal Info    → Customer (application form)
# Loan Info        → Customer (application form)
# Employment       → Customer (application form)
# Income           → Customer (application form)
# Debt             → Customer (application form)
# Credit Info      → src/credit/credit_profile.py    (simulated bureau)
# Verification     → src/verification/*.py           (PAN, KYC, OTP checks)
# Banking          → src/financial/bank_statement_analyzer.py
# Engineered       → src/features/feature_engineering.py
# Target           → Historical dataset only (for training)
# ===========================================================================
