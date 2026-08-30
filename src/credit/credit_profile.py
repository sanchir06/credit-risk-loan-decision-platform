"""
src/credit/credit_profile.py
==============================

PURPOSE:
    Represents credit bureau-style information for a loan applicant.
    In India, credit bureaus like CIBIL (TransUnion), Experian, CRIF, and
    Equifax maintain credit histories for individuals and share them with lenders.

IMPORTANT — DEMO NOTICE:
    This project does NOT have a real credit bureau integration.
    We do NOT connect to CIBIL, Experian, CRIF, or Equifax.
    We do NOT claim to have bureau API credentials.

    This module:
    1. Documents what information a real credit bureau provides.
    2. Simulates that information with clearly labeled DEMO DATA.
    3. Is structured so that a real bureau integration could replace
       the mock function later.

LAYER IN WORKFLOW:
    (External Credit Bureau)
    ↓
    Credit Profile  ← THIS FILE (fetches & structures bureau data)
    ↓
    Feature Engineering (credit features become ML inputs)
    ↓
    ML Credit Risk Model

IS THIS DATA SCIENCE?
    PARTIALLY:
    - Fetching the bureau data: NOT DS (it's an API call / data retrieval).
    - The CREDIT SCORE from the bureau: NOT DS (bureau calculates it, we receive it).
    - Using credit features in our ML model: YES — this is feature engineering.
    - Understanding which features predict default: YES — this is EDA.

INTERVIEW INSIGHT:
    "I integrated a credit profile layer that structures bureau-style information
    — credit score, utilization, payment history, enquiries — and feeds these
    as features into our XGBoost credit risk model."

STATUS:
    Skeleton — credit profile structure documented, mock data function written.
    No real bureau API. Clearly labeled as DEMO CREDIT PROFILE.
"""

# ---------------------------------------------------------------------------
# WHAT IS A CREDIT BUREAU? (India context)
# ---------------------------------------------------------------------------
# In India, four credit bureaus are licensed by RBI:
#   1. CIBIL (TransUnion CIBIL) — the most widely used
#   2. Experian India
#   3. CRIF High Mark
#   4. Equifax India
#
# Every time you take a loan, miss a payment, apply for credit, or pay on time,
# the lender reports this to the bureau. The bureau maintains your credit history
# and generates a credit score.
#
# CIBIL Score range: 300 – 900
#   < 600  : Poor    → high default risk
#   600–649: Fair    → moderate risk
#   650–699: Good    → acceptable
#   700–749: Very Good
#   750–900: Excellent → low default risk
#
# NOTE: These ranges are general industry benchmarks, not official RBI rules.
# ---------------------------------------------------------------------------

from dataclasses import dataclass
from typing import Optional
import random


# ---------------------------------------------------------------------------
# DATA STRUCTURE: Credit Profile
# ---------------------------------------------------------------------------

@dataclass
class CreditProfile:
    """
    Represents the credit history information for one applicant.

    In a real system: populated from a bureau API response (CIBIL, Experian, etc.)
    In this project: populated by generate_demo_credit_profile() — CLEARLY DEMO.

    FIELDS EXPLAINED (with credit risk relevance):

    credit_score:
        The bureau's summary score. Range: 300–900 (India).
        HIGH credit risk relevance — one of the top predictors.

    active_loans:
        Number of loans currently open.
        More active loans = more obligations = higher risk (generally).

    credit_cards:
        Number of credit cards.
        Many cards isn't automatically bad, but it contributes to utilization.

    credit_utilization_ratio:
        How much of their available credit limit they are using.
        e.g., 0.75 = 75% utilized.
        HIGH relevance — utilization > 30% is considered risky.

    late_payment_count:
        Number of late payments in the past 12 months.
        HIGH relevance — late payments = proven repayment issues.

    previous_defaults:
        Number of times the applicant has previously defaulted.
        VERY HIGH relevance — past default is the strongest predictor of future default.

    credit_history_length_months:
        How many months of credit history exist.
        Longer history = more data = better assessment possible.
        New borrowers (thin file) are harder to assess.

    recent_credit_enquiries:
        Number of credit checks (hard enquiries) in the past 6–12 months.
        Multiple enquiries = credit-hungry = possible financial stress signal.

    is_demo_data:
        Always True in this project.
    """

    applicant_id: str = ""

    # Core credit metrics
    credit_score: Optional[int] = None
    active_loans: Optional[int] = None
    credit_cards: Optional[int] = None

    # Credit utilization
    credit_utilization_ratio: Optional[float] = None
    # Range: 0.0 (using none of credit limit) to 1.0+ (over-limit)

    # Payment history
    late_payment_count: Optional[int] = None
    previous_defaults: Optional[int] = None

    # Credit history depth
    credit_history_length_months: Optional[int] = None

    # Recent credit-seeking behaviour
    recent_credit_enquiries: Optional[int] = None

    # Demo flag
    is_demo_data: bool = True   # Always True — this is a demo project


# ---------------------------------------------------------------------------
# FUNCTION: Generate Demo Credit Profile
# ---------------------------------------------------------------------------

def generate_demo_credit_profile(
    applicant_id: str,
    risk_level: str = "medium",
) -> CreditProfile:
    """
    ================================================================
    DEMO / MOCK FUNCTION — NOT REAL CIBIL/BUREAU DATA
    ================================================================

    Generates a simulated credit profile for a demo applicant.

    In a real system:
        - You would call the bureau's API with the applicant's PAN or consent token.
        - The bureau returns a structured credit report.
        - You parse the report into your internal CreditProfile structure.

    In this demo:
        - We generate plausible values based on a risk_level parameter.
        - This lets us test the ML model with realistic-looking credit data.

    ARGS:
        applicant_id (str): Reference ID for this applicant.
        risk_level (str):   "low", "medium", or "high" — controls the generated values.

    RETURNS:
        CreditProfile with simulated values. is_demo_data = True always.

    EXAMPLE:
        profile = generate_demo_credit_profile("APP001", risk_level="low")
        # profile.credit_score → ~780 (good score)
        # profile.previous_defaults → 0
        # profile.is_demo_data → True

        profile = generate_demo_credit_profile("APP002", risk_level="high")
        # profile.credit_score → ~550 (poor score)
        # profile.previous_defaults → 1–2
        # profile.is_demo_data → True

    DATA SCIENCE NOTE:
        These demo profiles are the placeholder for real bureau data.
        When we build the synthetic dataset for ML training, we will generate
        thousands of such profiles with realistic statistical distributions.
        The distribution will be designed to reflect real Indian lending portfolios.
    """

    profile = CreditProfile(applicant_id=applicant_id, is_demo_data=True)

    if risk_level == "low":
        # Good credit profile
        profile.credit_score = random.randint(720, 850)
        profile.active_loans = random.randint(0, 2)
        profile.credit_cards = random.randint(1, 3)
        profile.credit_utilization_ratio = round(random.uniform(0.05, 0.30), 2)
        profile.late_payment_count = 0
        profile.previous_defaults = 0
        profile.credit_history_length_months = random.randint(36, 120)
        profile.recent_credit_enquiries = random.randint(0, 2)

    elif risk_level == "high":
        # Poor credit profile
        profile.credit_score = random.randint(300, 600)
        profile.active_loans = random.randint(3, 7)
        profile.credit_cards = random.randint(2, 5)
        profile.credit_utilization_ratio = round(random.uniform(0.60, 1.0), 2)
        profile.late_payment_count = random.randint(2, 8)
        profile.previous_defaults = random.randint(1, 3)
        profile.credit_history_length_months = random.randint(6, 24)
        profile.recent_credit_enquiries = random.randint(4, 10)

    else:
        # Medium risk (default)
        profile.credit_score = random.randint(600, 719)
        profile.active_loans = random.randint(1, 3)
        profile.credit_cards = random.randint(1, 3)
        profile.credit_utilization_ratio = round(random.uniform(0.30, 0.60), 2)
        profile.late_payment_count = random.randint(0, 2)
        profile.previous_defaults = 0
        profile.credit_history_length_months = random.randint(12, 60)
        profile.recent_credit_enquiries = random.randint(1, 4)

    return profile


# ---------------------------------------------------------------------------
# IMPORTANT NOTE FOR INTERVIEW PREPARATION
# ---------------------------------------------------------------------------
#
# Question you might get: "How do you handle the thin-file problem?"
#
# The thin-file problem:
#   Many Indians are "thin-file" borrowers — they have little or no
#   formal credit history (no previous loans, no credit cards).
#   A bureau returns very little data for them.
#   Traditional credit scoring models struggle with thin-file customers.
#
# How our project handles it:
#   For thin-file customers, we rely MORE on:
#   - Bank statement analysis (income, cash flow, spending patterns)
#   - Employment stability (salaried vs. self-employed)
#   - Alternative data (mobile, social, utility payments — future enhancement)
#   LESS on:
#   - Traditional credit score (which may not exist)
#
# This is a real problem in Indian fintech that makes the ML challenge
# more interesting than a standard Western credit dataset.
# ---------------------------------------------------------------------------
