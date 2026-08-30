"""
src/financial/debt_analyzer.py
================================

PURPOSE:
    Calculates the applicant's existing debt burden and key debt-related ratios.
    These ratios are among the most important features in any credit risk model.

THE CORE CONCEPT — DEBT-TO-INCOME RATIO (DTI):
    DTI = Total Monthly EMI Obligations / Monthly Income

    Example:
        Monthly Income = ₹80,000
        Existing EMIs:
            Home Loan    ₹20,000
            Bike Loan    ₹8,000
            Personal Loan ₹7,000
        Total EMI = ₹35,000

        DTI = 35,000 / 80,000 = 0.4375 → 43.75%

    Meaning: 43.75% of this person's income is already committed to existing debt.
    Now they want a new loan. Can they afford it?

WHY DTI MATTERS IN LENDING:
    It tells us: "How much of the applicant's income is ALREADY spoken for?"
    Higher DTI = less disposable income = higher repayment risk.

    DTI is used globally by banks, fintechs, and credit bureaus.
    In India, RBI guidelines consider DTI in evaluating borrower stress.

IS THIS DATA SCIENCE?
    NO — DTI is a FINANCIAL DOMAIN CALCULATION, not ML.
    It is arithmetic using domain knowledge.
    But DTI BECOMES a feature for the ML model — one of the most predictive ones.

    Understanding this distinction is important:
        DTI calculation → Financial domain knowledge
        "Does high DTI predict default?" → Data Science (you discover this in EDA)
        Using DTI as an ML feature → Feature Engineering

STATUS:
    Skeleton — calculations documented and implemented.
    No ML. Thresholds are demo assumptions, clearly labeled.
"""

from dataclasses import dataclass
from typing import Optional, List


# ---------------------------------------------------------------------------
# DEMO ASSUMPTION CONSTANTS — configurable thresholds
# ---------------------------------------------------------------------------
# These DTI bucket boundaries are project-specific demo assumptions.
# They are NOT universal banking rules, RBI guidelines, or industry standards.
#
# In a real lending system, these would be determined by:
#   - Historical data: at what DTI level does default rate increase sharply?
#   - Business risk appetite: how much debt burden is the lender comfortable with?
#   - Product policy: home loans vs personal loans have very different DTI norms
#   - Lender-specific underwriting policy (varies by institution)
#
# How thresholds SHOULD be selected (you will learn this in Phase 1 EDA):
#   1. Look at the distribution of DTI for defaulters vs non-defaulters.
#   2. Find where the default rate begins to climb significantly.
#   3. Use that as a data-informed starting point for the threshold.
#   4. Validate with business stakeholders.
#
# Simple named constants are enough for now — no config file needed.

DTI_LOW_THRESHOLD = 0.30
# Projected DTI < 30% → LOW_DEBT
# DEMO ASSUMPTION: not a universal rule.

DTI_MODERATE_THRESHOLD = 0.50
# Projected DTI 30–50% → MODERATE_DEBT
# DEMO ASSUMPTION: not a universal rule.

DTI_HIGH_THRESHOLD = 0.60
# Projected DTI 50–60% → HIGH_DEBT
# DEMO ASSUMPTION: not a universal rule.
# Above 60% → VERY_HIGH_DEBT

DEFAULT_INTEREST_RATE_PCT = 14.0
# Default interest rate used in EMI estimation: 14% per annum.
# DEMO ASSUMPTION: represents a mid-range personal loan rate in India.
# Real rates vary by lender, product, applicant credit profile.


# ---------------------------------------------------------------------------
# DATA STRUCTURE: Debt Analysis Result
# ---------------------------------------------------------------------------

@dataclass
class DebtAnalysisResult:
    """
    Represents the output of analyzing an applicant's existing debt burden.

    FIELDS:
        monthly_income:          The income used for ratio calculations (₹/month).
        total_existing_emi:      Sum of all current loan EMIs (₹/month).
        existing_loan_count:     Number of active loans.
        debt_to_income_ratio:    DTI = total_existing_emi / monthly_income
        loan_to_income_ratio:    LTI = requested_loan_amount / annual_income
        new_emi_if_approved:     The estimated EMI for the new loan being applied for.
        projected_total_emi:     total_existing_emi + new_emi_if_approved
        projected_dti:           What DTI would be IF the new loan is approved.
        assessment:              "LOW_DEBT", "MODERATE_DEBT", "HIGH_DEBT", "VERY_HIGH_DEBT"
        remarks:                 Human-readable explanation.
    """
    monthly_income: float = 0.0
    total_existing_emi: float = 0.0
    existing_loan_count: int = 0
    debt_to_income_ratio: Optional[float] = None
    loan_to_income_ratio: Optional[float] = None
    new_emi_if_approved: Optional[float] = None
    projected_total_emi: Optional[float] = None
    projected_dti: Optional[float] = None
    assessment: str = "UNKNOWN"
    remarks: str = ""


# ---------------------------------------------------------------------------
# FUNCTION 1: Calculate Debt-to-Income Ratio (DTI)
# ---------------------------------------------------------------------------

def calculate_dti(
    total_existing_emi: float,
    monthly_income: float,
) -> float:
    """
    Calculate the Debt-to-Income Ratio.

    DTI = total_existing_emi / monthly_income

    ARGS:
        total_existing_emi (float): Sum of all current monthly loan payments (₹).
        monthly_income (float):     Net monthly income (₹).

    RETURNS:
        float: DTI as a decimal (e.g., 0.4375 means 43.75%).

    EXAMPLE:
        calculate_dti(35000, 80000) → 0.4375

    DATA SCIENCE NOTE:
        This single number is one of the most powerful features in credit risk.
        EDA will likely show: higher DTI → higher default rate.
        In model training, DTI will probably be in the top 3 features by importance.
    """
    if monthly_income <= 0:
        raise ValueError("Monthly income must be greater than zero.")
    if total_existing_emi < 0:
        raise ValueError("Existing EMI cannot be negative.")

    return round(total_existing_emi / monthly_income, 4)


# ---------------------------------------------------------------------------
# FUNCTION 2: Calculate Loan-to-Income Ratio (LTI)
# ---------------------------------------------------------------------------

def calculate_lti(
    loan_amount_requested: float,
    monthly_income: float,
) -> float:
    """
    Calculate the Loan-to-Income Ratio.

    LTI = loan_amount_requested / (monthly_income × 12)

    Measures: "Is the loan amount reasonable relative to annual income?"

    ARGS:
        loan_amount_requested (float): The loan amount the applicant wants (₹).
        monthly_income (float):        Net monthly income (₹).

    RETURNS:
        float: LTI ratio (e.g., 5.2 means they want 5.2× their annual income).

    EXAMPLE:
        calculate_lti(500000, 80000)
        → 500000 / (80000 × 12) = 500000 / 960000 = 0.52
        → They want 52% of their annual income as a loan (reasonable).

        calculate_lti(2000000, 40000)
        → 2000000 / (40000 × 12) = 2000000 / 480000 = 4.17
        → They want 4.17× their annual income (high — more risk).

    DATA SCIENCE NOTE:
        LTI is used in India by banks to cap loan eligibility.
        e.g., "We only lend up to 4× annual income."
        But in this project, it is an INPUT FEATURE to the model,
        not a hard cutoff — the model can use it along with other features.
    """
    if monthly_income <= 0:
        raise ValueError("Monthly income must be greater than zero.")
    if loan_amount_requested <= 0:
        raise ValueError("Loan amount must be greater than zero.")

    annual_income = monthly_income * 12
    return round(loan_amount_requested / annual_income, 4)


# ---------------------------------------------------------------------------
# FUNCTION 3: Estimate new EMI (simplified)
# ---------------------------------------------------------------------------

def estimate_emi(
    principal: float,
    annual_interest_rate_pct: float,
    tenure_months: int,
) -> float:
    """
    Estimate the monthly EMI for a new loan using the standard EMI formula.

    EMI Formula (compound interest):
        EMI = P × r × (1+r)^n / ((1+r)^n - 1)
        where:
            P = principal (loan amount)
            r = monthly interest rate (annual rate / 12 / 100)
            n = tenure in months

    ARGS:
        principal (float):              Loan amount in ₹.
        annual_interest_rate_pct (float): Annual interest rate as percentage.
                                          e.g., 12.5 means 12.5% per year.
        tenure_months (int):            Loan tenure in months.

    RETURNS:
        float: Monthly EMI in ₹.

    EXAMPLE:
        estimate_emi(500000, 12.0, 48)
        → EMI ≈ ₹13,166 per month

    NOTE:
        This is a standard financial formula, not an approximation.
        Real banks use this exact formula (or very close variants).
    """
    if principal <= 0 or annual_interest_rate_pct <= 0 or tenure_months <= 0:
        raise ValueError("Principal, interest rate, and tenure must all be positive.")

    r = annual_interest_rate_pct / 12 / 100  # Monthly interest rate
    n = tenure_months

    emi = principal * r * ((1 + r) ** n) / (((1 + r) ** n) - 1)
    return round(emi, 2)


# ---------------------------------------------------------------------------
# FUNCTION 4: Full Debt Analysis
# ---------------------------------------------------------------------------

def analyze_debt(
    monthly_income: float,
    existing_emis: List[float],
    loan_amount_requested: float,
    loan_tenure_months: int,
    assumed_interest_rate_pct: float = 14.0,
) -> DebtAnalysisResult:
    """
    Runs a full debt burden analysis for a loan applicant.

    Calculates:
        - Current DTI (before this new loan)
        - Projected DTI (if this loan is approved)
        - Loan-to-Income ratio
        - Debt assessment: LOW / MODERATE / HIGH / VERY_HIGH

    ARGS:
        monthly_income (float):         Net monthly income in ₹.
        existing_emis (List[float]):     List of current EMI amounts, e.g. [20000, 8000, 7000]
        loan_amount_requested (float):   The new loan amount requested.
        loan_tenure_months (int):        Desired tenure for the new loan.
        assumed_interest_rate_pct (float): Interest rate to estimate new EMI.
                                           Default: 14% (a mid-range personal loan rate).
                                           DEMO ASSUMPTION — configurable.

    RETURNS:
        DebtAnalysisResult

    EXAMPLE:
        result = analyze_debt(
            monthly_income=80000,
            existing_emis=[20000, 8000, 7000],
            loan_amount_requested=500000,
            loan_tenure_months=48,
        )
        # result.debt_to_income_ratio → 0.4375  (43.75% existing DTI)
        # result.new_emi_if_approved  → ~₹13,166
        # result.projected_dti        → ~0.6020  (60.2% if new loan approved)
        # result.assessment           → "VERY_HIGH_DEBT"

    IMPORTANT NOTE ON THRESHOLDS:
        The assessment buckets below are DEMO ASSUMPTIONS for a learning project.
        Real lenders define their own risk policy thresholds.
        These should be configurable, not treated as universal rules.

        DTI Buckets used in this demo:
            < 30%  → LOW_DEBT
            30–50% → MODERATE_DEBT
            50–60% → HIGH_DEBT
            > 60%  → VERY_HIGH_DEBT
    """

    result = DebtAnalysisResult(
        monthly_income=monthly_income,
        total_existing_emi=sum(existing_emis),
        existing_loan_count=len(existing_emis),
    )

    # Current DTI
    result.debt_to_income_ratio = calculate_dti(result.total_existing_emi, monthly_income)

    # LTI
    result.loan_to_income_ratio = calculate_lti(loan_amount_requested, monthly_income)

    # Estimate new EMI if approved
    new_emi = estimate_emi(
        loan_amount_requested, assumed_interest_rate_pct, loan_tenure_months
    )
    result.new_emi_if_approved = new_emi

    # Projected total EMI and DTI
    result.projected_total_emi = round(result.total_existing_emi + new_emi, 2)
    result.projected_dti = calculate_dti(result.projected_total_emi, monthly_income)

    # Assessment — based on PROJECTED DTI (using named constants — DEMO ASSUMPTIONS)
    proj_dti = result.projected_dti

    if proj_dti < DTI_LOW_THRESHOLD:
        result.assessment = "LOW_DEBT"
        result.remarks = (
            f"Projected DTI: {proj_dti * 100:.1f}%. "
            f"Low debt burden — applicant has significant repayment capacity."
        )
    elif proj_dti < DTI_MODERATE_THRESHOLD:
        result.assessment = "MODERATE_DEBT"
        result.remarks = (
            f"Projected DTI: {proj_dti * 100:.1f}%. "
            f"Moderate debt burden — manageable but should be monitored."
        )
    elif proj_dti < DTI_HIGH_THRESHOLD:
        result.assessment = "HIGH_DEBT"
        result.remarks = (
            f"Projected DTI: {proj_dti * 100:.1f}%. "
            f"High debt burden — significant portion of income committed to EMIs."
        )
    else:
        result.assessment = "VERY_HIGH_DEBT"
        result.remarks = (
            f"Projected DTI: {proj_dti * 100:.1f}%. "
            f"Very high debt burden — over {int(DTI_HIGH_THRESHOLD * 100)}% of income would go to EMIs."
        )

    return result
