"""
src/financial/income_verification.py
======================================

PURPOSE:
    Compares the income that the applicant DECLARED on the form
    with the income DETECTED from their bank statement analysis.

    This comparison is important because:
    - Applicants sometimes over-declare income to qualify for a larger loan.
    - Bank statement data gives us an independent, data-driven income estimate.
    - A large discrepancy is a risk signal.

REAL-WORLD ANALOGY:
    You say you earn ₹80,000/month.
    We look at your bank credits over 6 months.
    We see consistent monthly credits of ₹78,500.
    Difference = 1.9% → PASS (within acceptable range).

    But if you said ₹80,000 and we see ₹40,000 → MANUAL REVIEW or FAIL.

LAYER IN WORKFLOW:
    Bank Statement Analyzer → detects income from transactions
    ↓
    Income Verification  ← THIS FILE (compares declared vs detected)
    ↓
    Feature Engineering → income_consistency_score becomes an ML feature

IS THIS DATA SCIENCE?
    PARTLY.
    - The comparison logic is FINANCIAL DOMAIN KNOWLEDGE (not ML).
    - The OUTPUT (income_consistency_score, difference_pct) BECOMES an ML feature.
    - The threshold for PASS/FAIL is a BUSINESS RULE, not an ML decision.

STATUS:
    Skeleton — comparison logic documented and implemented.
    No ML. No external API. No real bank data.
"""

from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# DEMO ASSUMPTION CONSTANTS — configurable thresholds
# ---------------------------------------------------------------------------
# These values are project-specific assumptions for a learning/demo context.
# They are NOT universal banking rules or RBI guidelines.
#
# In a real lending system, thresholds like these are determined by:
#   - Historical data analysis (what difference % correlates with default?)
#   - Business risk appetite (how much income uncertainty are we willing to accept?)
#   - Lender policy (specific to each institution)
#   - Regulatory guidance (if any)
#
# For now, named constants make it easy to see and change these values
# without hunting through the logic.

INCOME_DIFF_PASS_THRESHOLD_PCT = 10.0
# Difference ≤ 10% → PASS
# DEMO ASSUMPTION: not a universal rule.

INCOME_DIFF_REVIEW_THRESHOLD_PCT = 30.0
# Difference 10–30% → MANUAL_REVIEW
# DEMO ASSUMPTION: not a universal rule.
# Above 30% → FAIL


# ---------------------------------------------------------------------------
# DATA STRUCTURE: Income Verification Result
# ---------------------------------------------------------------------------

@dataclass
class IncomeVerificationResult:
    """
    Represents the outcome of comparing declared vs. bank-detected income.

    FIELDS:
        declared_income:         What the applicant said on the form (₹/month).
        detected_income:         What we estimated from bank statements (₹/month).
        difference_pct:          % difference between declared and detected.
        income_consistency_score: 0.0 to 1.0 — how consistent is their income?
        salary_frequency:        How often does income appear? "MONTHLY", "IRREGULAR"
        status:                  "PASS", "MANUAL_REVIEW", or "FAIL"
        remarks:                 Human-readable explanation.
    """
    declared_income: float = 0.0
    detected_income: Optional[float] = None
    difference_pct: Optional[float] = None
    income_consistency_score: Optional[float] = None
    salary_frequency: Optional[str] = None    # "MONTHLY" | "IRREGULAR" | "UNKNOWN"
    status: str = "PENDING"    # "PASS" | "MANUAL_REVIEW" | "FAIL"
    remarks: str = ""


# ---------------------------------------------------------------------------
# FUNCTION 1: Calculate income difference percentage
# ---------------------------------------------------------------------------

def calculate_income_difference_pct(
    declared_income: float,
    detected_income: float,
) -> float:
    """
    Calculate the percentage difference between declared and detected income.

    Formula:
        diff_pct = |declared - detected| / declared × 100

    ARGS:
        declared_income (float): Income declared on the application (₹/month).
        detected_income (float): Income detected from bank statement (₹/month).

    RETURNS:
        float: Percentage difference (always positive).

    EXAMPLE:
        calculate_income_difference_pct(80000, 78500) → 1.875
        calculate_income_difference_pct(80000, 40000) → 50.0

    DATA SCIENCE NOTE:
        This value becomes `declared_vs_detected_income_diff_pct` in LoanApplicant.
        High values may indicate misrepresentation — a risk signal for the model.
    """
    if declared_income <= 0:
        raise ValueError("Declared income must be greater than zero.")

    difference = abs(declared_income - detected_income)
    difference_pct = (difference / declared_income) * 100
    return round(difference_pct, 2)


# ---------------------------------------------------------------------------
# FUNCTION 2: Verify income consistency
# ---------------------------------------------------------------------------

def verify_income(
    declared_income: float,
    detected_income: Optional[float],
    income_consistency_score: Optional[float] = None,
    salary_frequency: Optional[str] = None,
) -> IncomeVerificationResult:
    """
    Runs the full income verification comparison and returns a structured result.

    DECISION LOGIC (rule-based, not ML):
        ┌─────────────────────────────┬────────────────┐
        │ Condition                   │ Status         │
        ├─────────────────────────────┼────────────────┤
        │ No detected income at all   │ MANUAL_REVIEW  │
        │ Difference ≤ 10%            │ PASS           │
        │ Difference 10–30%           │ MANUAL_REVIEW  │
        │ Difference > 30%            │ FAIL           │
        └─────────────────────────────┴────────────────┘

    IMPORTANT NOTE:
        These thresholds are DEMO ASSUMPTIONS for a learning project.
        Real lenders define their own thresholds based on their risk policy.
        In production, these should be configurable parameters, not hard-coded.

    ARGS:
        declared_income (float):           Monthly income from application form (₹).
        detected_income (float):           Monthly income from bank analysis (₹).
        income_consistency_score (float):  How stable income is month-to-month (0–1).
        salary_frequency (str):           "MONTHLY", "IRREGULAR", or "UNKNOWN".

    RETURNS:
        IncomeVerificationResult

    EXAMPLE:
        result = verify_income(80000, 78500)
        # result.difference_pct → 1.875
        # result.status → "PASS"
        # result.remarks → "Declared income ₹80,000, detected ₹78,500. Difference: 1.88%."
    """
    result = IncomeVerificationResult(
        declared_income=declared_income,
        detected_income=detected_income,
        income_consistency_score=income_consistency_score,
        salary_frequency=salary_frequency,
    )

    # Handle missing detected income
    if detected_income is None:
        result.status = "MANUAL_REVIEW"
        result.remarks = (
            "No income detected from bank statement data. "
            "Manual review required."
        )
        return result

    # Calculate difference
    diff_pct = calculate_income_difference_pct(declared_income, detected_income)
    result.difference_pct = diff_pct

    # Apply decision rules (using named constants above — DEMO ASSUMPTIONS)
    if diff_pct <= INCOME_DIFF_PASS_THRESHOLD_PCT:
        result.status = "PASS"
        result.remarks = (
            f"Declared income ₹{declared_income:,.0f}, "
            f"detected ₹{detected_income:,.0f}. "
            f"Difference: {diff_pct:.2f}%. Income verified."
        )
    elif diff_pct <= INCOME_DIFF_REVIEW_THRESHOLD_PCT:
        result.status = "MANUAL_REVIEW"
        result.remarks = (
            f"Declared income ₹{declared_income:,.0f}, "
            f"detected ₹{detected_income:,.0f}. "
            f"Difference: {diff_pct:.2f}%. "
            f"Moderate discrepancy — manual review recommended."
        )
    else:
        result.status = "FAIL"
        result.remarks = (
            f"Declared income ₹{declared_income:,.0f}, "
            f"detected ₹{detected_income:,.0f}. "
            f"Difference: {diff_pct:.2f}%. "
            f"Large discrepancy — income declaration is unreliable."
        )

    return result
