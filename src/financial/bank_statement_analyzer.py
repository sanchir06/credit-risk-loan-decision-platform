"""
src/financial/bank_statement_analyzer.py
==========================================

PURPOSE:
    Converts raw bank transaction data into structured financial metrics.
    These metrics are later used as FEATURES for the credit risk ML model.

    This is one of the most important data processing steps in the entire system.
    The quality of the features produced here directly impacts model accuracy.

THE KEY INSIGHT — WHY THIS FILE MATTERS FOR DATA SCIENCE:
    Bank transactions are RAW DATA.
    The ML model cannot learn from raw transactions directly.
    We need to AGGREGATE them into meaningful monthly statistics.

    FLOW:
        Bank Transactions (raw CSV / JSON)
        ↓
        Bank Statement Analyzer  ← THIS FILE
        ↓
        Financial Behaviour Features (structured numbers)
        ↓
        Feature Engineering (src/features/feature_engineering.py)
        ↓
        Credit Risk ML Model

INTERVIEW INSIGHT:
    "I built a bank statement analyzer that converts raw transaction data
    into financial behaviour features — things like average monthly balance,
    bounce frequency, and cash flow consistency — which then feed our
    credit risk model."
    This is exactly what a Data Scientist at a lending company does.

IS THIS DATA SCIENCE?
    YES — This is DATA PROCESSING / FEATURE ENGINEERING.
    The ANALYSIS in this file is not ML, but its OUTPUTS directly become ML inputs.
    The distinction matters: aggregating transactions is engineering; 
    predicting default from features is ML.

STATUS:
    Skeleton — all output metrics documented with clear explanations.
    No real bank statement parsing yet. Demo/placeholder functions.
"""

from dataclasses import dataclass
from typing import Optional, List


# ---------------------------------------------------------------------------
# DATA STRUCTURE: Bank Statement Summary
# ---------------------------------------------------------------------------

@dataclass
class BankStatementSummary:
    """
    Aggregated financial metrics extracted from bank statement analysis.

    These are the OUTPUTS of this module and the INPUTS to Feature Engineering.

    Each field is documented with:
        - What it measures
        - Why it matters for credit risk
        - How it's calculated (at a high level)

    NOTE: All monetary values are in Indian Rupees (₹).
    """

    # -----------------------------------------------------------------------
    # INCOME SIGNALS
    # -----------------------------------------------------------------------

    avg_monthly_credits: Optional[float] = None
    """
    Average total money credited to the account per month.
    Proxy for income/revenue. Includes salary, transfers, refunds.
    Used in: Income Verification (compare to declared income).
    Credit risk relevance: HIGH — core income signal.
    """

    avg_salary_credit: Optional[float] = None
    """
    Average of identified salary credits per month.
    More specific than total credits — filters out non-salary inflows.
    How detected: Regular same-amount credits, often with "SALARY" in reference.
    Credit risk relevance: HIGH — most reliable income signal.
    """

    salary_months_detected: Optional[int] = None
    """
    Out of the last N months, in how many did we detect a salary?
    E.g., 5 out of 6 months → salary_months_detected = 5
    Credit risk relevance: HIGH — measures income regularity.
    """

    income_stability_score: Optional[float] = None
    """
    0.0 to 1.0 — How stable/consistent is the income?
    1.0 = same amount arrives every month (very stable)
    0.0 = completely irregular (freelancer/variable income)
    Calculation: coefficient of variation of monthly credits (inverted).
    Credit risk relevance: HIGH — stable income = lower default risk.
    """

    # -----------------------------------------------------------------------
    # EXPENSE SIGNALS
    # -----------------------------------------------------------------------

    avg_monthly_debits: Optional[float] = None
    """
    Average total money debited (spent/transferred out) per month.
    Credit risk relevance: MEDIUM — context for calculating cash flow.
    """

    avg_monthly_expenses: Optional[float] = None
    """
    Average estimated living expenses per month.
    Excludes EMI payments and large transfers (those are tracked separately).
    Credit risk relevance: HIGH — affects disposable income estimate.
    """

    # -----------------------------------------------------------------------
    # CASH FLOW SIGNALS
    # -----------------------------------------------------------------------

    avg_monthly_cashflow: Optional[float] = None
    """
    Average (Credits - Debits) per month.
    Positive = applicant spends less than they earn → healthy
    Negative = applicant spends more than they earn → risk signal
    Credit risk relevance: HIGH — directly indicates financial health.
    """

    expense_to_income_ratio: Optional[float] = None
    """
    avg_monthly_expenses / avg_monthly_credits
    E.g., 0.7 → spending 70% of income on expenses
    Credit risk relevance: HIGH — high ratio = less room for loan repayment.
    """

    # -----------------------------------------------------------------------
    # BALANCE SIGNALS
    # -----------------------------------------------------------------------

    avg_monthly_balance: Optional[float] = None
    """
    Average end-of-month account balance over the analysis period.
    Credit risk relevance: HIGH — consistent low balances = financial stress.
    """

    min_monthly_balance: Optional[float] = None
    """
    The lowest average monthly balance in the analysis period.
    Credit risk relevance: MEDIUM — one bad month may not matter much.
    """

    negative_balance_frequency: Optional[int] = None
    """
    Number of months (out of analysis period) where balance went negative.
    Credit risk relevance: HIGH — negative balance = overdraft/financial stress.
    """

    # -----------------------------------------------------------------------
    # DEBT PAYMENT SIGNALS
    # -----------------------------------------------------------------------

    detected_emi_payments: Optional[float] = None
    """
    Total monthly EMI payments detected from transaction patterns.
    Detection: Regular fixed debits, often with "EMI", "LOAN", "ECS" references.
    Credit risk relevance: HIGH — feeds into DTI calculation.
    """

    # -----------------------------------------------------------------------
    # STRESS SIGNALS
    # -----------------------------------------------------------------------

    bounce_count: Optional[int] = None
    """
    Number of bounced transactions (failed cheques, failed ECS/NACH debits).
    A bounce = a payment that was supposed to go out but couldn't (insufficient funds).
    Credit risk relevance: HIGH — bounces are a strong default predictor.
    Bounced EMI = serious delinquency signal.
    """

    inward_cheque_returns: Optional[int] = None
    """
    Number of cheques deposited that were returned (bounced on the other side).
    Credit risk relevance: MEDIUM — may indicate counterparty issues.
    """

    large_unusual_transactions: Optional[int] = None
    """
    Number of unusually large one-off transactions.
    Could indicate: asset sale, gambling, money movement, etc.
    Credit risk relevance: MEDIUM — requires context.
    """

    # -----------------------------------------------------------------------
    # METADATA
    # -----------------------------------------------------------------------

    analysis_period_months: Optional[int] = None
    """How many months of bank data were analyzed. 3–6 months is typical."""

    is_demo_data: bool = True
    """Always True in this project — flags this as simulated data."""


# ---------------------------------------------------------------------------
# FUNCTION: Analyze bank statements (DEMO / SKELETON)
# ---------------------------------------------------------------------------

def analyze_bank_statement_mock(
    monthly_credits: Optional[List[float]] = None,
    monthly_debits: Optional[List[float]] = None,
    monthly_balances: Optional[List[float]] = None,
    bounce_count: int = 0,
) -> BankStatementSummary:
    """
    ================================================================
    DEMO / SKELETON FUNCTION
    ================================================================

    Analyzes bank statement data and produces a BankStatementSummary.

    In a real system:
        - Input would be a raw list of transactions (date, amount, description)
        - We would parse, categorize, and aggregate them
        - Salary detection would use pattern matching on transaction descriptions
        - EMI detection would look for regular fixed debits

    In this demo:
        - We accept pre-aggregated monthly totals for simplicity
        - All outputs are clearly labeled as demo data

    ARGS:
        monthly_credits (List[float]):  List of total credits per month (₹).
                                        e.g., [78000, 79000, 78500, 80000, 79500, 78000]
        monthly_debits (List[float]):   List of total debits per month (₹).
        monthly_balances (List[float]): List of end-of-month balances (₹).
        bounce_count (int):             Number of bounced transactions.

    RETURNS:
        BankStatementSummary with all relevant metrics computed.

    EXAMPLE:
        summary = analyze_bank_statement_mock(
            monthly_credits=[78000, 79000, 78500, 80000, 79500, 78000],
            monthly_debits=[50000, 52000, 48000, 55000, 51000, 49000],
            monthly_balances=[30000, 28000, 32000, 25000, 31000, 29000],
            bounce_count=0,
        )
        # summary.avg_monthly_credits → 78833.33
        # summary.avg_monthly_cashflow → 26833.33
        # summary.income_stability_score → ~0.95 (very stable)
    """

    summary = BankStatementSummary(is_demo_data=True)

    # Use provided data or defaults
    credits = monthly_credits or []
    debits = monthly_debits or []
    balances = monthly_balances or []

    summary.analysis_period_months = max(len(credits), len(debits), len(balances))

    # Average credits (income proxy)
    if credits:
        summary.avg_monthly_credits = round(sum(credits) / len(credits), 2)
        summary.avg_salary_credit = summary.avg_monthly_credits  # Demo: assume all credits are salary
        summary.salary_months_detected = len([c for c in credits if c > 0])

        # Income stability: coefficient of variation (lower = more stable)
        if len(credits) > 1 and summary.avg_monthly_credits > 0:
            import statistics
            std_dev = statistics.stdev(credits)
            cv = std_dev / summary.avg_monthly_credits
            summary.income_stability_score = round(max(0.0, 1.0 - cv), 4)
        else:
            summary.income_stability_score = 1.0  # Only 1 data point — assume stable

    # Average debits (spending proxy)
    if debits:
        summary.avg_monthly_debits = round(sum(debits) / len(debits), 2)
        summary.avg_monthly_expenses = summary.avg_monthly_debits  # Demo: all debits = expenses

    # Cash flow
    if credits and debits and len(credits) == len(debits):
        monthly_cashflows = [c - d for c, d in zip(credits, debits)]
        summary.avg_monthly_cashflow = round(sum(monthly_cashflows) / len(monthly_cashflows), 2)

    # Expense-to-income ratio
    if summary.avg_monthly_expenses and summary.avg_monthly_credits:
        summary.expense_to_income_ratio = round(
            summary.avg_monthly_expenses / summary.avg_monthly_credits, 4
        )

    # Balance analysis
    if balances:
        summary.avg_monthly_balance = round(sum(balances) / len(balances), 2)
        summary.min_monthly_balance = round(min(balances), 2)
        summary.negative_balance_frequency = len([b for b in balances if b < 0])

    # Bounce count
    summary.bounce_count = bounce_count

    return summary


# ---------------------------------------------------------------------------
# WHAT IS NOT IN THIS FILE (and why)
# ---------------------------------------------------------------------------
#
# ❌ NOT HERE: Parsing raw bank statement PDFs
#    → A real enhancement requiring PDF parsing libraries.
#    → Would add complexity before we understand the feature logic.
#
# ❌ NOT HERE: ML model to classify transactions
#    → Transaction categorization (salary vs expense vs transfer) can eventually
#    → use ML, but rule-based keyword matching is sufficient to start.
#
# ❌ NOT HERE: Direct bank API integration (Account Aggregator / AA framework)
#    → India's Account Aggregator framework allows banks to share statement data.
#    → This is a future production integration, not for this learning phase.
# ---------------------------------------------------------------------------
