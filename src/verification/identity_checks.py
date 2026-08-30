"""
src/verification/identity_checks.py
=====================================

PURPOSE:
    Cross-checks identity information coming from different sources
    and flags inconsistencies that may require manual review.

    This is the CONSOLIDATION step after individual verifications
    (PAN, KYC, Mobile) are complete. It compares names, DOBs, and
    other identifiers across all sources to detect mismatches.

REAL-WORLD ANALOGY:
    Imagine you apply for a loan and provide:
        - Application form: Name = "Priya Sharma"
        - PAN card: Name = "P. Sharma"
        - Bank account: Name = "Priya S. Sharma"

    A fraud check system would flag: "Names don't fully match across sources."
    This module does that check.

LAYER IN WORKFLOW:
    PAN Verification
    ↓
    KYC Verification
    ↓
    Mobile Verification
    ↓
    Identity Checks  ← THIS FILE (consolidates all three)
    ↓
    Financial Analysis

IS THIS DATA SCIENCE?
    PARTIALLY. The initial version is RULE-BASED (no ML).
    Simple name matching, DOB comparison, status checks.
    ML is NOT needed here — clear rules are sufficient and more explainable.

    The OUTPUT of this module (identity_match_status) can become:
    - A feature for the ML model
    - A trigger for Manual Review workflow
    - An audit trail entry

STATUS:
    Skeleton — rule-based checks documented and implemented simply.
    No ML. No external API calls.
"""

# ---------------------------------------------------------------------------
# WHAT DOES IDENTITY CHECKING MEAN IN INDIAN LENDING?
# ---------------------------------------------------------------------------
# An Indian loan applicant provides details on multiple channels:
#
#   SOURCE               FIELD
#   ─────────────────────────────────────────────────
#   Application form   → Declared Name, DOB, Mobile
#   PAN verification   → Registered Name, DOB
#   Aadhaar/KYC        → KYC Name, DOB, Address
#   Bank statement     → Account Holder Name
#
# A legitimate applicant should have CONSISTENT information across all sources.
# Mismatches can indicate:
#   - Typos / data entry errors → need correction
#   - Name variations (maiden name, initials) → needs human review
#   - Fraud attempt → needs escalation
#
# This module does these cross-checks using simple rule-based logic.
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field
from typing import Optional, List


# ---------------------------------------------------------------------------
# DATA STRUCTURE: Identity Check Result
# ---------------------------------------------------------------------------

@dataclass
class IdentityCheckResult:
    """
    The consolidated result of all identity cross-checks.

    FIELDS:
        overall_status:   "PASS", "MANUAL_REVIEW", or "FAIL"
        checks_performed: List of checks we ran (for audit trail)
        issues_found:     List of specific problems found
        remarks:          Human-readable summary
    """
    overall_status: str = "PENDING"    # "PASS" | "MANUAL_REVIEW" | "FAIL"
    checks_performed: List[str] = field(default_factory=list)
    issues_found: List[str] = field(default_factory=list)
    remarks: str = ""


# ---------------------------------------------------------------------------
# HELPER: Simple name similarity check
# ---------------------------------------------------------------------------

def _names_are_consistent(name_a: str, name_b: str) -> bool:
    """
    Checks whether two names are "close enough" to be considered the same person.

    This is intentionally simple for a beginner project:
        - Lowercase both names
        - Strip extra whitespace
        - Check if one contains the other, or they're identical

    Real production systems use fuzzy string matching (e.g., Jaro-Winkler)
    to handle Indian name variations like:
        "Priya Sharma" vs "P. Sharma" vs "PRIYA SHARMA"

    For now: simple substring check + exact match.
    """
    a = name_a.strip().lower()
    b = name_b.strip().lower()

    if not a or not b:
        return False  # Can't match an empty name

    # Exact match
    if a == b:
        return True

    # One contains the other (handles "Priya Sharma" vs "Priya S.")
    if a in b or b in a:
        return True

    # Word overlap — at least the surname must match
    # (handles "Priya Sharma" vs "P. Sharma")
    words_a = set(a.split())
    words_b = set(b.split())
    overlap = words_a.intersection(words_b)

    # If at least 1 word in common and it's more than a single letter
    significant_overlap = [w for w in overlap if len(w) > 1]
    return len(significant_overlap) >= 1


# ---------------------------------------------------------------------------
# MAIN FUNCTION: Run all identity checks
# ---------------------------------------------------------------------------

def run_identity_checks(
    declared_name: str,
    declared_dob: str,
    pan_name: Optional[str] = None,
    pan_dob: Optional[str] = None,
    kyc_name: Optional[str] = None,
    kyc_dob: Optional[str] = None,
    bank_account_name: Optional[str] = None,
    mobile_verified: Optional[bool] = None,
    pan_verified: Optional[bool] = None,
    kyc_verified: Optional[bool] = None,
) -> IdentityCheckResult:
    """
    Cross-checks all identity information from multiple sources.

    RULE-BASED LOGIC (not ML):
        We define clear rules for PASS, MANUAL_REVIEW, and FAIL.
        These rules should be documented, reviewable, and configurable.

    ARGS:
        declared_name (str):        Name from application form.
        declared_dob (str):         DOB from application form.
        pan_name (str):             Name returned by PAN verification.
        pan_dob (str):              DOB returned by PAN verification.
        kyc_name (str):             Name returned by KYC verification.
        kyc_dob (str):              DOB returned by KYC verification.
        bank_account_name (str):    Name on the bank account (from bank statement).
        mobile_verified (bool):     Did mobile OTP verification pass?
        pan_verified (bool):        Did PAN verification pass?
        kyc_verified (bool):        Did KYC verification pass?

    RETURNS:
        IdentityCheckResult: Overall status, list of issues, and remarks.

    EXAMPLE:
        result = run_identity_checks(
            declared_name="Priya Sharma",
            declared_dob="1995-03-15",
            pan_name="P. Sharma",
            pan_dob="1995-03-15",
            kyc_name="Priya Sharma",
            kyc_dob="1995-03-15",
            bank_account_name="PRIYA S SHARMA",
            mobile_verified=True,
            pan_verified=True,
            kyc_verified=True,
        )
        # result.overall_status → "PASS" (names are consistent enough)

    DATA SCIENCE RELEVANCE:
        overall_status → stored as `identity_match_status` in LoanApplicant.
        issues_found → used to explain a MANUAL_REVIEW decision in the workflow.

        This field is an ELIGIBILITY / WORKFLOW SIGNAL, not a credit risk ML feature.
        It answers: "Are the identity documents consistent?" — an identity question.
        It does NOT answer: "How likely is this person to default?" — a credit question.

        IDENTITY RISK ≠ CREDIT DEFAULT RISK. Do not conflate them.
    """

    result = IdentityCheckResult()
    issues = []
    checks = []

    # ----------------------------------------------------------------
    # CHECK 1: PAN verification passed?
    # ----------------------------------------------------------------
    checks.append("PAN_VERIFICATION_STATUS")
    if pan_verified is False:
        issues.append("PAN verification did not pass.")
    elif pan_verified is None:
        issues.append("PAN verification not yet performed.")

    # ----------------------------------------------------------------
    # CHECK 2: KYC verification passed?
    # ----------------------------------------------------------------
    checks.append("KYC_VERIFICATION_STATUS")
    if kyc_verified is False:
        issues.append("KYC verification did not pass.")
    elif kyc_verified is None:
        issues.append("KYC verification not yet performed.")

    # ----------------------------------------------------------------
    # CHECK 3: Mobile verified?
    # ----------------------------------------------------------------
    checks.append("MOBILE_VERIFICATION_STATUS")
    if mobile_verified is False:
        issues.append("Mobile number not verified.")
    elif mobile_verified is None:
        issues.append("Mobile verification not yet performed.")

    # ----------------------------------------------------------------
    # CHECK 4: Name consistency — Application vs PAN
    # ----------------------------------------------------------------
    if pan_name:
        checks.append("NAME_MATCH_APPLICATION_VS_PAN")
        if not _names_are_consistent(declared_name, pan_name):
            issues.append(
                f"Name mismatch: Application says '{declared_name}', "
                f"PAN shows '{pan_name}'."
            )

    # ----------------------------------------------------------------
    # CHECK 5: Name consistency — Application vs KYC
    # ----------------------------------------------------------------
    if kyc_name:
        checks.append("NAME_MATCH_APPLICATION_VS_KYC")
        if not _names_are_consistent(declared_name, kyc_name):
            issues.append(
                f"Name mismatch: Application says '{declared_name}', "
                f"KYC shows '{kyc_name}'."
            )

    # ----------------------------------------------------------------
    # CHECK 6: Name consistency — Application vs Bank Account
    # ----------------------------------------------------------------
    if bank_account_name:
        checks.append("NAME_MATCH_APPLICATION_VS_BANK")
        if not _names_are_consistent(declared_name, bank_account_name):
            issues.append(
                f"Name mismatch: Application says '{declared_name}', "
                f"Bank account shows '{bank_account_name}'."
            )

    # ----------------------------------------------------------------
    # CHECK 7: DOB consistency — Application vs PAN
    # ----------------------------------------------------------------
    if pan_dob:
        checks.append("DOB_MATCH_APPLICATION_VS_PAN")
        if pan_dob.strip() != declared_dob.strip():
            issues.append(
                f"DOB mismatch: Application says '{declared_dob}', "
                f"PAN shows '{pan_dob}'."
            )

    # ----------------------------------------------------------------
    # CHECK 8: DOB consistency — Application vs KYC
    # ----------------------------------------------------------------
    if kyc_dob:
        checks.append("DOB_MATCH_APPLICATION_VS_KYC")
        if kyc_dob.strip() != declared_dob.strip():
            issues.append(
                f"DOB mismatch: Application says '{declared_dob}', "
                f"KYC shows '{kyc_dob}'."
            )

    # ----------------------------------------------------------------
    # DECISION LOGIC: How many issues trigger which status?
    # ----------------------------------------------------------------
    #
    # IMPORTANT NOTE FOR LEARNERS:
    #   These thresholds are DEMO ASSUMPTIONS, not universal banking rules.
    #   In production, lenders define their own risk policy thresholds.
    #   The thresholds should be configurable, not hard-coded.
    #
    # RULE (demo):
    #   0 issues   → PASS
    #   1–2 issues → MANUAL REVIEW (could be data entry errors)
    #   3+ issues  → FAIL (too many inconsistencies)
    #
    # Hard FAIL conditions (regardless of count):
    #   - PAN not verified
    #   - KYC not verified
    # ----------------------------------------------------------------

    hard_fail = (pan_verified is False) or (kyc_verified is False)

    if hard_fail:
        result.overall_status = "FAIL"
        result.remarks = (
            "Hard FAIL: PAN or KYC verification failed. "
            "Application cannot proceed without verified identity documents."
        )
    elif len(issues) == 0:
        result.overall_status = "PASS"
        result.remarks = "All identity checks passed. No inconsistencies found."
    elif len(issues) <= 2:
        result.overall_status = "MANUAL_REVIEW"
        result.remarks = (
            f"Identity check requires manual review. "
            f"{len(issues)} issue(s) found: " + " | ".join(issues)
        )
    else:
        result.overall_status = "FAIL"
        result.remarks = (
            f"Too many identity inconsistencies ({len(issues)} issues). "
            f"Application flagged for rejection."
        )

    result.checks_performed = checks
    result.issues_found = issues
    return result


# ---------------------------------------------------------------------------
# WHAT IS NOT IN THIS FILE (and why)
# ---------------------------------------------------------------------------
#
# ❌ NOT HERE: ML model for fraud detection
#    → Rule-based logic is sufficient for this step and more explainable.
#    → ML fraud models are a future enhancement, not a day-1 requirement.
#
# ❌ NOT HERE: Database lookups for known fraudsters
#    → Future enhancement requiring a fraud database.
#
# ❌ NOT HERE: Face matching / biometric verification
#    → Advanced, requires specialized APIs. Out of scope for this project.
#
# ✅ WHAT THIS FILE DOES:
#    Rule-based cross-matching of names/DOBs across PAN, KYC, Bank sources.
#    Simple, explainable, beginner-friendly. Exactly what's needed at Phase 0.
# ---------------------------------------------------------------------------
