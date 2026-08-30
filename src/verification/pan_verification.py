"""
src/verification/pan_verification.py
======================================

PURPOSE:
    Handles PAN (Permanent Account Number) related verification logic.
    PAN is issued by the Indian Income Tax Department.
    In a real lending workflow, PAN verification is one of the FIRST steps —
    it confirms the applicant's identity and tax filing history.

IMPORTANT — DEMO / MOCK VERIFICATION:
    This project does NOT have access to any real PAN verification API.
    We do NOT connect to Income Tax systems.
    We do NOT scrape government portals.
    We do NOT pretend to have production credentials.

    This module simulates what a PAN verification step would look like,
    so you can understand the WORKFLOW and the DATA it produces.

    In a real authorized production system, this module would be replaced
    by an integration with an authorized KYC/verification provider
    (e.g., a licensed fintech API aggregator).

LAYER IN WORKFLOW:
    Loan Application
    ↓
    PAN Verification  ← THIS FILE
    ↓
    Aadhaar/KYC Verification
    ↓
    ...

IS THIS DATA SCIENCE?
    NO. PAN verification is an IDENTITY VERIFICATION step, not ML.
    It is: Product Logic + Regulatory Compliance.
    The OUTPUT of this step (pan_verified: True/False) becomes an input
    to the Feature Engineering layer, which feeds the ML model.

STATUS:
    Skeleton — function signatures and documentation written.
    Logic is simulated/mock. No real API calls.
"""

# ---------------------------------------------------------------------------
# WHAT IS A PAN NUMBER?
# ---------------------------------------------------------------------------
# PAN = Permanent Account Number
# Format: AAAAA9999A
#   - 5 letters (first 3 are a sequence, 4th is entity type, 5th is surname initial)
#   - 4 digits
#   - 1 letter (checksum-like)
# Example (fictitious): ABCDE1234F
#
# Why does it matter for lending?
#   - Confirms applicant's legal identity
#   - Used to check ITR (Income Tax Return) filing history
#   - Cross-checked against credit bureau records
# ---------------------------------------------------------------------------

import re
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# DATA STRUCTURE: PAN Verification Result
# ---------------------------------------------------------------------------

@dataclass
class PANVerificationResult:
    """
    Represents the outcome of a PAN verification check.

    In a real system, this would be populated from an authorized API response.
    In this project, it is populated by our mock/simulated function.

    FIELDS:
        pan_number:     The PAN number that was checked (masked in logs/UI).
        format_valid:   Did the PAN pass the basic format check? (AAAAA9999A)
        name_match:     Does the PAN name match the application name?
        dob_match:      Does the PAN DOB match the application DOB?
        status:         Overall result: "VERIFIED", "FAILED", "MANUAL_REVIEW"
        remarks:        Human-readable reason for the result.
        is_demo_data:   Always True in this project — flags this as mock.
    """
    pan_number: str = ""
    format_valid: bool = False
    name_match: Optional[bool] = None
    dob_match: Optional[bool] = None
    status: str = "PENDING"       # "VERIFIED" | "FAILED" | "MANUAL_REVIEW"
    remarks: str = ""
    is_demo_data: bool = True     # Always True — this is a demo project


# ---------------------------------------------------------------------------
# FUNCTION 1: PAN Format Validator
# ---------------------------------------------------------------------------

def is_valid_pan_format(pan_number: str) -> bool:
    """
    Check whether a PAN number follows the correct format.

    PAN format: AAAAA9999A
        - Exactly 10 characters
        - First 5: uppercase letters
        - Next 4: digits
        - Last 1: uppercase letter

    This is a SIMPLE RULE CHECK — not ML, not an API call.
    It is the kind of basic validation that happens before even calling
    an external verification service.

    ARGS:
        pan_number (str): The PAN number string to validate.

    RETURNS:
        bool: True if format is valid, False otherwise.

    EXAMPLE:
        is_valid_pan_format("ABCDE1234F")  → True
        is_valid_pan_format("ABCDE12345")  → False (last char must be a letter)
        is_valid_pan_format("abcde1234f")  → False (must be uppercase)

    DATA SCIENCE NOTE:
        This is data validation, not feature engineering.
        Invalid PAN = the application cannot proceed.
    """
    # Regular expression pattern for PAN format
    pan_pattern = r"^[A-Z]{5}[0-9]{4}[A-Z]{1}$"
    return bool(re.match(pan_pattern, pan_number.strip()))


# ---------------------------------------------------------------------------
# FUNCTION 2: Mock PAN Verification (DEMO ONLY)
# ---------------------------------------------------------------------------

def verify_pan_mock(
    pan_number: str,
    declared_name: str,
    declared_dob: str,
) -> PANVerificationResult:
    """
    ================================================================
    DEMO / MOCK FUNCTION — NOT A REAL VERIFICATION API CALL
    ================================================================

    Simulates the verification response you would get from an
    authorized PAN verification service.

    In a real authorized integration:
        - You would send a POST request to an API
        - The API checks PAN against the Income Tax database
        - The API returns name, DOB, and verification status

    In this educational project:
        - We check PAN format (this is a real check)
        - We simulate name/DOB matching with simple string comparison
        - We return a result that looks like a real API response

    ARGS:
        pan_number (str):    The PAN number to verify.
        declared_name (str): Applicant's name as declared on the form.
        declared_dob (str):  Applicant's DOB as declared on the form (YYYY-MM-DD).

    RETURNS:
        PANVerificationResult: The verification outcome.

    EXAMPLE:
        result = verify_pan_mock("ABCDE1234F", "Priya Sharma", "1995-03-15")
        # result.status → "VERIFIED"
        # result.remarks → "PAN format valid. Name and DOB match."

    DATA SCIENCE RELEVANCE:
        The `status` field from this result ("VERIFIED"/"FAILED"/"MANUAL_REVIEW")
        eventually becomes the `pan_verified` boolean feature in the ML model.
        VERIFIED → pan_verified = True → a positive signal in the model.
    """

    result = PANVerificationResult(pan_number=pan_number, is_demo_data=True)

    # Step 1: Check format
    if not is_valid_pan_format(pan_number):
        result.format_valid = False
        result.status = "FAILED"
        result.remarks = "PAN format is invalid. Expected format: AAAAA9999A (e.g., ABCDE1234F)."
        return result

    result.format_valid = True

    # ----------------------------------------------------------------
    # DEMO SIMULATION:
    # In reality, we would call an external API here.
    # For the demo, we simulate a successful response.
    #
    # You can later replace this block with a real API call:
    #   response = requests.post(VERIFICATION_API_URL, json={...})
    #   data = response.json()
    # ----------------------------------------------------------------

    # Simulate: name match (for demo, we do a simple case-insensitive check)
    # In a real system, the API returns the registered name, and we compare.
    simulated_registered_name = declared_name  # Demo: assume match
    name_match = (
        simulated_registered_name.strip().lower() == declared_name.strip().lower()
    )

    # Simulate: DOB match
    simulated_registered_dob = declared_dob  # Demo: assume match
    dob_match = simulated_registered_dob == declared_dob

    result.name_match = name_match
    result.dob_match = dob_match

    # Determine overall status
    if name_match and dob_match:
        result.status = "VERIFIED"
        result.remarks = (
            "[DEMO DATA] PAN format valid. Name and DOB match. "
            "This is a simulated result — not a real government verification."
        )
    elif not name_match:
        result.status = "MANUAL_REVIEW"
        result.remarks = (
            "[DEMO DATA] PAN format valid but name does not match. "
            "Requires manual review."
        )
    else:
        result.status = "MANUAL_REVIEW"
        result.remarks = (
            "[DEMO DATA] PAN format valid but DOB does not match. "
            "Requires manual review."
        )

    return result


# ---------------------------------------------------------------------------
# FUNCTION 3: Convert result to a simple boolean for the applicant model
# ---------------------------------------------------------------------------

def pan_is_verified(result: PANVerificationResult) -> bool:
    """
    Simple helper: given a PANVerificationResult, is the PAN verified?

    RETURNS:
        True if status is "VERIFIED", False otherwise.

    USAGE:
        This boolean gets stored in LoanApplicant.pan_verified
        and later used as a feature in the ML model.
    """
    return result.status == "VERIFIED"


# ---------------------------------------------------------------------------
# WHAT IS NOT IN THIS FILE (and why)
# ---------------------------------------------------------------------------
#
# ❌ NOT HERE: Real Income Tax API credentials
#    → We don't have them. We don't pretend to have them.
#
# ❌ NOT HERE: ITR (Income Tax Return) lookup
#    → A future extension. Would require authorized API access.
#
# ❌ NOT HERE: ML model to validate PAN
#    → PAN verification is rule-based and API-based, NOT ML.
#
# ❌ NOT HERE: PAN-Aadhaar linking check
#    → Future feature. Would need UIDAI-PAN linkage API.
# ---------------------------------------------------------------------------
