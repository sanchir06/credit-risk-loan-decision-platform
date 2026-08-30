"""
src/verification/mobile_verification.py
========================================

PURPOSE:
    Represents the mobile number verification step via OTP (One-Time Password).
    Mobile verification is a standard security and identity step in Indian
    digital lending — it confirms the customer can receive communications
    and adds a layer of anti-fraud protection.

IMPORTANT — DEMO NOTICE:
    This project does NOT use Twilio, MSG91, or any paid SMS service.
    We do NOT send real SMS messages.
    This module simulates the OTP workflow so you understand the pattern.

    In a real production system:
        - You would integrate with an SMS gateway (Twilio, MSG91, Exotel, etc.)
        - OTPs are generated server-side, stored temporarily (with expiry)
        - OTPs are NEVER stored permanently or logged
        - The mobile number is verified only — not stored as a surveillance record

LAYER IN WORKFLOW:
    Aadhaar / KYC Verification
    ↓
    Mobile Verification  ← THIS FILE
    ↓
    Identity Checks

IS THIS DATA SCIENCE?
    NO. Mobile OTP verification is a PRODUCT / SECURITY step.
    Its output (mobile_verified: True/False) is a VERIFICATION FEATURE
    that can feed into the ML model as a signal of identity confidence.

STATUS:
    Skeleton — OTP workflow documented, simulated functions written.
    No real SMS gateway integration yet.
"""

# ---------------------------------------------------------------------------
# WHAT IS OTP VERIFICATION?
# ---------------------------------------------------------------------------
# OTP = One-Time Password
# The typical flow in Indian digital lending:
#
#   1. Customer enters their mobile number in the loan application
#   2. Our system sends a 6-digit OTP to that number via SMS
#   3. Customer enters the OTP in the application interface
#   4. Our system checks: did the customer enter the correct OTP?
#   5. If correct → Mobile Verified ✅
#   6. If incorrect or expired → Retry or Fail ❌
#
# Why is this important?
#   - Confirms the customer actually controls the mobile number
#   - Prevents fake applications with someone else's number
#   - In India, mobile numbers are often linked to Aadhaar — cross-check possible
# ---------------------------------------------------------------------------

import random
import datetime
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# DATA STRUCTURE: OTP Session
# ---------------------------------------------------------------------------

@dataclass
class OTPSession:
    """
    Represents a single OTP verification session.

    In production:
        - otp_code would be stored in a server-side cache (e.g., Redis) with TTL
        - It would NEVER be returned to the client or stored in the database
        - After verification, the session is marked consumed and OTP is discarded

    In this demo:
        - otp_code is stored in this object for simulation purposes
        - is_demo_data is always True to flag this as non-production

    FIELDS:
        mobile_number:      The mobile number being verified (10-digit Indian format).
        otp_code:           The 6-digit OTP generated (DEMO ONLY — not for production).
        generated_at:       When the OTP was generated.
        expires_at:         When the OTP expires (typically 5–10 minutes).
        is_verified:        Did the user successfully verify?
        attempts_remaining: How many attempts left before lockout.
        is_demo_data:       Always True in this project.
    """
    mobile_number: str = ""
    otp_code: str = ""              # DEMO ONLY — never expose in production
    generated_at: str = ""
    expires_at: str = ""
    is_verified: bool = False
    attempts_remaining: int = 3     # Standard: 3 attempts before lockout
    is_demo_data: bool = True       # Always True — this is a demo project


@dataclass
class MobileVerificationResult:
    """
    The final result of a completed mobile verification.

    FIELDS:
        mobile_number:   The verified mobile number.
        status:          "VERIFIED", "FAILED", "EXPIRED", "MAX_ATTEMPTS"
        verified_at:     Timestamp of successful verification.
        remarks:         Human-readable explanation.
        is_demo_data:    Always True in this project.
    """
    mobile_number: str = ""
    status: str = "PENDING"    # "VERIFIED" | "FAILED" | "EXPIRED" | "MAX_ATTEMPTS"
    verified_at: Optional[str] = None
    remarks: str = ""
    is_demo_data: bool = True  # Always True — this is a demo project


# ---------------------------------------------------------------------------
# FUNCTION 1: Validate Indian Mobile Number Format
# ---------------------------------------------------------------------------

def is_valid_mobile_number(mobile_number: str) -> bool:
    """
    Checks whether the mobile number follows the Indian format.

    Indian mobile numbers:
        - 10 digits
        - Start with 6, 7, 8, or 9 (landlines start with lower digits)
        - Example: 9876543210

    ARGS:
        mobile_number (str): The number to validate (digits only, no spaces or dashes).

    RETURNS:
        bool: True if valid Indian mobile format.

    EXAMPLE:
        is_valid_mobile_number("9876543210")  → True
        is_valid_mobile_number("1234567890")  → False (doesn't start with 6–9)
        is_valid_mobile_number("98765")       → False (too short)

    DATA SCIENCE NOTE:
        This is data validation, not ML.
        An invalid mobile number format would reject the application immediately.
    """
    import re
    pattern = r"^[6-9][0-9]{9}$"
    return bool(re.match(pattern, mobile_number.strip()))


# ---------------------------------------------------------------------------
# FUNCTION 2: Generate a Mock OTP (DEMO ONLY)
# ---------------------------------------------------------------------------

def generate_otp_mock(mobile_number: str, expiry_minutes: int = 5) -> OTPSession:
    """
    ================================================================
    DEMO / MOCK FUNCTION — DOES NOT SEND A REAL SMS
    ================================================================

    Simulates generating and sending an OTP to a mobile number.

    In a real system:
        - Generate OTP server-side
        - Send via SMS gateway (Twilio, MSG91, etc.)
        - Store OTP hash (not plain text) in a cache with TTL
        - Never return the OTP code in the API response

    In this demo:
        - We generate a random 6-digit code
        - We return it inside the OTPSession (for testing purposes only)
        - In production, you'd never do this

    ARGS:
        mobile_number (str):   The mobile number to "send" the OTP to.
        expiry_minutes (int):  How many minutes until the OTP expires.

    RETURNS:
        OTPSession: Contains the generated OTP and session metadata.

    EXAMPLE:
        session = generate_otp_mock("9876543210")
        # session.otp_code → "492817" (random, for demo)
        # session.is_demo_data → True

    PRIVACY NOTE:
        In production, otp_code should never be logged or returned to any client.
    """
    if not is_valid_mobile_number(mobile_number):
        raise ValueError(f"Invalid Indian mobile number format: {mobile_number}")

    now = datetime.datetime.now()
    expiry = now + datetime.timedelta(minutes=expiry_minutes)

    otp = str(random.randint(100000, 999999))  # 6-digit OTP

    return OTPSession(
        mobile_number=mobile_number,
        otp_code=otp,   # DEMO ONLY — never do this in production
        generated_at=now.strftime("%Y-%m-%d %H:%M:%S"),
        expires_at=expiry.strftime("%Y-%m-%d %H:%M:%S"),
        is_verified=False,
        attempts_remaining=3,
        is_demo_data=True,
    )


# ---------------------------------------------------------------------------
# FUNCTION 3: Verify a Submitted OTP (DEMO ONLY)
# ---------------------------------------------------------------------------

def verify_otp_mock(
    session: OTPSession,
    submitted_otp: str,
) -> MobileVerificationResult:
    """
    ================================================================
    DEMO / MOCK FUNCTION — SIMULATES OTP VERIFICATION
    ================================================================

    Checks whether the OTP submitted by the user matches the one generated.

    In a real system:
        - Compare submitted OTP against the server-stored hash
        - Check expiry time
        - Decrement attempt counter
        - Lock the session after too many failures

    ARGS:
        session (OTPSession):    The active OTP session.
        submitted_otp (str):     The OTP the user typed in.

    RETURNS:
        MobileVerificationResult: The verification outcome.

    EXAMPLE:
        session = generate_otp_mock("9876543210")
        result = verify_otp_mock(session, session.otp_code)  # Pass correct OTP
        # result.status → "VERIFIED"
    """
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    result = MobileVerificationResult(
        mobile_number=session.mobile_number,
        is_demo_data=True,
    )

    # Check: did OTP expire?
    if now > session.expires_at:
        result.status = "EXPIRED"
        result.remarks = "[DEMO DATA] OTP has expired. Please request a new one."
        return result

    # Check: attempts exhausted?
    if session.attempts_remaining <= 0:
        result.status = "MAX_ATTEMPTS"
        result.remarks = "[DEMO DATA] Maximum OTP attempts reached. Application flagged."
        return result

    # Check: OTP match
    if submitted_otp.strip() == session.otp_code:
        result.status = "VERIFIED"
        result.verified_at = now
        result.remarks = "[DEMO DATA] OTP verified successfully."
    else:
        session.attempts_remaining -= 1
        result.status = "FAILED"
        result.remarks = (
            f"[DEMO DATA] Incorrect OTP. "
            f"{session.attempts_remaining} attempts remaining."
        )

    return result


# ---------------------------------------------------------------------------
# FUNCTION 4: Convert result to boolean for the applicant model
# ---------------------------------------------------------------------------

def mobile_is_verified(result: MobileVerificationResult) -> bool:
    """
    Simple helper: Is the mobile verified?

    RETURNS:
        True if status is "VERIFIED", False otherwise.

    USAGE:
        This boolean gets stored in LoanApplicant.mobile_verified
        and used as a feature in the ML model.
    """
    return result.status == "VERIFIED"
