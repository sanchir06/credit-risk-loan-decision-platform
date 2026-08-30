"""
src/verification/aadhaar_kyc.py
================================

PURPOSE:
    Represents the Aadhaar-based KYC (Know Your Customer) step
    in the Indian digital lending workflow.

    KYC is a mandatory regulatory requirement for lending in India.
    It confirms who the customer is before any financial product is issued.

IMPORTANT — PRIVACY & DEMO NOTICE:
    Aadhaar numbers are highly sensitive biometric identifiers.
    This module is designed to be PRIVACY-CONSCIOUS from the start.

    We follow these principles:
    1. Aadhaar numbers are NOT stored permanently in this system.
    2. We use a masked reference (last 4 digits only) if needed for display.
    3. This project uses SIMULATED/DEMO KYC data only.
    4. We do NOT connect to UIDAI or any government portal.
    5. We do NOT use any Aadhaar e-KYC API — we have no such authorization.

    In a real production system:
        - The lender would use an authorized KYC provider
        - The customer would complete Offline/Paperless KYC using a signed XML file
        - Aadhaar number would only be used transiently for verification, not stored

LAYER IN WORKFLOW:
    PAN Verification
    ↓
    Aadhaar / KYC Verification  ← THIS FILE
    ↓
    Mobile Verification
    ↓
    Identity Checks

IS THIS DATA SCIENCE?
    NO. KYC is a REGULATORY COMPLIANCE + PRODUCT step.
    The output (kyc_verified: True/False, address, name) feeds into
    the Identity Checks module (identity_checks.py), which is RULE-BASED.
    The final boolean (kyc_verified) then becomes a feature for the ML model.

STATUS:
    Skeleton — structure and documentation written.
    Logic is simulated/mock. No UIDAI API calls.
"""

# ---------------------------------------------------------------------------
# WHAT IS AADHAAR-BASED KYC?
# ---------------------------------------------------------------------------
# Aadhaar is a 12-digit biometric identity number issued by UIDAI (Unique
# Identification Authority of India). Nearly every adult Indian resident has one.
#
# In digital lending, lenders use Aadhaar for:
#   - Confirming the customer's identity (name, DOB, address)
#   - Satisfying RBI KYC norms (mandatory for loan disbursement)
#   - Linking to PAN for additional identity cross-check
#
# "Paperless KYC" (also called Offline KYC):
#   - Customer downloads a signed XML file from the UIDAI portal
#   - Customer shares this XML with the lender
#   - Lender verifies the digital signature and extracts identity data
#   - Aadhaar number itself is NOT shared — only the signed XML
#   - This is the privacy-preserving approach mandated after 2018 SC ruling
#
# This project simulates the OUTPUT of such a KYC process.
# ---------------------------------------------------------------------------

from dataclasses import dataclass
from typing import Optional
import datetime


# ---------------------------------------------------------------------------
# DATA STRUCTURE: KYC Result
# ---------------------------------------------------------------------------

@dataclass
class KYCVerificationResult:
    """
    Represents the outcome of an Aadhaar KYC verification.

    In a real system:
        - This would be populated from a verified Aadhaar XML or API response.

    In this project:
        - This is a DEMO/SIMULATED result.
        - We never store or process real Aadhaar numbers.

    FIELDS EXPLAINED:
        kyc_reference_id:    Opaque reference ID (not the Aadhaar number itself).
        name_from_kyc:       Full name as returned by the KYC process.
        dob_from_kyc:        Date of birth as returned by the KYC process.
        address_verified:    Did we get a verified address from KYC?
        gender_from_kyc:     Gender as returned by KYC (optional).
        kyc_status:          "VERIFIED", "FAILED", or "MANUAL_REVIEW"
        verification_timestamp: When this check was performed.
        remarks:             Human-readable explanation of the outcome.
        is_demo_data:        Always True in this project.
    """
    kyc_reference_id: str = ""
    # A reference ID — NOT the Aadhaar number. Just an opaque token.

    name_from_kyc: str = ""
    # The name extracted from the verified KYC document.

    dob_from_kyc: str = ""
    # Date of birth from KYC document. Format: "YYYY-MM-DD"

    address_verified: bool = False
    # Did we successfully extract and verify an address?

    address_from_kyc: Optional[str] = None
    # The verified address (city, state at minimum). Full address is optional.
    # In production: be careful about how much address detail is stored.

    gender_from_kyc: Optional[str] = None
    # Gender from KYC, if needed. Optional.

    kyc_status: str = "PENDING"   # "VERIFIED" | "FAILED" | "MANUAL_REVIEW"
    verification_timestamp: str = ""
    remarks: str = ""
    is_demo_data: bool = True     # Always True — this is a demo project


# ---------------------------------------------------------------------------
# FUNCTION 1: Mock KYC Verification (DEMO ONLY)
# ---------------------------------------------------------------------------

def verify_kyc_mock(
    declared_name: str,
    declared_dob: str,
    declared_city: Optional[str] = None,
) -> KYCVerificationResult:
    """
    ================================================================
    DEMO / MOCK FUNCTION — NOT A REAL KYC API CALL
    ================================================================

    Simulates the output of a successful Aadhaar Paperless KYC.

    In a real authorized system:
        - Customer provides a signed Aadhaar XML or completes eKYC
        - The lender's system verifies the digital signature
        - Name, DOB, address are extracted and used for verification
        - The Aadhaar number is discarded after verification (not stored)

    In this educational project:
        - We simulate a response where name and DOB match
        - We produce a result in the same structure as a real response

    ARGS:
        declared_name (str):  Name from the application form.
        declared_dob (str):   DOB from the application form. Format: "YYYY-MM-DD"
        declared_city (str):  City from the application form (optional).

    RETURNS:
        KYCVerificationResult: The simulated KYC outcome.

    EXAMPLE:
        result = verify_kyc_mock("Priya Sharma", "1995-03-15", "Mumbai")
        # result.kyc_status → "VERIFIED"
        # result.name_from_kyc → "Priya Sharma"
        # result.is_demo_data → True

    DATA SCIENCE RELEVANCE:
        kyc_verified (True/False) → becomes a feature in Feature Engineering
        address_verified → can become a feature signal
        name_from_kyc → feeds into identity_checks.py for cross-matching
    """

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    result = KYCVerificationResult(
        kyc_reference_id=f"DEMO_KYC_{timestamp.replace(' ', '_')}",
        verification_timestamp=timestamp,
        is_demo_data=True,
    )

    # ----------------------------------------------------------------
    # DEMO SIMULATION:
    # In a real system, we would parse the signed Aadhaar XML here.
    # For the demo, we simulate a successful response where KYC data
    # matches the declared data.
    # ----------------------------------------------------------------

    # Simulate: KYC returns the same name and DOB as declared
    result.name_from_kyc = declared_name   # Demo: assume match
    result.dob_from_kyc = declared_dob     # Demo: assume match
    result.address_verified = True
    result.address_from_kyc = declared_city if declared_city else "Not provided"

    # Determine status (in real system, this comes from API response)
    result.kyc_status = "VERIFIED"
    result.remarks = (
        "[DEMO DATA] KYC simulated as verified. "
        "Name and DOB from KYC match application. "
        "This is NOT a real UIDAI or bureau verification."
    )

    return result


# ---------------------------------------------------------------------------
# FUNCTION 2: Convert result to a simple boolean
# ---------------------------------------------------------------------------

def kyc_is_verified(result: KYCVerificationResult) -> bool:
    """
    Simple helper: Is the KYC verified?

    RETURNS:
        True if kyc_status is "VERIFIED", False otherwise.

    USAGE:
        This boolean gets stored in LoanApplicant.kyc_verified
        and used as a feature in the ML model.
    """
    return result.kyc_status == "VERIFIED"


# ---------------------------------------------------------------------------
# PRIVACY REMINDER (embedded in code for learning purposes)
# ---------------------------------------------------------------------------
#
# THINGS WE INTENTIONALLY DO NOT DO IN THIS FILE:
#
# ❌ Store Aadhaar numbers anywhere in the database or file system
# ❌ Log Aadhaar numbers in any output
# ❌ Display Aadhaar numbers in the dashboard
# ❌ Connect to UIDAI — we have no authorization to do so
# ❌ Pretend we have eKYC API credentials
#
# THINGS TO DO IN PRODUCTION:
#
# ✅ Use only the last 4 digits if you need to display any reference
# ✅ Discard Aadhaar number after verification — store only the result
# ✅ Use a licensed, RBI-compliant KYC API provider
# ✅ Maintain an audit log of KYC checks (reference ID, timestamp, status)
# ✅ Encrypt any KYC-related data at rest
#
# See: docs/INDIA_DATA_PRIVACY.md for more details.
# ---------------------------------------------------------------------------
