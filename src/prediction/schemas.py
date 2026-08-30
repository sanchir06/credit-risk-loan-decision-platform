"""
src/prediction/schemas.py
==========================

PURPOSE:
    Defines the applicant input schema for the UCI Statlog German Credit
    benchmark inference pipeline, along with the structured prediction
    result output.

UCI BENCHMARK FEATURE SCHEMA:
    The German Credit dataset contains 20 features (7 numerical, 13 categorical)
    sourced from Prof. Dr. Hans Hofmann's 1994 historical European bank records.

    This schema exactly mirrors the feature set the preprocessing pipeline
    was fitted on during Phase 3/4. No features may be added or removed
    from the inference path without re-training the model.

IMPORTANT — ARCHITECTURAL SEPARATION:
    This UCI schema does NOT include any Indian lending fields.
    The following fields are INTENTIONALLY ABSENT:
        - pan_verified
        - aadhaar_kyc
        - cibil_score
        - Indian-specific KYC or identity fields

    Identity Risk ≠ Credit Risk ML Features.
    See src/features/feature_engineering.py for the design rationale.

VALIDATION STRATEGY:
    - Uses Python dataclasses (no external dependencies)
    - Manual validation via validate() method
    - Raises InputValidationError with human-readable messages
    - Rejects: missing fields, NaN, infinity, invalid categoricals

DATASET CONTEXT:
    UCI Statlog German Credit Data (1994) — historical European banking records.
    NOT Indian borrower data, NOT CIBIL data, NOT RBI-regulated production data.

PHASE 5 — STEP 1
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# UCI German Credit — categorical whitelists
# ---------------------------------------------------------------------------
# These are the exact category codes used in the original UCI dataset.
# Any input value not in this whitelist will be rejected.

UCI_CATEGORICAL_VALID_VALUES: Dict[str, List[str]] = {
    "status_existing_checking_account": ["A11", "A12", "A13", "A14"],
    "credit_history": ["A30", "A31", "A32", "A33", "A34"],
    "purpose": [
        "A40", "A41", "A42", "A43", "A44", "A45",
        "A46", "A47", "A48", "A49", "A410",
    ],
    "savings_account_bonds": ["A61", "A62", "A63", "A64", "A65"],
    "present_employment_since": ["A71", "A72", "A73", "A74", "A75"],
    "personal_status_sex": ["A91", "A92", "A93", "A94", "A95"],
    "other_debtors_guarantors": ["A101", "A102", "A103"],
    "property": ["A121", "A122", "A123", "A124"],
    "other_installment_plans": ["A141", "A142", "A143"],
    "housing": ["A151", "A152", "A153"],
    "job": ["A171", "A172", "A173", "A174"],
    "telephone": ["A191", "A192"],
    "foreign_worker": ["A201", "A202"],
}

# Numerical feature names (must be finite numeric values)
UCI_NUMERICAL_FEATURES: List[str] = [
    "duration_in_months",
    "credit_amount",
    "installment_rate_pct_disposable_income",
    "present_residence_since",
    "age_years",
    "existing_credits_count",
    "people_liable_maintenance",
]

# Categorical feature names
UCI_CATEGORICAL_FEATURES: List[str] = list(UCI_CATEGORICAL_VALID_VALUES.keys())

# All 20 features in original column order (matches training data)
UCI_ALL_FEATURES: List[str] = [
    "status_existing_checking_account",
    "duration_in_months",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings_account_bonds",
    "present_employment_since",
    "installment_rate_pct_disposable_income",
    "personal_status_sex",
    "other_debtors_guarantors",
    "present_residence_since",
    "property",
    "age_years",
    "other_installment_plans",
    "housing",
    "existing_credits_count",
    "job",
    "people_liable_maintenance",
    "telephone",
    "foreign_worker",
]

# Plausible numeric ranges drawn from UCI dataset distribution
# Format: feature -> (min_valid, max_valid)
# These are generous bounds to avoid rejecting edge cases while catching
# clearly erroneous values (e.g., negative loan amounts, age of 0).
UCI_NUMERICAL_RANGES: Dict[str, tuple] = {
    "duration_in_months":                   (1,    120),
    "credit_amount":                        (100,  200_000),
    "installment_rate_pct_disposable_income": (1,  4),
    "present_residence_since":              (1,    4),
    "age_years":                            (18,   100),
    "existing_credits_count":               (1,    4),
    "people_liable_maintenance":            (1,    2),
}


# ---------------------------------------------------------------------------
# Custom Exception
# ---------------------------------------------------------------------------

class InputValidationError(ValueError):
    """
    Raised when applicant input fails validation.

    Provides a human-readable description of what failed and why.
    Never silently passes invalid input.

    Examples:
        - Missing required field: 'age_years'
        - Invalid category 'XYZ' for 'credit_history'. Valid values: [A30, A31, ...]
        - 'credit_amount' is NaN — finite numeric value required.
        - 'duration_in_months' = -5 is outside valid range [1, 120].
    """


# ---------------------------------------------------------------------------
# ApplicantInput — UCI benchmark applicant representation
# ---------------------------------------------------------------------------

@dataclass
class ApplicantInput:
    """
    Represents one applicant's features for the UCI German Credit inference pipeline.

    All 20 UCI feature columns must be present. The feature names match
    the column headers in data/raw/german_credit_1000.csv exactly.

    IMPORTANT:
        This schema corresponds to the UCI Statlog German Credit Data (1994).
        It is a historical European benchmark dataset.
        It does NOT represent Indian borrowers, CIBIL scores, or RBI-regulated data.

    Construction:
        # From individual keyword arguments
        applicant = ApplicantInput(
            status_existing_checking_account="A11",
            duration_in_months=24,
            ...
        )

        # From a plain dict (most common usage)
        applicant = ApplicantInput.from_dict(raw_dict)
    """

    # Categorical features
    status_existing_checking_account: str = ""
    credit_history: str                   = ""
    purpose: str                          = ""
    savings_account_bonds: str            = ""
    present_employment_since: str         = ""
    personal_status_sex: str              = ""
    other_debtors_guarantors: str         = ""
    property: str                         = ""
    other_installment_plans: str          = ""
    housing: str                          = ""
    job: str                              = ""
    telephone: str                        = ""
    foreign_worker: str                   = ""

    # Numerical features
    duration_in_months: float                     = 0.0
    credit_amount: float                          = 0.0
    installment_rate_pct_disposable_income: float = 0.0
    present_residence_since: float                = 0.0
    age_years: float                              = 0.0
    existing_credits_count: float                 = 0.0
    people_liable_maintenance: float              = 0.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ApplicantInput":
        """
        Construct an ApplicantInput from a plain dictionary.

        The dictionary keys must match UCI column names exactly.
        Validation is performed via validate() after construction.

        Args:
            data (dict): Mapping of feature name → value.

        Returns:
            ApplicantInput instance.

        Raises:
            InputValidationError: If required fields are missing, types are wrong,
                                  values are invalid, NaN/Inf is detected, or
                                  categorical values are unrecognized.
        """
        instance = cls()
        for feature in UCI_ALL_FEATURES:
            if feature not in data:
                raise InputValidationError(
                    f"Missing required field: '{feature}'.\n"
                    f"All 20 UCI German Credit features must be provided."
                )
            setattr(instance, feature, data[feature])
        instance.validate()
        return instance

    def validate(self) -> None:
        """
        Validate all field values according to the UCI schema.

        Checks:
            1. All 20 required fields are present and not None.
            2. Numerical fields are finite numeric values (not NaN, not Inf).
            3. Numerical fields are within plausible UCI dataset ranges.
            4. Categorical fields match the UCI coding whitelist.

        Raises:
            InputValidationError: With a specific, human-readable message.
        """
        errors: List[str] = []

        # --- Numerical validation ---
        for feature in UCI_NUMERICAL_FEATURES:
            value = getattr(self, feature, None)

            # 1. Must be present
            if value is None:
                errors.append(
                    f"'{feature}' is None — a finite numeric value is required."
                )
                continue

            # 2. Must be numeric
            if not isinstance(value, (int, float)):
                errors.append(
                    f"'{feature}' = {value!r} is not numeric "
                    f"(got {type(value).__name__})."
                )
                continue

            # 3. Must not be NaN
            if math.isnan(value):
                errors.append(
                    f"'{feature}' is NaN — a finite numeric value is required."
                )
                continue

            # 4. Must not be infinite
            if math.isinf(value):
                errors.append(
                    f"'{feature}' is infinite — a finite numeric value is required."
                )
                continue

            # 5. Must be within plausible UCI range
            if feature in UCI_NUMERICAL_RANGES:
                lo, hi = UCI_NUMERICAL_RANGES[feature]
                if not (lo <= value <= hi):
                    errors.append(
                        f"'{feature}' = {value} is outside valid range "
                        f"[{lo}, {hi}]."
                    )

        # --- Categorical validation ---
        for feature, valid_values in UCI_CATEGORICAL_VALID_VALUES.items():
            value = getattr(self, feature, None)

            # 1. Must be present
            if value is None:
                errors.append(
                    f"'{feature}' is None — a category code is required."
                )
                continue

            # 2. Must be a string
            if not isinstance(value, str):
                errors.append(
                    f"'{feature}' = {value!r} is not a string "
                    f"(got {type(value).__name__})."
                )
                continue

            # 3. Must be one of the UCI whitelisted values
            if value not in valid_values:
                errors.append(
                    f"Invalid category '{value}' for '{feature}'. "
                    f"Valid UCI values: {valid_values}"
                )

        if errors:
            error_lines = "\n  - ".join(errors)
            raise InputValidationError(
                f"Applicant input validation failed ({len(errors)} error(s)):\n"
                f"  - {error_lines}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to a plain dictionary in UCI column order.

        Returns:
            Dict[str, Any]: Feature name → value, ordered as per UCI_ALL_FEATURES.
        """
        return {feature: getattr(self, feature) for feature in UCI_ALL_FEATURES}


# ---------------------------------------------------------------------------
# PredictionResult — structured inference output
# ---------------------------------------------------------------------------

@dataclass
class PredictionResult:
    """
    Structured result returned by CreditRiskPredictor.predict().

    Fields:
        estimated_pd  (float): Estimated Probability of Default in [0, 1].
                               Not formally calibrated.
        risk_score    (float): Internal demonstration risk score in [0, 100].
                               Formula: (1 - estimated_pd) * 100
        risk_tier     (str):   Risk tier: "LOW", "MEDIUM", or "HIGH".
        decision      (str):   Loan decision: "APPROVE", "MANUAL REVIEW", or "REJECT".
        model_name    (str):   Name of the model that produced this prediction.
        dataset       (str):   Dataset context for this prediction.

    IMPORTANT:
        This result is a DEMONSTRATION from the UCI German Credit benchmark model.
        It is NOT an Indian banking approval decision.
        Estimated PD is NOT formally calibrated.
    """

    estimated_pd: float
    risk_score:   float
    risk_tier:    str
    decision:     str
    model_name:   str = "Logistic Regression"
    dataset:      str = "UCI Statlog German Credit Data"

    def to_dict(self) -> Dict[str, Any]:
        """Return the prediction result as a plain dictionary."""
        return {
            "estimated_pd": self.estimated_pd,
            "risk_score":   self.risk_score,
            "risk_tier":    self.risk_tier,
            "decision":     self.decision,
            "model_name":   self.model_name,
            "dataset":      self.dataset,
        }
