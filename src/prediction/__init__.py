"""
src/prediction/__init__.py
===========================

Production inference pipeline package.

Provides the reusable CreditRiskPredictor, input schema validation,
and risk policy for the Phase 5 Step 1 inference layer.

Exported:
    CreditRiskPredictor — End-to-end prediction class
    ApplicantInput      — UCI benchmark applicant input dataclass
    PredictionResult    — Structured prediction output dataclass
    RiskPolicy          — Named policy constants and tier/decision functions
    InputValidationError — Raised on invalid applicant input
"""

from src.prediction.predictor import CreditRiskPredictor
from src.prediction.schemas import ApplicantInput, PredictionResult, InputValidationError
from src.prediction.risk_policy import RiskPolicy

__all__ = [
    "CreditRiskPredictor",
    "ApplicantInput",
    "PredictionResult",
    "InputValidationError",
    "RiskPolicy",
]
