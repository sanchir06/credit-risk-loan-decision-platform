"""
src/prediction/predictor.py
============================

PURPOSE:
    The production inference class for the UCI German Credit benchmark pipeline.
    Converts a raw applicant feature dictionary into a structured prediction
    result — without ever requiring the Jupyter notebook to run.

INFERENCE PIPELINE:
    Raw Applicant Input (dict or ApplicantInput)
            ↓
    Input Validation (schemas.py — rejects invalid/missing/NaN/Inf)
            ↓
    Feature Preparation (DataFrame with UCI column order)
            ↓
    Saved Preprocessor (ColumnTransformer — transform only, NO .fit())
            ↓
    Saved Model (LogisticRegression.predict_proba())
            ↓
    Estimated Probability of Default (PD)
            ↓
    Internal Risk Score  [(1 - PD) * 100]
            ↓
    Risk Tier  (LOW / MEDIUM / HIGH)
            ↓
    Loan Decision  (APPROVE / MANUAL REVIEW / REJECT)
            ↓
    PredictionResult (structured output)

LEAKAGE PREVENTION:
    The predictor NEVER calls .fit() on any object.
    The preprocessor and model are loaded from disk (already fitted on X_train).
    New applicant data is only ever transformed, never used for fitting.

DATASET CONTEXT:
    This predictor serves the UCI Statlog German Credit Data benchmark model.
    The UCI dataset is historical European banking data (1994).
    It is NOT Indian lending data, CIBIL data, or RBI-regulated production data.

PHASE 5 — STEP 1
"""

import logging
from typing import Any, Dict, Union

import numpy as np
import pandas as pd

from src.models.model_artifacts import ModelArtifacts, ArtifactError
from src.prediction.schemas import (
    ApplicantInput,
    InputValidationError,
    PredictionResult,
    UCI_ALL_FEATURES,
)
from src.prediction.risk_policy import RiskPolicy

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CreditRiskPredictor
# ---------------------------------------------------------------------------

class CreditRiskPredictor:
    """
    End-to-end production inference class for the UCI German Credit benchmark.

    Loads model artifacts once at initialization, then provides fast,
    stateless prediction on new applicant inputs.

    Usage:
        predictor = CreditRiskPredictor()

        result = predictor.predict({
            "status_existing_checking_account": "A11",
            "duration_in_months": 24,
            "credit_history": "A32",
            ... (all 20 UCI features)
        })

        print(result.estimated_pd)   # e.g. 0.23
        print(result.risk_score)     # e.g. 77.0
        print(result.risk_tier)      # e.g. "MEDIUM"
        print(result.decision)       # e.g. "MANUAL REVIEW"

    Thread Safety:
        The predictor does not maintain mutable state after initialization.
        It is safe to share a single instance across multiple requests.

    IMPORTANT:
        - No .fit() is called at any point during prediction.
        - Invalid input raises InputValidationError immediately.
        - Missing or corrupt artifacts raise ArtifactError immediately.
    """

    def __init__(self, policy: RiskPolicy = None) -> None:
        """
        Initialize the predictor by loading and validating all model artifacts.

        Args:
            policy (RiskPolicy, optional): Risk policy to use for tier and decision
                                           assignment. Defaults to the standard
                                           demonstration policy
                                           (LOW < 0.20, MEDIUM < 0.45, HIGH ≥ 0.45).

        Raises:
            ArtifactError: If any model artifact is missing, corrupt, or inconsistent.
        """
        logger.info("Initializing CreditRiskPredictor...")

        # Load policy
        self._policy: RiskPolicy = policy or RiskPolicy()

        # Validate and load artifacts
        self._validate_artifacts()
        self._preprocessor = ModelArtifacts.load_preprocessor()
        self._model        = ModelArtifacts.load_model()
        self._metadata     = ModelArtifacts.load_metadata()

        logger.info(
            "CreditRiskPredictor initialized. Model: %s | Dataset: %s",
            self._metadata.get("model_name", "unknown"),
            self._metadata.get("dataset", "unknown"),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        applicant: Union[Dict[str, Any], ApplicantInput],
    ) -> PredictionResult:
        """
        Generate a credit risk prediction for a single applicant.

        The prediction pipeline:
            1. Validates input (raises InputValidationError on any issue)
            2. Converts to DataFrame in UCI column order
            3. Transforms using the saved preprocessor (no .fit())
            4. Calls model.predict_proba() (no .fit())
            5. Applies risk policy
            6. Returns PredictionResult

        Args:
            applicant: Either:
                - A plain dict with all 20 UCI feature names as keys, or
                - An already-validated ApplicantInput instance.

        Returns:
            PredictionResult with estimated_pd, risk_score, risk_tier, decision.

        Raises:
            InputValidationError: If input is missing fields, has NaN/Inf,
                                  or uses invalid categorical codes.
            ArtifactError:        If a runtime artifact issue is detected.
        """
        # Step 1: Validate and normalize input
        applicant_input = self._normalize_input(applicant)

        # Step 2: Convert to DataFrame in training column order
        input_df = self._prepare_dataframe(applicant_input)

        # Step 3: Transform using saved preprocessor (NO .fit())
        input_processed = self._transform(input_df)

        # Step 4: Generate probability estimate (NO .fit())
        estimated_pd = self._predict_proba(input_processed)

        # Step 5: Apply risk policy
        risk_score = self._policy.compute_score(estimated_pd)
        risk_tier  = self._policy.assign_tier(estimated_pd)
        decision   = self._policy.assign_decision(risk_tier)

        # Step 6: Build structured result
        result = PredictionResult(
            estimated_pd=round(estimated_pd, 6),
            risk_score=risk_score,
            risk_tier=risk_tier,
            decision=decision,
            model_name=self._metadata.get("model_name", "Logistic Regression"),
            dataset=self._metadata.get("dataset", "UCI Statlog German Credit Data"),
        )

        logger.info(
            "Prediction complete: PD=%.4f | Tier=%s | Decision=%s",
            estimated_pd, risk_tier, decision,
        )

        return result

    @property
    def metadata(self) -> Dict[str, Any]:
        """Return the model metadata dictionary (read-only)."""
        return dict(self._metadata)

    @property
    def policy(self) -> RiskPolicy:
        """Return the current risk policy (read-only)."""
        return self._policy

    # ------------------------------------------------------------------
    # Private helpers — prediction path
    # ------------------------------------------------------------------

    def _normalize_input(
        self, applicant: Union[Dict[str, Any], ApplicantInput]
    ) -> ApplicantInput:
        """
        Convert applicant input to a validated ApplicantInput.

        Args:
            applicant: dict or ApplicantInput instance.

        Returns:
            Validated ApplicantInput.

        Raises:
            InputValidationError: If validation fails.
            TypeError: If the input type is not dict or ApplicantInput.
        """
        if isinstance(applicant, ApplicantInput):
            # Re-validate to be safe (validate() is idempotent)
            applicant.validate()
            return applicant
        elif isinstance(applicant, dict):
            return ApplicantInput.from_dict(applicant)
        else:
            raise TypeError(
                f"applicant must be a dict or ApplicantInput, "
                f"got {type(applicant).__name__}."
            )

    def _prepare_dataframe(self, applicant_input: ApplicantInput) -> pd.DataFrame:
        """
        Convert a validated ApplicantInput to a single-row DataFrame.

        The columns are ordered to exactly match the UCI training feature order
        that the ColumnTransformer was fitted on.

        Args:
            applicant_input: A validated ApplicantInput instance.

        Returns:
            pd.DataFrame with shape (1, 20), columns = UCI_ALL_FEATURES.
        """
        row = applicant_input.to_dict()
        df = pd.DataFrame([row], columns=UCI_ALL_FEATURES)
        return df

    def _transform(self, input_df: pd.DataFrame) -> np.ndarray:
        """
        Apply the saved preprocessor to the input DataFrame.

        IMPORTANT: This calls .transform() only — NEVER .fit() or .fit_transform().

        Args:
            input_df: Single-row DataFrame with 20 UCI feature columns.

        Returns:
            np.ndarray of shape (1, n_processed_features).

        Raises:
            ArtifactError: If the transformation fails unexpectedly.
        """
        try:
            processed = self._preprocessor.transform(input_df)
            return processed
        except Exception as exc:
            raise ArtifactError(
                f"Preprocessor.transform() failed: {exc}\n"
                "Check that input features match the UCI training schema."
            ) from exc

    def _predict_proba(self, input_processed: np.ndarray) -> float:
        """
        Run model inference and extract the Probability of Default.

        IMPORTANT: This calls .predict_proba() only — NEVER .fit().

        Args:
            input_processed: np.ndarray of shape (1, n_processed_features).

        Returns:
            float: Estimated Probability of Default (class 1 probability).

        Raises:
            ArtifactError: If model inference fails.
        """
        try:
            proba = self._model.predict_proba(input_processed)
            # proba shape: (1, 2) — [P(good=0), P(default=1)]
            estimated_pd: float = float(proba[0, 1])
            return estimated_pd
        except Exception as exc:
            raise ArtifactError(
                f"Model.predict_proba() failed: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Private helpers — initialization
    # ------------------------------------------------------------------

    def _validate_artifacts(self) -> None:
        """
        Run all artifact validation gates before loading.

        Raises:
            ArtifactError: If any gate fails.
        """
        ModelArtifacts.validate_all_artifacts()
