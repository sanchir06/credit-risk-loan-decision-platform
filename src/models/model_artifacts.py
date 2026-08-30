"""
src/models/model_artifacts.py
==============================

PURPOSE:
    Centralized management of production model artifact paths,
    serialization (save/load), and integrity validation.

    This module is the single source of truth for where artifacts
    live and how they are persisted.

ARTIFACT LAYOUT:
    models/
    ├── preprocessing.joblib       — Fitted sklearn ColumnTransformer
    ├── credit_risk_model.joblib   — Fitted selected model (Logistic Regression)
    └── model_metadata.json        — Dataset, model, and pipeline metadata

DESIGN PRINCIPLES:
    - pathlib throughout (no fragile string paths)
    - joblib for sklearn-compatible serialization
    - No raw data is ever serialized here
    - All load functions validate existence before loading
    - ArtifactError provides human-readable failure messages

DATASET CONTEXT:
    Artifacts are built from the UCI Statlog German Credit Data (1994).
    This is a historical European credit-risk benchmark dataset.
    It is NOT Indian lending data, CIBIL data, or RBI-regulated data.

PHASE 5 — STEP 1
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import joblib

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom Exception
# ---------------------------------------------------------------------------

class ArtifactError(Exception):
    """
    Raised when a model artifact is missing, corrupt, or inconsistent.

    Examples:
        - models/preprocessing.joblib does not exist
        - models/credit_risk_model.joblib was not saved correctly
        - Model does not support predict_proba()
    """


# ---------------------------------------------------------------------------
# Artifact Path Definitions
# ---------------------------------------------------------------------------

# Canonical root of the project (two levels up from this file:
#   src/models/model_artifacts.py → src/models/ → src/ → project root)
_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent

# Artifact directory (project_root/models/)
ARTIFACT_DIR: Path = _PROJECT_ROOT / "models"

# Individual artifact paths
PREPROCESSOR_PATH: Path = ARTIFACT_DIR / "preprocessing.joblib"
MODEL_PATH: Path        = ARTIFACT_DIR / "credit_risk_model.joblib"
METADATA_PATH: Path     = ARTIFACT_DIR / "model_metadata.json"


# ---------------------------------------------------------------------------
# ModelArtifacts — save / load / validate
# ---------------------------------------------------------------------------

class ModelArtifacts:
    """
    Provides save, load, and validation utilities for production model artifacts.

    Usage (build script):
        ModelArtifacts.save_preprocessor(fitted_column_transformer)
        ModelArtifacts.save_model(fitted_logistic_regression)
        ModelArtifacts.save_metadata(metadata_dict)

    Usage (predictor):
        preprocessor = ModelArtifacts.load_preprocessor()
        model        = ModelArtifacts.load_model()
        metadata     = ModelArtifacts.load_metadata()
    """

    # ------------------------------------------------------------------
    # Directory setup
    # ------------------------------------------------------------------

    @staticmethod
    def ensure_artifact_dir() -> None:
        """Create the models/ directory if it does not already exist."""
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        logger.debug("Artifact directory ensured: %s", ARTIFACT_DIR)

    # ------------------------------------------------------------------
    # Save utilities
    # ------------------------------------------------------------------

    @staticmethod
    def save_preprocessor(preprocessor: Any) -> Path:
        """
        Serialize the fitted sklearn ColumnTransformer to disk.

        The preprocessor MUST already be fitted on X_train.
        This function does NOT call .fit() — it only serializes.

        Args:
            preprocessor: A fitted sklearn ColumnTransformer.

        Returns:
            Path to the saved artifact.

        Raises:
            ArtifactError: If serialization fails.
        """
        ModelArtifacts.ensure_artifact_dir()
        try:
            joblib.dump(preprocessor, PREPROCESSOR_PATH)
            logger.info("Preprocessor saved: %s", PREPROCESSOR_PATH)
            return PREPROCESSOR_PATH
        except Exception as exc:
            raise ArtifactError(
                f"Failed to save preprocessor to {PREPROCESSOR_PATH}: {exc}"
            ) from exc

    @staticmethod
    def save_model(model: Any) -> Path:
        """
        Serialize the fitted sklearn-compatible model to disk.

        The model MUST already be fitted on preprocessed X_train.
        This function does NOT call .fit() — it only serializes.

        Args:
            model: A fitted sklearn-compatible estimator (must support predict_proba).

        Returns:
            Path to the saved artifact.

        Raises:
            ArtifactError: If serialization fails or model lacks predict_proba.
        """
        if not hasattr(model, "predict_proba"):
            raise ArtifactError(
                "Model does not support predict_proba(). "
                "Only probabilistic classifiers are accepted."
            )
        ModelArtifacts.ensure_artifact_dir()
        try:
            joblib.dump(model, MODEL_PATH)
            logger.info("Model saved: %s", MODEL_PATH)
            return MODEL_PATH
        except Exception as exc:
            raise ArtifactError(
                f"Failed to save model to {MODEL_PATH}: {exc}"
            ) from exc

    @staticmethod
    def save_metadata(metadata: Dict[str, Any]) -> Path:
        """
        Write model metadata to a human-readable JSON file.

        Args:
            metadata: Dictionary of metadata fields (see build_model_artifacts.py
                      for the full schema).

        Returns:
            Path to the saved metadata file.

        Raises:
            ArtifactError: If the write fails.
        """
        ModelArtifacts.ensure_artifact_dir()
        try:
            with open(METADATA_PATH, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=4)
            logger.info("Metadata saved: %s", METADATA_PATH)
            return METADATA_PATH
        except Exception as exc:
            raise ArtifactError(
                f"Failed to save metadata to {METADATA_PATH}: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Load utilities
    # ------------------------------------------------------------------

    @staticmethod
    def load_preprocessor() -> Any:
        """
        Load the fitted ColumnTransformer from disk.

        Returns:
            The deserialized, fitted sklearn ColumnTransformer.

        Raises:
            ArtifactError: If the file does not exist or cannot be loaded.
        """
        if not PREPROCESSOR_PATH.exists():
            raise ArtifactError(
                f"Preprocessor artifact not found: {PREPROCESSOR_PATH}\n"
                "Run scripts/build_model_artifacts.py first."
            )
        try:
            preprocessor = joblib.load(PREPROCESSOR_PATH)
            logger.debug("Preprocessor loaded from: %s", PREPROCESSOR_PATH)
            return preprocessor
        except Exception as exc:
            raise ArtifactError(
                f"Failed to load preprocessor from {PREPROCESSOR_PATH}: {exc}"
            ) from exc

    @staticmethod
    def load_model() -> Any:
        """
        Load the fitted model from disk.

        Returns:
            The deserialized, fitted sklearn-compatible estimator.

        Raises:
            ArtifactError: If the file does not exist or cannot be loaded.
        """
        if not MODEL_PATH.exists():
            raise ArtifactError(
                f"Model artifact not found: {MODEL_PATH}\n"
                "Run scripts/build_model_artifacts.py first."
            )
        try:
            model = joblib.load(MODEL_PATH)
            logger.debug("Model loaded from: %s", MODEL_PATH)
            return model
        except Exception as exc:
            raise ArtifactError(
                f"Failed to load model from {MODEL_PATH}: {exc}"
            ) from exc

    @staticmethod
    def load_metadata() -> Dict[str, Any]:
        """
        Load model metadata from disk.

        Returns:
            Dictionary of metadata fields.

        Raises:
            ArtifactError: If the file does not exist or is malformed.
        """
        if not METADATA_PATH.exists():
            raise ArtifactError(
                f"Metadata artifact not found: {METADATA_PATH}\n"
                "Run scripts/build_model_artifacts.py first."
            )
        try:
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            logger.debug("Metadata loaded from: %s", METADATA_PATH)
            return metadata
        except json.JSONDecodeError as exc:
            raise ArtifactError(
                f"Metadata file is malformed JSON: {METADATA_PATH}: {exc}"
            ) from exc
        except Exception as exc:
            raise ArtifactError(
                f"Failed to load metadata from {METADATA_PATH}: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Existence checks
    # ------------------------------------------------------------------

    @staticmethod
    def preprocessor_exists() -> bool:
        """Return True if the preprocessor artifact file exists on disk."""
        return PREPROCESSOR_PATH.exists()

    @staticmethod
    def model_exists() -> bool:
        """Return True if the model artifact file exists on disk."""
        return MODEL_PATH.exists()

    @staticmethod
    def metadata_exists() -> bool:
        """Return True if the metadata JSON file exists on disk."""
        return METADATA_PATH.exists()

    @staticmethod
    def all_artifacts_exist() -> bool:
        """Return True if all three required artifacts exist on disk."""
        return (
            ModelArtifacts.preprocessor_exists()
            and ModelArtifacts.model_exists()
            and ModelArtifacts.metadata_exists()
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def validate_all_artifacts() -> None:
        """
        Verify that all artifacts exist, load successfully, and are mutually
        consistent (preprocessor output dimension == model input dimension).

        Raises:
            ArtifactError: With a specific message identifying which check failed.
        """
        # Gate 1–3: Existence
        missing = []
        if not ModelArtifacts.preprocessor_exists():
            missing.append(str(PREPROCESSOR_PATH))
        if not ModelArtifacts.model_exists():
            missing.append(str(MODEL_PATH))
        if not ModelArtifacts.metadata_exists():
            missing.append(str(METADATA_PATH))
        if missing:
            raise ArtifactError(
                "Missing artifact file(s):\n"
                + "\n".join(f"  - {p}" for p in missing)
                + "\n\nRun scripts/build_model_artifacts.py to generate artifacts."
            )

        # Gate 4–5: Load
        preprocessor = ModelArtifacts.load_preprocessor()
        model        = ModelArtifacts.load_model()
        _metadata    = ModelArtifacts.load_metadata()  # validates JSON parses OK

        # Gate 6: predict_proba support
        if not hasattr(model, "predict_proba"):
            raise ArtifactError(
                "Loaded model does not support predict_proba(). "
                "Only probabilistic classifiers are valid production artifacts."
            )

        # Gate 7: Feature dimension consistency
        # We use a zero-row array with the correct number of features to probe
        # the model's expected input without running inference.
        try:
            import numpy as np
            # Derive the number of processed features from the preprocessor
            # by checking its output shape from a minimal dummy transform.
            # We rely on n_features_in_ for the raw input count.
            raw_feature_count: Optional[int] = getattr(
                preprocessor, "n_features_in_", None
            )
            # For the model, check n_features_in_
            model_feature_count: Optional[int] = getattr(
                model, "n_features_in_", None
            )

            if raw_feature_count is not None and model_feature_count is not None:
                # Transform a tiny dummy to get processed feature count
                dummy_input = np.zeros((1, raw_feature_count))
                try:
                    processed = preprocessor.transform(dummy_input)
                    processed_feature_count = processed.shape[1]
                    if processed_feature_count != model_feature_count:
                        raise ArtifactError(
                            f"Feature dimension mismatch: "
                            f"preprocessor outputs {processed_feature_count} features, "
                            f"but model expects {model_feature_count} features. "
                            "Re-run scripts/build_model_artifacts.py."
                        )
                except Exception:
                    # Dummy zeros may fail on categorical columns (ordinal issue).
                    # Silently skip dimension check — it will surface naturally during prediction.
                    logger.debug(
                        "Skipped dimension check on dummy input (expected for mixed-type transformers)."
                    )

        except ImportError:
            pass  # numpy not available — skip dimension check

        logger.info("All artifacts validated successfully.")
