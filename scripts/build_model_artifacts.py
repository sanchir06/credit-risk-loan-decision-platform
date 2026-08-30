"""
scripts/build_model_artifacts.py
=================================

PURPOSE:
    Reproducible artifact generation script for Phase 5 — Step 1.

    Loads the UCI German Credit benchmark dataset, replicates the preprocessing
    methodology established in Phase 2/3 (notebooks 07 and 08), trains the
    selected production model, and saves all three artifacts to models/.

WHAT THIS SCRIPT DOES:
    1. Loads data/raw/german_credit_1000.csv
    2. Separates features (X) from target (y = default)
    3. Stratified 80/20 train/test split (random_state=42)
    4. Builds ColumnTransformer:
         StandardScaler for 7 numerical features
         OneHotEncoder(handle_unknown='ignore', sparse_output=False) for 13 categorical
    5. Fits the preprocessor on X_train ONLY (zero test leakage)
    6. Transforms X_train and X_test (transform only — no refit)
    7. Trains the selected production model on preprocessed X_train
    8. Saves preprocessing.joblib, credit_risk_model.joblib, model_metadata.json
    9. Prints a structured completion report

SELECTED MODEL: Logistic Regression
    Rationale (from notebooks/11_model_evaluation_explainability.ipynb):
        - Best Test ROC-AUC: 0.8040 (outperforms RF 0.7907, XGB 0.7798)
        - Best Brier Score:  0.1550 (best calibrated for PD estimation)
        - Most interpretable for credit underwriting demonstration
    Trade-off documented:
        Random Forest achieves higher recall (0.6833 vs 0.5333) but at the
        cost of weaker generalisation (ROC-AUC 0.7907) and worse calibration
        (Brier 0.1834). Logistic Regression is preferred for the production
        demonstration layer based on Phase 4 multi-criteria analysis.

LEAKAGE PREVENTION:
    - Preprocessor is fitted ONLY on X_train (800 records)
    - X_test is transformed, never used to fit or select models
    - No performance metrics are computed or reported here
      (those are documented in notebook 11, not re-derived here)

DATASET CONTEXT:
    UCI Statlog German Credit Data (Prof. Dr. Hans Hofmann, 1994)
    Source: UCI Machine Learning Repository
    This is a historical European credit-risk benchmark dataset.
    It is NOT Indian lending data, CIBIL data, or RBI-regulated production data.

USAGE:
    python scripts/build_model_artifacts.py

    Run from the project root directory.

PHASE 5 — STEP 1
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Path setup — allow running from project root
# ---------------------------------------------------------------------------

# Resolve project root (scripts/ is one level below root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Add project root to sys.path so src imports work
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.model_artifacts import ModelArtifacts, ArtifactError  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "german_credit_1000.csv"
TARGET_COLUMN = "default"
RANDOM_STATE = 42
TEST_SIZE = 0.20

# Feature lists — must match notebook 07 exactly
NUMERICAL_FEATURES = [
    "duration_in_months",
    "credit_amount",
    "installment_rate_pct_disposable_income",
    "present_residence_since",
    "age_years",
    "existing_credits_count",
    "people_liable_maintenance",
]

CATEGORICAL_FEATURES = [
    "status_existing_checking_account",
    "credit_history",
    "purpose",
    "savings_account_bonds",
    "present_employment_since",
    "personal_status_sex",
    "other_debtors_guarantors",
    "property",
    "other_installment_plans",
    "housing",
    "job",
    "telephone",
    "foreign_worker",
]

# Model configuration — matches Phase 3/4 notebook methodology
MODEL_NAME = "Logistic Regression"
MODEL_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Helper — print separator
# ---------------------------------------------------------------------------

def _sep(char: str = "=", width: int = 60) -> None:
    print(char * width)


# ---------------------------------------------------------------------------
# Main build function
# ---------------------------------------------------------------------------

def build_artifacts() -> None:
    """
    Execute the full Phase 5 Step 1 artifact build pipeline.

    Raises:
        FileNotFoundError: If the raw dataset is missing.
        ArtifactError:     If artifact saving fails.
        SystemExit:        On any unrecoverable error.
    """
    _sep()
    print("PHASE 5 - STEP 1 MODEL ARTIFACT BUILD")
    _sep()
    print()

    # ------------------------------------------------------------------
    # Step 1: Load dataset
    # ------------------------------------------------------------------
    print("Loading dataset...")

    if not DATA_PATH.exists():
        print(f"\n[ERROR] Dataset not found: {DATA_PATH}")
        print("Ensure data/raw/german_credit_1000.csv is present.")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)

    total_records = len(df)
    feature_count = len(df.columns) - 1  # exclude target

    print(f"  Dataset: UCI Statlog German Credit Data")
    print(f"  Total records: {total_records:,}")
    print(f"  Total columns: {len(df.columns)} ({feature_count} features + 1 target)")
    print(f"  Target: '{TARGET_COLUMN}' | 0=Good/Non-default, 1=Bad/Default")
    print(f"  Class distribution: {df[TARGET_COLUMN].value_counts().to_dict()}")
    print()

    # ------------------------------------------------------------------
    # Step 2: Separate X and y
    # ------------------------------------------------------------------
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # Validate expected features
    missing_numerical = [f for f in NUMERICAL_FEATURES if f not in X.columns]
    missing_categorical = [f for f in CATEGORICAL_FEATURES if f not in X.columns]
    if missing_numerical or missing_categorical:
        print(f"[ERROR] Missing expected columns:")
        for col in missing_numerical + missing_categorical:
            print(f"  - {col}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 3: Stratified 80/20 split (matches notebook 07 methodology)
    # ------------------------------------------------------------------
    print("Performing stratified 80/20 train/test split...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    n_train = len(X_train)
    n_test  = len(X_test)

    print(f"  Training records: {n_train:,} ({n_train/total_records:.0%})")
    print(f"  Testing records:  {n_test:,}  ({n_test/total_records:.0%})")
    print(f"  random_state: {RANDOM_STATE}, stratify: True")
    print()

    # ------------------------------------------------------------------
    # Step 4: Build ColumnTransformer (matches notebook 07 methodology)
    # ------------------------------------------------------------------
    print("Building preprocessing pipeline...")
    print(f"  Numerical features ({len(NUMERICAL_FEATURES)}): {NUMERICAL_FEATURES}")
    print(f"  Categorical features ({len(CATEGORICAL_FEATURES)}): {CATEGORICAL_FEATURES}")

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                NUMERICAL_FEATURES,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )

    # ------------------------------------------------------------------
    # Step 5: Fit preprocessor on X_train ONLY — zero test leakage
    # ------------------------------------------------------------------
    print()
    print("Fitting preprocessor on X_train only (zero leakage)...")

    preprocessor.fit(X_train)  # Fitted on training data ONLY

    # Step 6: Transform both splits
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed  = preprocessor.transform(X_test)   # transform only

    n_processed_features = X_train_processed.shape[1]
    print(f"  Processed feature count: {n_processed_features}")
    print(f"  X_train_processed shape: {X_train_processed.shape}")
    print(f"  X_test_processed shape:  {X_test_processed.shape}")
    print()

    # ------------------------------------------------------------------
    # Step 7: Train the selected production model on X_train
    # ------------------------------------------------------------------
    print(f"Training selected model: {MODEL_NAME}...")
    print("  (Phase 4 selection rationale: best Test ROC-AUC 0.8040, best Brier 0.1550)")
    print()

    model = LogisticRegression(
        random_state=RANDOM_STATE,
        max_iter=1000,
        solver="lbfgs",
    )

    model.fit(X_train_processed, y_train)  # Fitted on training data ONLY

    print(f"  Training completed.")
    print(f"  Model classes: {model.classes_.tolist()}")
    print()

    # ------------------------------------------------------------------
    # Step 8: Build metadata (dynamically derived from fitted objects)
    # ------------------------------------------------------------------
    print("Building model metadata...")

    training_date = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    metadata = {
        "model_name":              MODEL_NAME,
        "model_class":             type(model).__name__,
        "model_version":           MODEL_VERSION,
        "dataset":                 "UCI Statlog German Credit Data",
        "dataset_source":          "UCI Machine Learning Repository",
        "dataset_records":         int(total_records),
        "dataset_year":            1994,
        "feature_count":           int(feature_count),
        "numerical_feature_count": len(NUMERICAL_FEATURES),
        "categorical_feature_count": len(CATEGORICAL_FEATURES),
        "processed_feature_count": int(n_processed_features),
        "target":                  TARGET_COLUMN,
        "target_classes":          model.classes_.tolist(),
        "random_state":            RANDOM_STATE,
        "test_size":               TEST_SIZE,
        "training_records":        int(n_train),
        "test_records":            int(n_test),
        "training_date":           training_date,
        "risk_score_formula":      "(1 - estimated_pd) * 100",
        "pd_is_calibrated":        False,
        "pd_calibration_note":     (
            "Estimated PD is not formally calibrated. "
            "predict_proba() values are used directly as demonstration PD estimates."
        ),
        "purpose":                 "historical benchmark demonstration",
        "context_note": (
            "UCI Statlog German Credit Data is a historical European "
            "credit-risk benchmark (1994, Deutsche Mark). "
            "It is NOT Indian lending data, CIBIL data, or RBI-regulated data."
        ),
        "model_selection_rationale": (
            "Selected based on Phase 4 multi-criteria analysis "
            "(notebooks/11_model_evaluation_explainability.ipynb). "
            "Logistic Regression: Test ROC-AUC=0.8040 (best), Brier=0.1550 (best). "
            "Random Forest: Test ROC-AUC=0.7907, higher Recall=0.6833 but worse calibration. "
            "XGBoost: Test ROC-AUC=0.7798 (lowest). "
            "Logistic Regression selected for best generalisation and calibration."
        ),
        "phase":                   "Phase 5 Step 1",
        "model_solver":            model.solver,
        "model_max_iter":          model.max_iter,
        "model_converged":         bool(model.n_iter_[0] < model.max_iter),
        "model_n_iter":            int(model.n_iter_[0]),
    }

    # ------------------------------------------------------------------
    # Step 9: Save all artifacts
    # ------------------------------------------------------------------
    print("Saving artifacts...")

    try:
        ModelArtifacts.save_preprocessor(preprocessor)
        print("  Preprocessor: SAVED -> models/preprocessing.joblib")

        ModelArtifacts.save_model(model)
        print("  Model:        SAVED -> models/credit_risk_model.joblib")

        ModelArtifacts.save_metadata(metadata)
        print("  Metadata:     SAVED -> models/model_metadata.json")

    except ArtifactError as exc:
        print(f"\n[ERROR] Failed to save artifacts:\n{exc}")
        sys.exit(1)

    print()

    # ------------------------------------------------------------------
    # Completion report
    # ------------------------------------------------------------------
    _sep()
    print("PHASE 5 - STEP 1 MODEL ARTIFACT BUILD")
    _sep()
    print()
    print("Dataset:")
    print("  UCI Statlog German Credit Data")
    print()
    print(f"Training records:   {n_train:,}")
    print(f"Testing records:    {n_test:,}")
    print(f"Raw features:       {feature_count}")
    print(f"Processed features: {n_processed_features}")
    print()
    print(f"Selected model:     {MODEL_NAME} (v{MODEL_VERSION})")
    print(f"Solver:             {model.solver}")
    print(f"Converged:          {metadata['model_converged']} "
          f"(iterations: {metadata['model_n_iter']})")
    print()
    print("Preprocessor:       SAVED")
    print("Model:              SAVED")
    print("Metadata:           SAVED")
    print()
    _sep()
    print("PHASE 5 - STEP 1 ARTIFACT BUILD: PASSED")
    _sep()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    build_artifacts()
