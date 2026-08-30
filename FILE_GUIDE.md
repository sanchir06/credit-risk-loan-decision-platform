# FILE GUIDE: Credit Risk & Loan Decision Intelligence Platform

A comprehensive file catalog documenting every active file in the repository: its functional purpose, implementation status, and role in the platform.

---

## 1. Root Configuration & Project Documentation

### `README.md`
- **Purpose**: Primary repository entrypoint presenting the project executive overview, dual-dataset architecture, two-flow system design, implementation milestones, and disclaimers.
- **Phase**: Ongoing / Phase 0–8
- **Role**: Executive portfolio documentation and architectural reference.

### `PROJECT_GUIDE.md`
- **Purpose**: Deep domain and technical reference explaining credit risk underwriting theory, identity risk separation, FOIR/DTI mathematics, feature taxonomy, and both the synthetic prototyping and real-data benchmark ML pipelines.
- **Phase**: Ongoing / Phase 0–8
- **Role**: Master technical companion guide for architecture and system design.

### `FILE_GUIDE.md` *(This File)*
- **Purpose**: Complete directory blueprint explaining the exact role of every file and folder in the workspace.
- **Phase**: Ongoing / Phase 0–8
- **Role**: Structural catalog and codebase navigation index.

### `requirements.txt`
- **Purpose**: Standard Python dependency file pinning all libraries required to run data processing, exploratory analysis, feature preprocessing, and machine learning models.
- **Phase**: Phase 1, 2, 3 & 4 Active (`pandas`, `numpy`, `scikit-learn`, `xgboost`, `shap`, `matplotlib`, `seaborn`, `jupyterlab`, `ipykernel`).
- **Role**: Environment reproducibility and dependency management.

### `.gitignore`
- **Purpose**: Version control exclusion rules preventing virtual environments (`.venv`), Python cache directories (`__pycache__`), temporary checkpoints, and large data files from entering Git.
- **Phase**: Active
- **Role**: Repository hygiene and VCS configuration.

---

## 2. `data/` — Data Storage Layer

### `data/raw/credit_risk_50.csv`
- **Purpose**: Canonical raw synthetic dataset containing 50 applicant records across 16 columns (1 identifier, 14 predictive features, and target `default`). Modeled on Indian retail lending domain (INR ₹, CIBIL 605–800).
- **Phase**: Phase 1, 2 & 3 Active
- **Role**: Prototyping and architecture validation dataset for Flow A & Flow B.

### `data/raw/german_credit_1000.csv`
- **Purpose**: Canonical real-world historical credit-risk benchmark dataset from the UCI Machine Learning Repository (Statlog German Credit Data, 1994, Prof. Dr. Hans Hofmann). Contains 1,000 records across 21 columns (20 predictive features + binary target `default`).
- **Phase**: Real-Data Benchmark Pipeline Active
- **Role**: Statistical ML benchmark, cross-validation, and decision intelligence dataset.

### `data/raw/README.md`
- **Purpose**: Raw data governance guide detailing the dual-dataset architecture, official UCI provenance, feature dictionaries, and the immutability principle.
- **Phase**: Active
- **Role**: Data dictionary, provenance, and governance reference.

### `data/processed/README.md`
- **Purpose**: Documents the preprocessing transformation architecture, stratified 80/20 train/test split, `ColumnTransformer` specification, and leakage prevention protocol.
- **Phase**: Active
- **Role**: Specifications for model-ready feature matrices (`X_train_processed`, `X_test_processed`).

### `data/synthetic/README.md`
- **Purpose**: Documents the data contract and schema for multi-field synthetic applicant fixtures used to test Flow A (the product/identity simulation flow and future UI dashboard).
- **Phase**: Planned for Flow A integration
- **Role**: Fixture specification for application workflow simulation.

---

## 3. `docs/` — Regulatory & Compliance Layer

### `docs/INDIA_DATA_PRIVACY.md`
- **Purpose**: In-depth regulatory compliance guide covering the Digital Personal Data Protection (DPDP) Act 2023, RBI Digital Lending Guidelines, and strict data minimization/masking principles.
- **Phase**: Active
- **Role**: Privacy, regulatory, and ethical underwriting standards.

---

## 4. `notebooks/` — Interactive Research & Preprocessing Layer

### A. Synthetic Educational Development Pipeline (50 Records)

#### `notebooks/01_data_understanding.ipynb`
- **Purpose**: Phase 1 exploratory data analysis notebook for the 50-row synthetic dataset. Ingests raw data, verifies distributions, inspects correlations, and establishes domain baseline understanding.
- **Phase**: Phase 1 ✅ **COMPLETE**
- **Role**: Exploratory analysis and data understanding artifact (Synthetic Pipeline).

#### `notebooks/02_data_preprocessing.ipynb`
- **Purpose**: Phase 2 data preprocessing and validation notebook for the 50-row synthetic dataset. Executes stratified train/test split, fits `ColumnTransformer` (`StandardScaler` + `OneHotEncoder`) strictly on training data, validates zero-leakage, and runs 10 automated model-ready integrity gates.
- **Phase**: Phase 2 ✅ **COMPLETE**
- **Role**: Preprocessing, feature matrix generation, and validation artifact (Synthetic Pipeline).

#### `notebooks/03_model_development.ipynb`
- **Purpose**: Phase 3 Step 1 baseline model development notebook. Replicates Phase 2 transformation pipeline, fits Logistic Regression baseline on training data with strict test isolation, generates estimated Probability of Default (PD), evaluates baseline metrics, and executes 10 automated validation gates.
- **Phase**: Phase 3 Step 1 ✅ **COMPLETE**
- **Role**: Baseline model benchmark and estimated PD generation artifact (Synthetic Pipeline).

#### `notebooks/04_tree_based_models.ipynb`
- **Purpose**: Phase 3 Step 2 tree-based models & benchmark comparison notebook. Trains Random Forest and XGBoost classifiers alongside the Logistic Regression baseline, conducts Stratified 5-Fold Cross-Validation on training data, benchmarks performance on the unseen test set, inspects model-derived feature importances (non-causal), and executes 12 automated validation gates.
- **Phase**: Phase 3 Step 2 ✅ **COMPLETE**
- **Role**: Non-linear tree model development, cross-validation, and multi-model benchmark artifact (Synthetic Pipeline).

#### `notebooks/05_risk_scoring_and_decision.ipynb`
- **Purpose**: Phase 3 Step 3 risk scoring and loan decision layer notebook. Derives transparent demonstration risk scores (0–100) from estimated PD, maps scores into named policy risk tiers (LOW, MEDIUM, HIGH), executes automated multi-tier loan decision policies (`APPROVE`, `MANUAL REVIEW`, `REJECT`), audits identity vs. credit risk separation, and executes 12 automated validation gates.
- **Phase**: Phase 3 Step 3 ✅ **COMPLETE**
- **Role**: Operational decision layer, risk score derivation, and policy mapping artifact (Synthetic Pipeline).

### B. Real-World Benchmark Dataset Pipeline (1,000 Records — UCI Statlog German Credit Data)

#### `notebooks/06_real_data_understanding.ipynb`
- **Purpose**: Phase 1-style 20-section Exploratory Data Analysis on the 1,000-record UCI benchmark dataset. Dynamically inspects shapes, dtypes, missing values, duplicates, target distribution (70/30), descriptive statistics, categorical distributions, histograms, outlier analysis, feature-vs-target relationships, correlation matrix, data leakage audit, and executes 10 automated validation gates.
- **Phase**: Real Benchmark ✅ **COMPLETE**
- **Role**: Exploratory analysis and data understanding artifact (Real-World Benchmark).

#### `notebooks/07_real_data_preprocessing.ipynb`
- **Purpose**: Phase 2-style preprocessing pipeline for the UCI dataset. Dynamically detects numerical and categorical features, executes stratified 80/20 split (800 train / 200 test), fits `ColumnTransformer` (`StandardScaler` + `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`) strictly on training data, generates 61 processed features with zero leakage, and executes 10 automated validation gates.
- **Phase**: Real Benchmark ✅ **COMPLETE**
- **Role**: Preprocessing, feature matrix generation, and validation artifact (Real-World Benchmark).

#### `notebooks/08_real_data_model_development.ipynb`
- **Purpose**: Phase 3-style multi-model benchmark on the UCI dataset. Trains Logistic Regression, Random Forest, and XGBoost classifiers. Evaluates via 5-Fold Stratified Cross-Validation on training data as the primary selection evidence, evaluates candidate models on the untouched holdout test set (200 records), generates ROC/PR curves, feature importances, and executes 10 automated validation gates.
- **Phase**: Real Benchmark ✅ **COMPLETE**
- **Role**: Multi-model training, cross-validation, and benchmarking artifact (Real-World Benchmark).

#### `notebooks/09_real_data_risk_scoring.ipynb`
- **Purpose**: Phase 3-style operational risk scoring and loan decision layer for the UCI dataset. Generates estimated Probability of Default (PD), derives internal demonstration risk scores (`risk_score = (1 - estimated_pd) * 100`), maps to configurable demo policy risk tiers (`LOW`, `MEDIUM`, `HIGH`), simulates decision policies (`APPROVE`, `MANUAL REVIEW`, `REJECT`), audits decision quality, and executes 10 automated validation gates.
- **Phase**: Real Benchmark ✅ **COMPLETE**
- **Role**: Operational decision layer, risk score derivation, and policy mapping artifact (Real-World Benchmark).

#### `notebooks/10_dataset_comparison.ipynb`
- **Purpose**: Comprehensive comparative analysis between the 50-row synthetic educational dataset and the 1,000-row UCI German Credit benchmark dataset. Compares record counts, feature schemas, target distributions, numerical/categorical structure, statistical power, modeling utility, and verifies strict dataset isolation with 10 automated validation gates.
- **Phase**: Real Benchmark ✅ **COMPLETE**
- **Role**: Comparative architecture and dataset governance artifact.

### C. Consolidated Phase 4 Model Evaluation, Explainability & Final Selection

#### `notebooks/11_model_evaluation_explainability.ipynb`
- **Purpose**: Consolidated Phase 4 (Step 1 & Step 2) evaluation, calibration, analytical threshold exploration, SHAP explainability (global beeswarm & local waterfall reason codes), risk portfolio segmentation (`LOW`, `MEDIUM`, `HIGH` tiers), multi-criteria model selection, and business tradeoff analysis. Executes 16 automated validation gates across both the 50-row synthetic pipeline and 1,000-row UCI benchmark.
- **Phase**: Phase 4 Steps 1 & 2 ✅ **COMPLETE**
- **Role**: Master model evaluation, SHAP interpretability, portfolio segmentation, and final model recommendation artifact.

---

## 5. `src/` — Modular Source Code Architecture

### `src/README.md`
- **Purpose**: Architectural guide to the 5 subpackages in `src/`, module responsibilities, and layer-to-layer data flows.
- **Phase**: Active
- **Role**: Source architecture reference.

### A. `src/application/`
- **`src/application/__init__.py`**: Package initializer making the folder importable.
- **`src/application/applicant.py`**: Central domain entity defining the `LoanApplicant` dataclass (encapsulates applicant personal info, requested loan parameters, verification statuses, financial metrics, and bureau data).

### B. `src/verification/` *(Identity Risk & Eligibility Layer)*
- **`src/verification/__init__.py`**: Package initializer.
- **`src/verification/pan_verification.py`**: Validates standard Indian PAN regex syntax (`AAAAA9999A`) and simulates mock registry verification.
- **`src/verification/aadhaar_kyc.py`**: Simulates UIDAI Aadhaar biometric KYC verification and format checks.
- **`src/verification/mobile_verification.py`**: Validates 10-digit Indian mobile numbers and simulates OTP verification.
- **`src/verification/identity_checks.py`**: Implements cross-source fuzzy Levenshtein name matching and DOB consistency checks.

### C. `src/financial/` *(Financial Capability Layer)*
- **`src/financial/__init__.py`**: Package initializer.
- **`src/financial/income_verification.py`**: Compares customer-declared monthly income against detected bank salary credits.
- **`src/financial/bank_statement_analyzer.py`**: Analyzes multi-month bank transactions to extract average balances, net cashflow, negative balance frequency, and cheque/mandate bounces.
- **`src/financial/debt_analyzer.py`**: Calculates debt burden metrics including Debt-to-Income (DTI) and Loan-to-Income (LTI) ratios.

### D. `src/credit/` *(Credit Bureau Layer)*
- **`src/credit/__init__.py`**: Package initializer.
- **`src/credit/credit_profile.py`**: Simulates bureau credit profile retrieval (CIBIL score 300–900, credit utilization ratio, past delinquency counts, previous default history, and recent inquiries).

### E. `src/features/` *(Feature Vector Extraction Layer)*
- **`src/features/__init__.py`**: Package initializer.
- **`src/features/feature_engineering.py`**: Implements `calculate_age()` and `build_feature_vector()`, extracting credit risk features from `LoanApplicant` while strictly excluding identity verification flags.

### F. `src/models/` *(Model Artifact Management — Phase 5 Step 1)*
- **`src/models/__init__.py`**: Package initializer exposing `ModelArtifacts` and `ArtifactError`.
- **`src/models/model_artifacts.py`**: Centralized artifact management module. Defines canonical paths (`models/preprocessing.joblib`, `models/credit_risk_model.joblib`, `models/model_metadata.json`), provides `save_preprocessor()`, `save_model()`, `save_metadata()`, `load_preprocessor()`, `load_model()`, `load_metadata()`, and `validate_all_artifacts()`. Uses `joblib` and `pathlib` throughout.

### G. `src/prediction/` *(Production Inference Pipeline — Phase 5 Step 1)*
- **`src/prediction/__init__.py`**: Package initializer exposing `CreditRiskPredictor`, `ApplicantInput`, `PredictionResult`, `RiskPolicy`, and `InputValidationError`.
- **`src/prediction/schemas.py`**: UCI benchmark applicant input schema. Defines all 20 UCI German Credit feature names, categorical whitelists, numerical ranges, `ApplicantInput` dataclass with `validate()` and `from_dict()`, and `PredictionResult` output dataclass. Rejects NaN, Inf, invalid categoricals, and missing fields with human-readable errors.
- **`src/prediction/risk_policy.py`**: Demonstration risk policy. Named constants (`TIER_LOW_MAX_PD=0.20`, `TIER_MEDIUM_MAX_PD=0.45`), frozen `RiskPolicy` dataclass with `compute_score()`, `assign_tier()`, `assign_decision()`. Documented explicitly as demonstration policy only — not RBI/regulatory thresholds.
- **`src/prediction/predictor.py`**: `CreditRiskPredictor` class. End-to-end inference: load artifacts, validate input, transform via preprocessor (no `.fit()`), `predict_proba()`, apply risk policy, return `PredictionResult`. Zero `.fit()` on prediction path.

---

## 6. `models/` — Production Model Artifacts *(Phase 5 Step 1)*

### `models/preprocessing.joblib`
- **Purpose**: Fitted `ColumnTransformer` (7 numerical → `StandardScaler`, 13 categorical → `OneHotEncoder`). Fitted on `X_train` (800 records) only. Outputs 61 processed features.
- **Phase**: Phase 5 Step 1 ✅ **COMPLETE**
- **Role**: Saved preprocessor loaded by `CreditRiskPredictor` at inference time.

### `models/credit_risk_model.joblib`
- **Purpose**: Fitted `LogisticRegression(random_state=42, max_iter=1000, solver='lbfgs')`. Selected from Phase 4 evidence: Test ROC-AUC=0.8040 (best), Brier=0.1550 (best). Accepts 61-feature processed input.
- **Phase**: Phase 5 Step 1 ✅ **COMPLETE**
- **Role**: Production inference model loaded by `CreditRiskPredictor`.

### `models/model_metadata.json`
- **Purpose**: Human-readable JSON document containing dataset context, model configuration, training date, feature counts, risk score formula, calibration notice, model selection rationale, and phase metadata. Dynamically derived from fitted objects during build.
- **Phase**: Phase 5 Step 1 ✅ **COMPLETE**
- **Role**: Model provenance and audit trail.

---

## 7. `scripts/` — Executable Scripts *(Phase 5 Step 1)*

### `scripts/build_model_artifacts.py`
- **Purpose**: Reproducible artifact generation script. Loads `data/raw/german_credit_1000.csv`, performs stratified 80/20 split (random_state=42), fits `ColumnTransformer` on `X_train` only, trains Logistic Regression on processed `X_train`, saves all three artifacts. Enforces zero test leakage. Prints structured completion report.
- **Phase**: Phase 5 Step 1 ✅ **COMPLETE**
- **Role**: One-command artifact regeneration. Run: `python scripts/build_model_artifacts.py`

---

## 8. `examples/` — Demonstration Scripts *(Phase 5 Step 1)*

### `examples/run_prediction.py`
- **Purpose**: Demonstrates the production inference pipeline using one clearly synthetic and fictional applicant. Shows estimated PD, risk score, risk tier, and decision. Includes explicit disclaimer that this is not an Indian banking decision.
- **Phase**: Phase 5 Step 1 ✅ **COMPLETE**
- **Role**: Usage example. Run: `python examples/run_prediction.py`

---

## 9. `tests/` — Automated Test Suites

### `tests/test_prediction_pipeline.py`
- **Purpose**: 15-gate, 44-test validation suite for the production inference pipeline. Covers: artifact existence (Gates 1–3), loading (4–5), `predict_proba` support (6), feature dimension compatibility (7), end-to-end prediction (8), PD bounds (9), risk score bounds (10), valid tier (11), valid decision (12), invalid input rejection (13), NaN/Inf rejection (14), and no-refit enforcement via monkey-patching (15).
- **Phase**: Phase 5 Step 1 ✅ **COMPLETE** (44/44 passing)
- **Role**: Continuous validation. Run: `pytest tests/test_prediction_pipeline.py -v`

### `tests/test_api.py`
- **Purpose**: 18-gate, 47-test validation suite for the Phase 5 Step 2 FastAPI REST API. Uses FastAPI `TestClient` (no live server needed). Covers: app import (Gate 1), `/health` response (2), `model_loaded` dynamic detection (3), `/model-info` response (4), metadata correctness (5), valid prediction (6), PD bounds (7), risk score bounds (8), valid tier (9), valid decision (10), missing field rejection (11), invalid categorical rejection (12), invalid numeric rejection (13), NaN/Inf rejection (14), no `model.fit()` during prediction (15), no `preprocessor.fit()` during prediction (16), no dataset loaded during prediction (17), OpenAPI schema generation (18).
- **Phase**: Phase 5 Step 2 ✅ **COMPLETE** (47/47 passing)
- **Role**: API continuous validation. Run: `pytest tests/test_api.py -v`

### `tests/test_streamlit_app.py`
- **Purpose**: 16-gate, 38-test validation suite for the Phase 5 Step 3 Streamlit dashboard. Uses mocked `streamlit` and mocked `httpx` so tests run headlessly without a live server or browser session. Covers: module import (Gate 1), API URL config (2), `/health` integration (3), `/model-info` integration (4), `/predict` integration (5), 20-feature UCI payload generation (6-7), no training code in source (8), no CSV loading in source (9), HTTP 200/422/503 response handling (10-12), connection failure handling (13), prediction response processing (14), disclaimer presence (15), absence of Indian-specific fields (16).
- **Phase**: Phase 5 Step 3 ✅ **COMPLETE** (38/38 passing; 129/129 total across all test suites)
- **Role**: UI/client continuous validation. Run: `pytest tests/test_streamlit_app.py -v`

---

## 10. `api/` — FastAPI REST API *(Phase 5 Step 2)*

### `api/__init__.py`
- **Purpose**: Package initializer for the `api/` module. Marks the directory as an importable Python package.
- **Phase**: Phase 5 Step 2 ✅ **COMPLETE**
- **Role**: Package entry point.

### `api/schemas.py`
- **Purpose**: Pydantic request and response models for the FastAPI HTTP layer. `PredictRequest` mirrors the 20 UCI German Credit features with Pydantic `Field()` constraints for type coercion and OpenAPI documentation generation. `PredictResponse`, `HealthResponse`, `ModelInfoResponse`, and `ErrorResponse` define structured JSON output models. Categorical domain validation is delegated to `ApplicantInput.from_dict()` inside the predictor (no logic duplication).
- **Phase**: Phase 5 Step 2 ✅ **COMPLETE**
- **Role**: HTTP-layer schema definitions.

### `api/main.py`
- **Purpose**: FastAPI application instance and route definitions. Uses a lifespan context manager to load `CreditRiskPredictor` once at startup and store in `app.state`. Exposes three endpoints: `GET /health` (dynamic `model_loaded` check), `GET /model-info` (dynamic metadata from `predictor.metadata`), `POST /predict` (delegates entirely to `CreditRiskPredictor.predict()`). Custom exception handlers: `InputValidationError` → HTTP 422, `ArtifactError` → HTTP 503, catch-all → HTTP 500. No internal paths or stack traces exposed. No `.fit()` calls. No dataset loading. Full OpenAPI documentation via `/docs` and `/redoc`.
- **Phase**: Phase 5 Step 2 ✅ **COMPLETE**
- **Role**: API entry point. Run: `python -m uvicorn api.main:app --reload`

---

## 11. Updated `examples/` *(Phase 5 Step 1–3)*

### `examples/run_prediction.py`
- **Purpose**: Demonstrates the production inference pipeline using one clearly synthetic and fictional applicant. Shows estimated PD, risk score, risk tier, and decision. Includes explicit disclaimer that this is not an Indian banking decision.
- **Phase**: Phase 5 Step 1 ✅ **COMPLETE**
- **Role**: Direct predictor demo (no server needed). Run: `python examples/run_prediction.py`

### `examples/run_api_prediction.py`
- **Purpose**: Demonstrates calling the FastAPI REST API via `httpx`. First checks `GET /health` to verify the API is ready, then submits a synthetic applicant to `POST /predict` and prints estimated PD, risk score, risk tier, and decision. Includes explicit disclaimer. Requires the API to be running locally.
- **Phase**: Phase 5 Step 2 ✅ **COMPLETE**
- **Role**: HTTP client demo. Prerequisites: `python -m uvicorn api.main:app --reload` then `python examples/run_api_prediction.py`

### `examples/run_streamlit_demo.md`
- **Purpose**: Step-by-step user guide and architecture summary for launching and interacting with the Streamlit credit risk dashboard alongside the FastAPI backend.
- **Phase**: Phase 5 Step 3 ✅ **COMPLETE**
- **Role**: Dashboard user documentation.

---

## 12. `streamlit_app.py` — Streamlit Credit Risk Dashboard *(Phase 5 Step 3)*

### `streamlit_app.py`
- **Purpose**: Interactive, interview-friendly frontend dashboard for loan underwriting analysts. Functions strictly as a presentation/HTTP-client layer over the FastAPI REST API (`http://127.0.0.1:8000`). Renders a sidebar with dynamic API health check and model metadata, a structured 4-section form for the 20 UCI German Credit features, an "Assess Credit Risk" trigger, and a professional KPI result panel with risk tier badges, estimated PD, score calculation, attribution disclaimers, and error banners. Contains zero training code, zero CSV reading, and zero artifact loading.
- **Phase**: Phase 5 Step 3 ✅ **COMPLETE**
- **Role**: Primary user interface. Run: `streamlit run streamlit_app.py`

