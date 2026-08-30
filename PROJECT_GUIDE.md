# PROJECT GUIDE: Credit Risk & Loan Decision Intelligence Platform

A comprehensive technical and domain guide to the architecture, risk taxonomy, data pipelines, and decision intelligence workflow.

---

## 1. Executive Problem Statement & Context

In retail lending and digital micro-finance, institutions face an ongoing optimization trade-off:

$$\text{Maximize Interest Revenue (Approve Good Loans)} \iff \text{Minimize Capital Loss (Reject Defaults)}$$

- **Under-filtering Risk (False Positives)**: Approving an applicant who defaults causes direct financial loss and raises Non-Performing Assets (NPAs).
- **Over-filtering Risk (False Negatives)**: Rejecting a creditworthy applicant causes lost interest revenue and borrower exclusion.

This platform automates credit risk evaluation through machine learning while enforcing strict regulatory compliance (RBI guidelines, DPDP Act 2023) and domain underwriting logic.

---

## 2. Core Architectural Separation: Identity Risk vs. Credit Default Risk

A critical design principle of this platform is the strict separation between **Identity Verification** and **Credit Default Prediction**:

```
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                                APPLICATION ARCHITECTURE                                 │
├─────────────────────────────────────────────┬────────────────────────────────────────────┤
│ 1. IDENTITY & ELIGIBILITY RISK              │ 2. CREDIT DEFAULT RISK                     │
│    (src/verification/ & Product Workflow)   │    (src/credit/, src/features/ & ML)       │
├─────────────────────────────────────────────┼────────────────────────────────────────────┤
│ • PAN syntax validation & registry lookup   │ • Bureau credit score (e.g. CIBIL: 300–900)│
│ • Aadhaar / Biometric KYC verification      │ • Debt-to-Income (DTI) & FOIR ratios       │
│ • Mobile number format & OTP validation     │ • Delinquency & past default frequency     │
│ • Cross-document fuzzy identity matching    │ • Average bank balance & cashflow cushion  │
├─────────────────────────────────────────────┼────────────────────────────────────────────┤
│ Purpose: ELIGIBILITY GATE                   │ Purpose: UNDERWRITING & PROBABILITY OF DEF │
│ Decision: Approve / Reject / Manual Review  │ Output: Estimated Probability of Default   │
│ Gating: Runs BEFORE credit risk modeling    │ Evaluates: Willingness & Ability to Repay  │
└─────────────────────────────────────────────┴────────────────────────────────────────────┘
```

> **The Separation Rule**: A failed PAN or Aadhaar verification check means the applicant is unverified or potentially fraudulent — it is **NOT** a calibrated statistical predictor of loan repayment capacity. Identity checks gate the workflow at the product layer. The machine learning model is invoked only for verified applicants.

---

## 3. The Two Coordinated System Flows

```
============================================================================================
FLOW A: PRODUCT / APPLICATION FLOW (src/)
============================================================================================

    Customer Loan Application
              │
              ▼
    PAN / KYC / Mobile OTP Verification (src/verification/)
              │
              ▼
    Cross-Source Identity Consistency Checks (src/verification/identity_checks.py)
              │
              ▼
    Eligibility Gate ──► [FAIL / MANUAL KYC REVIEW] ──► Application Halted / Queued
              │ (PASS)
              ▼
    Financial Capability Analysis (src/financial/ - Income, Bank Statement, DTI/FOIR)
              │
              ▼
    Credit Bureau Profile (src/credit/ - CIBIL score, utilization, payment history)
              │
              ▼
    Feature Vector Extraction (src/features/feature_engineering.py)
              │
              ▼
    [Phase 3 ML Model Scoring & Probability of Default] ──► [Phase 5 Decision Engine]


============================================================================================
FLOW B: CURRENT MACHINE LEARNING PIPELINE (notebooks/ & data/)
============================================================================================

    Raw Credit Dataset (data/raw/credit_risk_50.csv)
              │
              ▼
    Phase 1: Data Understanding & Exploratory Analysis (notebooks/01_data_understanding.ipynb)
              │
              ▼
    Phase 2: Stratified Train / Test Split (80% Train [40 rows], 20% Test [10 rows])
              │
              ▼
    Phase 2: Preprocessing Pipeline (notebooks/02_data_preprocessing.ipynb)
              ├─ StandardScaler (13 numerical features, fitted strictly on X_train)
              └─ OneHotEncoder (employment_type, fitted strictly on X_train)
              │
              ▼
    Model-Ready Feature Matrices (X_train_processed: 40×16, X_test_processed: 10×16)
              │
              ▼
    Phase 3: Machine Learning Model Development & Decision Intelligence ✅ COMPLETE (notebooks/03, 04, 05)
              │
              ▼
    Phase 4: Model Evaluation, Explainability & Final Selection (notebook 11)
              |
              v
    Phase 5 Step 1: Production Inference Pipeline (COMPLETE)
        src/models/model_artifacts.py      -- Artifact management (save/load/validate)
        src/prediction/predictor.py        -- CreditRiskPredictor (no notebook required)
        src/prediction/schemas.py          -- UCI input schema & validation
        src/prediction/risk_policy.py      -- Risk tier/decision policy
        scripts/build_model_artifacts.py   -- Reproducible artifact build
        tests/test_prediction_pipeline.py  -- 15-gate, 44-test validation suite
              |
              v
    Phase 5 Step 2: FastAPI REST API (NOT STARTED)
              |
              v
    Phase 5 Step 3: Streamlit Dashboard (NOT STARTED)
```

---

## 4. Key Financial Domain Concepts

### 1. Probability of Default (PD)
The core output of the credit risk machine learning model: a continuous probability value $0.0 \le \text{PD} \le 1.0$ representing the likelihood that a borrower will fail to meet scheduled loan repayment obligations within a specified tenure.

### 2. Debt-to-Income (DTI) & FOIR
$$\text{DTI} = \frac{\text{Existing Monthly Debt EMI}}{\text{Monthly Income}}$$
$$\text{FOIR (Fixed Obligation to Income Ratio)} = \frac{\text{Existing EMI} + \text{Proposed New Loan EMI} + \text{Rent/Fixed Expenses}}{\text{Monthly Income}}$$
Lenders generally require $\text{FOIR} \le 50\%–60\%$ to ensure the borrower maintains an adequate disposable income buffer.

### 3. Credit Bureau Profile & CIBIL Score
In India, credit scores range from **300 to 900**:
- **750–900**: Excellent / Low Risk
- **700–749**: Good / Moderate Risk
- **650–699**: Fair / Elevated Risk
- **< 650**: High Risk / Subprime

### 4. Thin-File / New-to-Credit Borrowers
Borrowers lacking prior bureau credit history require alternative underwriting signals (bank statement average monthly balance, cashflow volatility, expense-to-income ratio, mandate bounce frequency) to establish repayment capacity.

---

## 5. Feature Engineering Taxonomy

Features in this platform are classified into three operational categories:

| Feature Category | Description | Examples |
|---|---|---|
| **1. Raw Features** | Direct applicant inputs collected at source | `monthly_income`, `loan_amount`, `employment_type` |
| **2. Calculated Features** | Derived mathematically from raw or bureau fields | `age` (from DOB), `debt_to_income_ratio`, `loan_to_income_ratio` |
| **3. Aggregated Features** | Multi-record transaction summaries from banking data | `average_bank_balance`, `avg_monthly_cashflow`, `bounce_count` |

---

## 6. Preprocessing & Data Leakage Prevention

### The Golden Rule of Preprocessing:
$$\text{Fit transformers strictly on } X_{\text{train}} \longrightarrow \text{Transform } X_{\text{train}} \text{ and } X_{\text{test}} \text{ using training-fitted parameters}$$

### Dual-Dataset Preprocessing Architectures:
1. **Synthetic Educational Dataset (`credit_risk_50.csv`)**:
   - 80% Train (`40` samples), 20% Test (`10` samples) using `stratify=y` and `random_state=42`.
   - `ColumnTransformer`: `StandardScaler` for 13 numerical columns, `OneHotEncoder` for `employment_type` (16 processed features).
2. **Real-World Historical Benchmark Dataset (`german_credit_1000.csv`)**:
   - 80% Train (`800` samples), 20% Test (`200` samples) using `stratify=y` and `random_state=42`.
   - `ColumnTransformer`: `StandardScaler` for 7 numerical columns, `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` for 13 categorical columns (61 processed features).
   - Zero-leakage transformation with 10 automated validation integrity gates.

---

## 7. Dual-Dataset Architecture & Multi-Model Benchmarking

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                DUAL DATASET ARCHITECTURE                                    │
├──────────────────────────────────────────────┬──────────────────────────────────────────────┤
│ 1. SYNTHETIC EDUCATIONAL PIPELINE (50 rows)  │ 2. REAL-WORLD BENCHMARK PIPELINE (1,000 rows)│
├──────────────────────────────────────────────┼──────────────────────────────────────────────┤
│ • `01_data_understanding.ipynb`              │ • `06_real_data_understanding.ipynb`         │
│ • `02_data_preprocessing.ipynb`              │ • `07_real_data_preprocessing.ipynb`         │
│ • `03_model_development.ipynb` (Baseline LR) │ • `08_real_data_model_development.ipynb`    │
│ • `04_tree_based_models.ipynb` (RF & XGB)    │ • `09_real_data_risk_scoring.ipynb`          │
│ • `05_risk_scoring_and_decision.ipynb`       │ • `10_dataset_comparison.ipynb`              │
├──────────────────────────────────────────────┼──────────────────────────────────────────────┤
│ Purpose: Architecture & UI flow prototyping  │ Purpose: Statistical ML & Cross-Validation   │
└──────────────────────────────────────────────┴──────────────────────────────────────────────┘
```
- **Consolidated Phase 4 Benchmark & Explainability**: `notebooks/11_model_evaluation_explainability.ipynb` evaluates and benchmarks both pipelines side-by-side.

---

## 8. Project Roadmap & Implementation Milestones

```
Phase 1: Data Understanding & EDA             COMPLETE (notebooks/01 & 06)
Phase 2: Data Preprocessing & Validation      COMPLETE (notebooks/02 & 07)
Phase 3: Credit Risk ML Modeling & Scores     COMPLETE (notebooks/03, 04, 05, 08, 09)
Comparative: Dataset Comparison & Governance  COMPLETE (notebooks/10)
Phase 4: Model Evaluation & Explainability    COMPLETE (notebooks/11 - Step 1 & Step 2)
Phase 5 Step 1: Production Inference Layer    COMPLETE (src/models/, src/prediction/, scripts/, tests/)
Phase 5 Step 2: FastAPI REST API              COMPLETE (api/main.py, api/schemas.py, tests/test_api.py)
Phase 5 Step 3: Streamlit Dashboard           COMPLETE (streamlit_app.py, tests/test_streamlit_app.py)
```

### Production Architecture (Phase 5 Step 1–3)
```
User / Analyst
      ↓
Streamlit UI (streamlit_app.py)
      ↓ HTTP JSON (POST /predict)
FastAPI REST API (api/main.py)
      ↓ Pydantic validation (api/schemas.py)
CreditRiskPredictor (src/prediction/predictor.py)
      ↓ Domain schema & range checks (src/prediction/schemas.py)
Saved ColumnTransformer (.transform() only, models/preprocessing.joblib)
      ↓ 61 one-hot encoded features
Saved Logistic Regression (.predict_proba() only, models/credit_risk_model.joblib)
      ↓ Estimated PD (Probability of Default)
Risk Policy (src/prediction/risk_policy.py)
      ↓ Score: (1 - PD) * 100, Tier: LOW/MEDIUM/HIGH, Decision: APPROVE/REVIEW/REJECT
FastAPI PredictResponse
      ↓ HTTP 200 JSON
Streamlit Result Panel (KPI cards, badge styling, attribution disclaimer)
```

---

## 9. Summary Governance Verification

- [x] Canonical raw synthetic dataset (`data/raw/credit_risk_50.csv`) verified intact.
- [x] Canonical raw benchmark dataset (`data/raw/german_credit_1000.csv`) verified (1,000 rows, 21 columns).
- [x] Zero-leakage preprocessing pipelines certified with automated assertion suites.
- [x] Both 50-row synthetic and 1,000-row benchmark notebook suites executed cleanly with 0 errors.
- [x] Phase 4 Steps 1 & 2 evaluation, SHAP explainability, and multi-criteria model selection executed and verified (notebook 11).
- [x] Phase 5 Step 1 production inference pipeline implemented and verified (44/44 tests).
- [x] Phase 5 Step 2 FastAPI REST API implemented and verified (47/47 tests).
- [x] Phase 5 Step 3 Streamlit dashboard implemented and verified (38/38 tests, 129/129 total).
- [x] Clean repository structure audited and verified.


