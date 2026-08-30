# India-Focused Credit Risk & Digital Lending Intelligence Platform

An end-to-end Machine Learning and Decision Intelligence platform modeling an Indian digital lending institution's credit underwriting workflow — from KYC/identity verification to probability of default (PD) estimation and risk decisioning.

---

## 1. Executive Summary & Problem Statement

Every day, commercial banks, Non-Banking Financial Companies (NBFCs), and digital fintech lenders in India receive thousands of loan applications. Manually evaluating every applicant's creditworthiness is slow, expensive, and prone to human inconsistency.

Getting the lending decision wrong has severe financial consequences:
- **Too lenient (Under-filtering risk)**: Approves high-risk borrowers who default, directly driving up Non-Performing Assets (NPAs) and capital loss.
- **Too strict (Over-filtering risk)**: Rejects solvent, creditworthy applicants, resulting in lost revenue and unfair financial exclusion — particularly critical in India, where millions of borrowers are **new-to-credit (thin-file)**.

This platform bridges domain underwriting expertise with machine learning to evaluate loan applications rapidly, consistently, and transparently.

---

## 2. Core Architectural Principle: Identity Risk ≠ Credit Default Risk

A foundational architectural separation is maintained throughout this platform:

```
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                                APPLICATION WORKFLOW                                      │
├─────────────────────────────────────────────┬────────────────────────────────────────────┤
│ 1. IDENTITY & ELIGIBILITY RISK              │ 2. CREDIT DEFAULT RISK (ML MODEL)          │
│    (src/verification/ & Product Workflow)   │    (src/credit/, src/features/ & ML)       │
├─────────────────────────────────────────────┼────────────────────────────────────────────┤
│ • PAN verification & structure checks       │ • Bureau credit score (CIBIL / Experian)   │
│ • Aadhaar / Biometric KYC status            │ • Debt-to-Income (DTI) & FOIR ratios       │
│ • Mobile number & OTP validation            │ • Historical default & delinquency count   │
│ • Cross-document identity consistency       │ • Banking balance & cashflow stability     │
├─────────────────────────────────────────────┼────────────────────────────────────────────┤
│ Purpose: ELIGIBILITY GATE                   │ Purpose: UNDERWRITING & RISK PRICING       │
│ Decision: Reject fraud / Manual KYC review  │ Output: Estimated Probability of Default   │
│ Gating: Evaluated BEFORE credit scoring     │ Evaluates: Financial capacity & solvency   │
└─────────────────────────────────────────────┴────────────────────────────────────────────┘
```

> **Why this matters:** A failed PAN or KYC check indicates identity mismatch or fraud risk, not statistical credit default propensity. Mixing identity flags directly into a credit-risk ML model corrupts probability calibration. Identity risk gates the application *before* the ML model is invoked.

---

## 3. Two Coordinated System Flows

### Flow A: Product / Application Simulation Flow
Models the complete customer application lifecycle:
1. **Loan Application Submission**: Applicant details collected into domain entity (`LoanApplicant`).
2. **Identity Verification Layer**:
   - `pan_verification.py`: Validates PAN syntax and mock registry status.
   - `aadhaar_kyc.py`: Simulates biometric/UIDAI KYC verification.
   - `mobile_verification.py`: Validates 10-digit Indian mobile and OTP verification.
   - `identity_checks.py`: Performs cross-source fuzzy name and DOB consistency checks.
3. **Financial Analysis Layer**:
   - `income_verification.py`: Analyzes declared vs. detected income variance.
   - `bank_statement_analyzer.py`: Extracts average balances, monthly cashflows, and cheque/mandate bounces.
   - `debt_analyzer.py`: Computes Debt-to-Income (DTI) and Loan-to-Income (LTI) ratios.
4. **Credit Bureau Layer**:
   - `credit_profile.py`: Simulates CIBIL-style bureau records (credit score, active loans, late payments).

### Flow B: Credit Risk Machine Learning Pipeline
Takes validated financial and credit profiles to predict default:
```
Raw Dataset (data/raw/credit_risk_50.csv)
       │
       ▼
Phase 1: Data Understanding & EDA (notebooks/01_data_understanding.ipynb)
       │
       ▼
Phase 2: Stratified Train/Test Split (80% Train [40], 20% Test [10])
       │
       ▼
Phase 2: Preprocessing (ColumnTransformer: StandardScaler + OneHotEncoder)
       │
       ▼
Model-Ready Feature Matrices (X_train_processed, X_test_processed)
       │
       ▼
Phase 3: Machine Learning Model Development (COMPLETE)
```

---

## 4. Dual-Dataset Architecture & Data Governance

The platform maintains two distinct datasets, each serving a specific architectural and analytical objective:

```
┌──────────────────────────────────────────────┬──────────────────────────────────────────────┐
│ 1. SYNTHETIC EDUCATIONAL DATASET             │ 2. REAL-WORLD HISTORICAL BENCHMARK DATASET   │
│    `data/raw/credit_risk_50.csv`             │    `data/raw/german_credit_1000.csv`         │
├──────────────────────────────────────────────┼──────────────────────────────────────────────┤
│ • 50 synthetic applicant observations        │ • 1,000 historical bank loan records         │
│ • Indian lending domain schema (INR ₹, CIBIL)│ • UCI Statlog German Credit Data (1994, DM)  │
│ • Integrated with Flow A (PAN/KYC/Bureau)    │ • Decoupled benchmark for statistical ML     │
│ • Prototyping end-to-end platform workflow   │ • 5-Fold Cross-Validation & discrimination   │
└──────────────────────────────────────────────┴──────────────────────────────────────────────┘
```

### Dataset 1: Synthetic Educational Development Dataset (`credit_risk_50.csv`)
- **Record Count**: 50 synthetic loan applicant observations.
- **Column Count**: 16 columns (1 identifier, 14 predictive features, 1 target).
- **Target Variable**: `default` (Binary: `0` = Repaid, `1` = Defaulted | 62.0% Class 0, 38.0% Class 1).
- **Features in ML Matrix**: 13 numerical (`age`, `monthly_income`, `loan_amount`, `loan_term_months`, `existing_emi`, `credit_score`, `credit_utilization_pct`, `late_payment_count`, `previous_defaults`, `employment_years`, `average_bank_balance`, `monthly_expenses`, `debt_to_income_ratio`) + 1 categorical (`employment_type`).
- **Role**: End-to-end simulation of Flow A (application/identity checks) and Flow B (ML pipeline).

### Dataset 2: Real-World Historical Benchmark Dataset (`german_credit_1000.csv`)
- **Source**: UCI Machine Learning Repository (Statlog German Credit Data, Prof. Dr. Hans Hofmann, 1994).
- **Record Count**: 1,000 historical bank loan records.
- **Column Count**: 21 columns (20 predictive features + 1 binary target `default`).
- **Target Variable**: `default` (Binary: `0` = Good / Non-default [70.0%], `1` = Bad / Default [30.0%]).
- **Features in ML Matrix**: 7 numerical features + 13 categorical features (expanding to 61 processed features via `ColumnTransformer`).
- **Role**: Real-world benchmark for algorithm comparison, 5-Fold Stratified Cross-Validation, and decision policy stress-testing.
- **Context Note**: Historical European banking records in Deutsche Mark. It is **NOT** Indian lending data, production banking data, or representative of Indian borrowers.

> **Dataset Boundary Rule**: The synthetic and real-world benchmark datasets are maintained in strict isolation. They represent different currencies, economic systems, and schemas, and are **never combined or merged into a single training dataset**.

---

## 5. Current Implementation Status vs. Future Roadmap

| Phase | Milestone | Focus Area | Status |
|---|---|---|---|
| **Phase 0** | **Architecture & Domain Design** | Modular package hierarchy (`src/`), KYC simulation, domain dataclasses | ✅ **COMPLETE** |
| **Phase 1** | **Data Understanding & EDA** | Structural inspection, distributions, target analysis in `01` & `06` notebooks | ✅ **COMPLETE** |
| **Phase 2** | **Data Preprocessing & Validation** | Stratified split, `ColumnTransformer`, zero-leakage scaling & encoding in `02` & `07` | ✅ **COMPLETE** |
| **Phase 3** | **Credit Risk ML Modeling & Decisions** | Baseline (LR), Ensembles (RF, XGB), Risk Scoring (0–100) & Decision Policy (`03`, `04`, `05`, `08`, `09`) | ✅ **COMPLETE** |
| **Comparative** | **Dataset Comparison & Governance** | Cross-dataset comparison, statistical power & governance audit in `10_dataset_comparison.ipynb` | ✅ **COMPLETE** |
| **Phase 4** | **Model Evaluation, Explainability & Selection** | Multi-model evaluation, calibration, analytical thresholds, SHAP explainability, portfolio segmentation & model selection in `11_model_evaluation_explainability.ipynb` | ✅ **COMPLETE** |
| **Phase 5** | **Risk Scoring & Decision Engine** | Probability of Default $\rightarrow$ 0–100 Risk Score, business cutoffs (Approve/Review/Reject) | ⏳ Planned |
| **Phase 6** | **Product & Analyst Dashboard** | Interactive Streamlit interface for underwriting analysts | ⏳ Planned |

---

## 6. Project Directory Structure

```
Credit Risk & Loan Decision System/
├── .gitignore
├── requirements.txt                   # Environment dependencies
├── README.md                          # Project executive overview & architecture
├── PROJECT_GUIDE.md                   # Master technical & domain reference
├── FILE_GUIDE.md                      # Complete catalog of repository files
│
├── data/
│   ├── raw/
│   │   ├── credit_risk_50.csv         # Synthetic educational dataset (50 rows × 16 cols)
│   │   ├── german_credit_1000.csv     # UCI benchmark dataset (1,000 rows × 21 cols)
│   │   └── README.md                  # Raw data dictionary, provenance & governance rules
│   ├── processed/
│   │   └── README.md                  # Documentation of model-ready feature matrices
│   └── synthetic/
│       └── README.md                  # Specification for demo applicant fixtures
│
├── docs/
│   └── INDIA_DATA_PRIVACY.md          # DPDP Act 2023 & RBI digital lending guidelines
│
├── notebooks/
│   ├── 01_data_understanding.ipynb    # Synthetic Phase 1: EDA & Understanding
│   ├── 02_data_preprocessing.ipynb    # Synthetic Phase 2: Split, Preprocessing & Validation Gates
│   ├── 03_model_development.ipynb     # Synthetic Phase 3 Step 1: Baseline Logistic Regression
│   ├── 04_tree_based_models.ipynb     # Synthetic Phase 3 Step 2: Random Forest & XGBoost Ensembles
│   ├── 05_risk_scoring_and_decision.ipynb # Synthetic Phase 3 Step 3: Risk Scoring & Loan Decision Layer
│   ├── 06_real_data_understanding.ipynb    # Real Data Phase 1: 20-Section Comprehensive EDA
│   ├── 07_real_data_preprocessing.ipynb    # Real Data Phase 2: Dynamic ColumnTransformer & Validation
│   ├── 08_real_data_model_development.ipynb # Real Data Phase 3: Multi-Model Benchmark (CV + Test)
│   ├── 09_real_data_risk_scoring.ipynb     # Real Data Phase 3: Estimated PD, Internal Risk Score & Decisions
│   ├── 10_dataset_comparison.ipynb         # Comparative Analysis: Synthetic vs. Real UCI Benchmark
│   └── 11_model_evaluation_explainability.ipynb # Phase 4 Steps 1 & 2: Evaluation, Calibration, SHAP & Model Selection
│
└── src/
    ├── README.md                      # Architecture guide for source modules
    ├── application/
    │   ├── __init__.py
    │   └── applicant.py               # Central domain model (LoanApplicant dataclass)
    ├── verification/
    │   ├── __init__.py
    │   ├── pan_verification.py        # PAN syntax & mock registry verification
    │   ├── aadhaar_kyc.py             # Mock UIDAI biometric KYC verification
    │   ├── mobile_verification.py     # Mobile number format & OTP verification
    │   └── identity_checks.py         # Fuzzy string & cross-document identity checks
    ├── financial/
    │   ├── __init__.py
    │   ├── income_verification.py     # Declared vs detected income variance analysis
    │   ├── bank_statement_analyzer.py # Cashflow, average balance & bounce detection
    │   └── debt_analyzer.py           # DTI, LTI & EMI obligation calculations
    ├── credit/
    │   ├── __init__.py
    │   └── credit_profile.py          # Simulated CIBIL bureau profile generator
    └── features/
        ├── __init__.py
        └── feature_engineering.py     # Feature vector extraction for ML
```

---

## 7. Technology Stack

- **Data Manipulation**: `pandas>=2.0.0`, `numpy>=1.26.0`
- **Machine Learning & Preprocessing**: `scikit-learn>=1.3.0` (`ColumnTransformer`, `StandardScaler`, `OneHotEncoder`, `train_test_split`, `cross_validate`), `xgboost>=2.0.0`
- **Model Explainability & Interpretability**: `shap>=0.52.0` (LinearExplainer, TreeExplainer, Beeswarm & Waterfall plots)
- **Data Visualization**: `matplotlib>=3.8.0`, `seaborn>=0.13.0`
- **Interactive Development**: `jupyterlab>=4.0.0`, `ipykernel>=6.0.0`
- **Planned for Future Phases**: `streamlit` (Risk Analyst Dashboard)

---

## 8. Important Disclaimer

> **Simulation & Benchmark Dataset Notice**:  
> All applicant records, verification checks (PAN, Aadhaar KYC, Mobile OTP), bank statement data, and credit bureau scores in the synthetic dataset are **fictitious simulations** designed for educational, architectural, and workflow prototyping demonstrations. The UCI German Credit dataset is a historical credit-risk benchmark from 1994 (European banking context, Deutsche Mark currency) and does not represent contemporary Indian borrowers or live production banking data. No real personal data (PII) is stored or processed. In a commercial production environment, mock verification modules would be replaced with authorized, licensed API integrations complying with the **Digital Personal Data Protection (DPDP) Act 2023** and **Reserve Bank of India (RBI) Digital Lending Guidelines**.

