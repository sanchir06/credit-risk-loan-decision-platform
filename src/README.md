# `src/` — Production Source Architecture

This package contains the core business logic, domain entities, simulated verification workflows, and feature engineering transformations for the **Credit Risk & Loan Decision Platform**.

---

## 1. Package Structure & Submodules

```
src/
├── application/
│   ├── __init__.py
│   └── applicant.py               ← Central LoanApplicant domain entity dataclass
│
├── verification/                  ← Identity Risk & Eligibility Layer
│   ├── __init__.py
│   ├── pan_verification.py        ← PAN syntax validation & mock registry checks
│   ├── aadhaar_kyc.py             ← Mock UIDAI biometric KYC verification
│   ├── mobile_verification.py     ← Mobile number syntax & mock OTP verification
│   └── identity_checks.py         ← Fuzzy string & cross-document identity consistency
│
├── financial/                     ← Financial Capability & Cashflow Layer
│   ├── __init__.py
│   ├── income_verification.py     ← Declared vs detected income variance analysis
│   ├── bank_statement_analyzer.py ← Average balance, net cashflow & bounce metrics
│   └── debt_analyzer.py           ← Debt-to-Income (DTI) & Loan-to-Income (LTI) ratios
│
├── credit/                        ← Credit Bureau Layer
│   ├── __init__.py
│   └── credit_profile.py          ← Simulated CIBIL/Experian credit profile generator
│
└── features/                      ← ML Feature Vector Extraction
    ├── __init__.py
    └── feature_engineering.py     ← Extracts credit risk feature vector from LoanApplicant
```

---

## 2. Module Responsibilities

### A. `src/application/`
- **`applicant.py`**: Defines the central `LoanApplicant` dataclass modeling a single loan application. Encapsulates raw applicant inputs (income, loan requested, DOB), verification statuses (`pan_verified`, `kyc_verified`), financial signals, and bureau metrics in a single structured object.

### B. `src/verification/` *(Identity Risk Layer)*
- **`pan_verification.py`**: Validates the standard 10-character Indian PAN regex format (`AAAAA9999A`) and simulates government registry verification.
- **`aadhaar_kyc.py`**: Simulates UIDAI Aadhaar biometric KYC verification (Verhoeff checksum format validation & mock e-KYC response).
- **`mobile_verification.py`**: Validates 10-digit Indian mobile number formats and simulates SMS OTP delivery and verification.
- **`identity_checks.py`**: Executes cross-source identity matching (fuzzy Levenshtein name similarity, DOB alignment) across PAN, Aadhaar, and bank account records.

> **Architectural Isolation**: Outputs from this layer (`pan_verified`, `kyc_verified`, `identity_match_status`) function as **Eligibility Gates** in the application workflow and are **not** fed into the credit-risk ML feature vector.

### C. `src/financial/`
- **`income_verification.py`**: Compares customer-declared monthly income against detected bank salary credits to flag inflation or inconsistency.
- **`bank_statement_analyzer.py`**: Analyzes multi-month bank transactions to extract average balance, monthly net cashflow, negative balance frequency, and mandate/cheque bounce counts.
- **`debt_analyzer.py`**: Calculates debt burden metrics including Debt-to-Income (DTI) ratio, Loan-to-Income (LTI) ratio, and Fixed Obligation to Income Ratio (FOIR).

### D. `src/credit/`
- **`credit_profile.py`**: Simulates bureau credit profile retrieval (e.g., CIBIL score 300–900, credit utilization ratio, past delinquency counts, previous default history, and recent inquiries).

### E. `src/features/`
- **`feature_engineering.py`**: Acts as the bridge between raw application objects and machine learning.
  - Implements dynamic age calculation from date of birth (`calculate_age()`).
  - Implements `build_feature_vector()`: Extracts purely credit-default-relevant indicators while strictly omitting identity flags and surrogate keys.

---

## 3. Data Flow Between Layers

```
LoanApplicant (Application Data Model)
     │
     ├──────────► verification/ ──► Fills: pan_verified, kyc_verified, mobile_verified, identity_match
     │                             (Gates application: PASS / MANUAL_REVIEW / REJECT)
     │
     ├──────────► financial/    ──► Fills: income_consistency, avg_balance, cashflow, DTI, LTI
     │
     ├──────────► credit/       ──► Fills: credit_score, credit_utilization, late_payments, defaults
     │
     └──────────► features/     ──► Reads applicant fields ──► Produces Credit Risk Feature Vector
                                                                        │
                                                                        ▼
                                                             Machine Learning Pipeline
                                                             (Phase 3: Model Development)
```

---

## 4. Future Planned Modules & Architectural Roadmap

As future project phases are implemented for production deployment or dashboard integration:
- **`src/models/`** (Planned): Production model inference and pipeline serialization wrappers (model training, cross-validation, and benchmarking are currently executed directly within interactive notebooks `03`, `04`, `08`, and `11`).
- **`src/explainability/`** (Planned): Standalone SHAP explanation modules for deployment serving (Phase 4 evaluation and SHAP explainability are currently executed in `notebooks/11_model_evaluation_explainability.ipynb`).
- **`src/decisioning/`** (Planned / Phase 5): Standalone production risk scoring (PD $\rightarrow$ 0–100 score) and decision engine policy wrappers.
