# `data/raw/` — Canonical Raw Data Layer

This folder contains the **immutable raw datasets** used by the platform.

---

## 1. Dual-Dataset Architecture

The platform maintains two distinct raw datasets, each serving a clear architectural and research purpose:

```
┌──────────────────────────────────────────────┬──────────────────────────────────────────────┐
│ 1. Synthetic Educational Dataset             │ 2. Real-World Benchmark Dataset              │
│    `data/raw/credit_risk_50.csv`             │    `data/raw/german_credit_1000.csv`         │
├──────────────────────────────────────────────┼──────────────────────────────────────────────┤
│ • 50 synthetic applicant profiles            │ • 1,000 historical bank loan records         │
│ • Indian lending domain features (₹, CIBIL)  │ • Historical European banking (Deutsche Mark)│
│ • Prototyping Flow A application workflow    │ • Real-world credit risk benchmark for ML    │
│ • Small-sample pipeline architecture test    │ • Statistical cross-validation & evaluation  │
└──────────────────────────────────────────────┴──────────────────────────────────────────────┘
```

> **Important Boundary Rule**: The synthetic and real-world benchmark datasets are maintained in strict isolation. They represent different currencies, banking ecosystems, and schemas, and are **never combined or merged into a single training dataset**.

---

## 2. Dataset 1: `credit_risk_50.csv` (Synthetic Development Dataset)

| Property | Value | Notes |
|---|---|---|
| **File Name** | `credit_risk_50.csv` | Educational development dataset |
| **Observation Count** | **50** | Synthetic applicant profiles |
| **Feature Count** | **16 columns** | 1 ID + 14 predictive features + 1 target |
| **Target Variable** | `default` | Binary integer (`0` = Repaid, `1` = Defaulted) |
| **Class Distribution** | 31 Non-default (62.0%) / 19 Default (38.0%) | Moderate class imbalance |
| **File Format** | CSV (Comma-Separated Values) | Standard tabular format |

### Column Dictionary (`credit_risk_50.csv`)

| # | Column Name | Type | Description |
|---|---|---|---|
| 1 | `applicant_id` | `int64` | Surrogate key (Excluded from ML feature vector) |
| 2 | `age` | `int64` | Applicant age in years (23–55) |
| 3 | `employment_type` | `str` | Nominal category: `Salaried`, `Self-employed`, `Business` |
| 4 | `monthly_income` | `int64` | Net verified monthly income in Indian Rupees (₹30k–₹1.25L) |
| 5 | `loan_amount` | `int64` | Principal loan amount requested in Rupees (₹1.2L–₹7.0L) |
| 6 | `loan_term_months` | `int64` | Requested loan tenure in months (24, 36, 48, 60) |
| 7 | `existing_emi` | `int64` | Total monthly active debt obligations (₹9k–₹30k) |
| 8 | `credit_score` | `int64` | Bureau credit score (605–800) |
| 9 | `credit_utilization_pct` | `int64` | Revolving credit line utilization percentage (18%–88%) |
| 10 | `late_payment_count` | `int64` | Number of past late EMI payments (0–5) |
| 11 | `previous_defaults` | `int64` | Historical count of loan defaults / write-offs (0–2) |
| 12 | `employment_years` | `int64` | Years of continuous employment / business (1–22) |
| 13 | `average_bank_balance` | `int64` | Average monthly bank balance (₹12k–₹1.5L) |
| 14 | `monthly_expenses` | `int64` | Estimated monthly living expenses (₹21k–₹70k) |
| 15 | `debt_to_income_ratio` | `float64` | Calculated DTI ratio (`existing_emi / monthly_income`, 0.18–0.55) |
| 16 | `default` | `int64` | **Target label** (0 = Non-default, 1 = Default) |

---

## 3. Dataset 2: `german_credit_1000.csv` (UCI German Credit Benchmark Dataset)

### Dataset Provenance & Citation
- **Source**: UCI Machine Learning Repository
- **Dataset Name**: Statlog (German Credit Data)
- **Official URL**: `https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data`
- **Donor / Creator**: Professor Dr. Hans Hofmann, Institut für Statistik und Ökonometrie, Universität Hamburg (1994)
- **Nature**: Real-world historical credit-risk benchmark dataset (1,000 credit applications from German commercial banking, denominated in Deutsche Mark).
- **Geographic & Regulatory Context**: Historical European banking. It is **NOT** Indian lending data, production banking data, or representative of Indian borrowers.

### Dataset Specification

| Property | Value | Notes |
|---|---|---|
| **File Name** | `german_credit_1000.csv` | Real-world benchmark dataset |
| **Observation Count** | **1,000** | Historical bank loan records |
| **Feature Count** | **21 columns** | 20 predictive features + 1 target |
| **Target Variable** | `default` | Binary integer (`0` = Good / Non-default, `1` = Bad / Default) |
| **Class Distribution** | 700 Non-default (70.0%) / 300 Default (30.0%) | 70/30 class balance |
| **File Format** | CSV (Comma-Separated Values) | Converted from canonical `german.data` |

### Feature Dictionary (`german_credit_1000.csv`)

| # | Feature Name | Data Type | Kind | Meaning / Categories | Available Pre-Decision | In ML Set | Notes / Justification |
|---|---|---|---|---|---|---|---|
| 1 | `status_existing_checking_account` | `object` | Categorical | Checking account status (`A11`: < 0 DM, `A12`: 0–200 DM, `A13`: >= 200 DM, `A14`: no checking account) | Yes | Yes | Core liquidity indicator |
| 2 | `duration_in_months` | `int64` | Numerical | Credit duration in months | Yes | Yes | Requested loan term |
| 3 | `credit_history` | `object` | Categorical | Credit repayment history (`A30`–`A34`: past credits, delays, critical accounts) | Yes | Yes | Bureau repayment track record |
| 4 | `purpose` | `object` | Categorical | Loan purpose (`A40`: car new, `A41`: car used, `A42`: furniture, `A43`: radio/TV, `A49`: business, etc.) | Yes | Yes | Loan purpose category |
| 5 | `credit_amount` | `int64` | Numerical | Principal credit amount in Deutsche Mark (DM) | Yes | Yes | Loan principal requested |
| 6 | `savings_account_bonds` | `object` | Categorical | Savings account balance / bonds (`A61`: < 100 DM to `A65`: unknown/none) | Yes | Yes | Reserve assets indicator |
| 7 | `present_employment_since` | `object` | Categorical | Employment duration (`A71`: unemployed to `A75`: >= 7 years) | Yes | Yes | Job stability indicator |
| 8 | `installment_rate_pct_disposable_income` | `int64` | Numerical | Installment rate as % of disposable income (1–4) | Yes | Yes | Debt burden / capacity proxy |
| 9 | `personal_status_sex` | `object` | Categorical | Personal status and sex (`A91`–`A95`: marital status & gender) | Yes | Yes | Demographic attribute in historical dataset (subject to fairness/bias scrutiny in modern applications) |
| 10 | `other_debtors_guarantors` | `object` | Categorical | Other debtors or guarantors (`A101`: none, `A102`: co-applicant, `A103`: guarantor) | Yes | Yes | Collateral / co-signer backing |
| 11 | `present_residence_since` | `int64` | Numerical | Years at present residence (1–4) | Yes | Yes | Residential stability |
| 12 | `property` | `object` | Categorical | Property ownership (`A121`: real estate to `A124`: unknown/no property) | Yes | Yes | Tangible asset indicator |
| 13 | `age_years` | `int64` | Numerical | Age in years | Yes | Yes | Applicant age |
| 14 | `other_installment_plans` | `object` | Categorical | Other installment debt plans (`A141`: bank, `A142`: stores, `A143`: none) | Yes | Yes | External debt obligations |
| 15 | `housing` | `object` | Categorical | Housing status (`A151`: rent, `A152`: own, `A153`: for free) | Yes | Yes | Living arrangement |
| 16 | `existing_credits_count` | `int64` | Numerical | Number of existing credits at this bank (1–4) | Yes | Yes | Relationship depth / active facility count |
| 17 | `job` | `object` | Categorical | Employment qualification (`A171`: unemployed/unskilled non-resident to `A174`: management/self-employed) | Yes | Yes | Occupation skill tier |
| 18 | `people_liable_maintenance` | `int64` | Numerical | Number of dependents liable to provide maintenance for (1–2) | Yes | Yes | Financial dependents count |
| 19 | `telephone` | `object` | Categorical | Registered landline telephone (`A191`: none, `A192`: yes) | Yes | Yes | Historical contactability indicator |
| 20 | `foreign_worker` | `object` | Categorical | Foreign worker status (`A201`: yes, `A202`: no) | Yes | Yes | Historical labor status attribute |
| 21 | `default` | `int64` | Target | Binary credit risk outcome (`0` = Good / Repaid, `1` = Bad / Default) | No (Post-outcome) | Target | Target variable for supervised classification |

---

## 4. Raw Data Governance Rule

> **Immutability Principle**: Never edit, filter, or overwrite files in `data/raw/` manually. All transformations, cleaning, and encodings must be executed strictly through reproducible code in `notebooks/` or `src/`, with outputs directed to `data/processed/`.
