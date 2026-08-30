# `data/synthetic/` — Product & Workflow Simulation Fixtures

This directory is designated for **synthetic Indian loan applicant fixtures** designed to support **Flow A (The Product / Application Simulation Flow)**.

---

## 1. Important Distinction: Product Simulation vs. ML Dataset

> **CRITICAL ARCHITECTURAL DISTINCTION**:
> - `data/raw/credit_risk_50.csv`: The **historical credit dataset** used for training and evaluating machine learning default-prediction models.
> - `data/synthetic/`: The **application simulation fixture directory** containing mock multi-field applicant records used to test the end-to-end identity verification, financial checks, bureau profiling, and UI dashboard workflows.

---

## 2. Supported Use Cases

| Use Case | Description |
|---|---|
| **Identity Verification Testing** | Testing PAN formatting, mock Aadhaar KYC checks, and mobile OTP validation in `src/verification/`. |
| **Integration & Workflow Testing** | Running end-to-end integration tests where a mock applicant passes or fails specific eligibility gates. |
| **UI & Dashboard Demonstration** | Providing realistic mock borrower profiles for the Streamlit risk analyst interface in later phases. |
| **Edge-Case Simulation** | Simulating thin-file borrowers, name mismatch edge cases, high DTI applicants, and salary inflation scenarios. |

---

## 3. Schema Example for Synthetic Applicant Records

```json
{
  "applicant_id": "APP00042",
  "full_name": "Priya Sharma",
  "pan_number": "DEMO01234X",
  "aadhaar_number": "123456789012",
  "date_of_birth": "1995-06-15",
  "mobile_number": "9876543210",
  "employment_type": "Salaried",
  "employer_name": "Infosys Ltd",
  "employment_length_years": 4.5,
  "monthly_income_declared": 72000.0,
  "loan_amount_requested": 400000.0,
  "loan_tenure_months": 36,
  "existing_emi_monthly": 18000.0,
  "existing_loan_count": 1,
  "credit_score": 745,
  "credit_utilization_ratio": 0.32,
  "late_payment_count": 0,
  "previous_defaults": 0,
  "avg_monthly_balance": 52000.0,
  "monthly_expenses": 36000.0,
  "is_demo_data": true
}
```

---

## 4. Privacy & Compliance Compliance

All fixtures placed in this folder must use **fictitious sample names, test PAN formats (`DEMO...`), and non-routable phone numbers** in full adherence with [`docs/INDIA_DATA_PRIVACY.md`](file:///c:/Users/sanch/Downloads/Credit%20Risk%20&%20Loan%20Decision%20System/docs/INDIA_DATA_PRIVACY.md).
