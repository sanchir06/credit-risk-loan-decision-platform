# docs/INDIA_DATA_PRIVACY.md
# India Data Privacy & Security Principles
## Credit Risk & Digital Lending Intelligence Platform

> **Who this guide is for:** You — the developer of this project.
> Read this before handling any applicant data, even synthetic data.
> Good privacy habits start at the learning stage, not the production stage.

---

## Why does privacy matter in a lending project?

A loan application contains some of the most sensitive information a person
can share: their income, their debts, their identity documents, their bank
transactions, and their credit history.

If this data is mishandled:
- The applicant can suffer financial harm or identity theft.
- The company can face legal and regulatory action.
- Trust in the entire digital lending system is damaged.

This is why financial data projects must be privacy-conscious from day one.

---

## What kind of sensitive data are we handling?

| Data Type | Why it's sensitive | Examples |
|-----------|-------------------|---------|
| PAN Number | Unique tax identity — can be misused for fraud | ABCDE1234F |
| Aadhaar Number | Biometric identity — most sensitive ID in India | 1234 5678 9012 |
| Bank account data | Financial privacy — reveals spending habits | Account number, transactions |
| Mobile number | Identity link — can be used for social engineering | 9876543210 |
| Income details | Financial privacy — can affect employment | Salary slips |
| Credit score | Financial reputation — affects loan eligibility | 720 |
| OTP | Temporary access token — must never be stored | 492817 |

---

## Core Privacy Principles for This Project

### 1. Do NOT commit real PAN numbers

PAN numbers identify real Indian taxpayers.

**Rule:** Never put a real PAN number in:
- Source code
- Jupyter notebooks
- Comments
- Example outputs
- Test data

**What to do instead:** Use clearly fictitious PANs:
- `DEMO01234X` or `TEST01234Z` — obviously not real
- Or use a format-valid but randomly generated string

---

### 2. Do NOT commit Aadhaar numbers

Aadhaar numbers are among the most sensitive identifiers in India.
Even storing them unnecessarily is a regulatory violation.

**Rule:**
- Never store full Aadhaar numbers, even in test files.
- If displaying any Aadhaar reference, show only the last 4 digits: `XXXX XXXX 9012`
- The system should discard the Aadhaar number immediately after verification.
- Store only the verification RESULT, not the number itself.

---

### 3. Do NOT commit real bank statements

Bank statements contain complete financial histories.

**Rule:**
- Never include real bank statement files in the repository.
- Use only synthetic/generated transaction data for development.
- If testing with real data locally, keep it outside the project folder
  or add it to `.gitignore`.

---

### 4. Do NOT store OTPs

OTPs are single-use temporary codes. They have no value after use.

**Rule:**
- Never log OTPs to files or databases.
- Never display OTPs in any API response or UI (except in the demo simulation context, clearly labeled).
- In production: store only a hashed version in cache with a short TTL (5–10 minutes), then delete.

---

### 5. Do NOT expose API keys or credentials

If you ever add a real verification API, SMS gateway, or bureau API:

**Rule:**
- Store credentials in environment variables (`.env` file).
- The `.env` file is already in `.gitignore`.
- Never hard-code API keys, tokens, or passwords in source code.
- Use a tool like `python-dotenv` to load environment variables.

**Example of what NOT to do:**
```python
# BAD — never do this
API_KEY = "sk-live-abc123xyz789"
```

**Example of what TO do:**
```python
# GOOD — load from environment
import os
API_KEY = os.getenv("VERIFICATION_API_KEY")
```

---

### 6. Use synthetic/demo identities for all development

This project uses simulated data for all development and testing.

**Rule:**
- All applicant data in the codebase, notebooks, and tests must be synthetic.
- Clearly label any demo data with `is_demo_data = True` or similar.
- See `data/synthetic/` for where future synthetic datasets will live.

**Portfolio statement:**
> "This portfolio project uses synthetic/mock verification data and does not
> claim access to government, Aadhaar, PAN, or credit bureau systems."

---

### 7. Mask sensitive identifiers in logs and UI

When you eventually build logging or dashboards:

**Rule:**
- Show only masked versions of sensitive data:
  - PAN: `XXXXX1234F` (mask first 5 characters)
  - Bank Account: `XXXXXX7890` (show only last 4)
  - Mobile: `XXXXXX3210` (show only last 4)
  - Aadhaar: Never show — show only reference ID
- Log application IDs and outcomes, not raw sensitive fields.

---

### 8. Keep raw sensitive data separate from ML features

The ML model should never receive:
- Raw PAN numbers
- Aadhaar numbers
- Full bank account numbers
- Raw transaction descriptions

The ML model should receive:
- Numeric and boolean features (credit_score, dti, pan_verified)
- Aggregated statistics (avg_balance, bounce_count)
- Anonymized flags (identity_match_score: 0 or 1)

This design is called **data minimization** — the model gets only what it needs.

---

### 9. Understand India's data privacy regulations

For a fintech/lending project, be aware of:

| Framework | What it covers |
|-----------|---------------|
| **IT Act 2000 + SPDI Rules 2011** | Sensitive personal data (financial info, health data) |
| **DPDP Act 2023** (Digital Personal Data Protection Act) | India's new comprehensive data protection law |
| **RBI KYC Master Direction** | How lenders must perform and store KYC |
| **Aadhaar Act 2016** | Rules on who can use Aadhaar for verification |
| **PCI-DSS** | Payment card industry data security (if handling cards) |

> **Note for interview:** You don't need to memorize these laws.
> Knowing that they EXIST and that your project is designed with them in mind
> shows data engineering maturity.

---

## What this project is — and is NOT

| This project IS | This project is NOT |
|----------------|---------------------|
| A learning portfolio for Data Science | A production lending system |
| Using synthetic/demo verification data | Connected to UIDAI, Income Tax, or banks |
| Showing the architecture of a real system | Claiming to be a real authorized KYC provider |
| Privacy-conscious by design | Making claims about regulatory compliance |

---

## Quick Privacy Checklist (before each commit)

Before you run `git commit`, check:

- [ ] No real PAN numbers in any file
- [ ] No Aadhaar numbers anywhere
- [ ] No real bank statement data
- [ ] No OTPs stored in code or notebooks
- [ ] No API keys or credentials in source code
- [ ] `.env` file is in `.gitignore`
- [ ] All test data is clearly synthetic or fictional

---

*Privacy is not a feature you add at the end. It is a design principle from the start.*
