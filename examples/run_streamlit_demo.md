# Running the Streamlit Dashboard (Phase 5 Step 3)

This document explains how to run the Streamlit credit risk dashboard locally.

## Prerequisites

Model artifacts must be built and the FastAPI REST API must be running before using the dashboard.

## Step 1 — Build Model Artifacts (if not already done)

```bash
python scripts/build_model_artifacts.py
```

## Step 2 — Start the FastAPI REST API (Terminal 1)

```bash
python -m uvicorn api.main:app --reload
```

The API will be available at: `http://127.0.0.1:8000`

Verify it is working by visiting: `http://127.0.0.1:8000/docs`

## Step 3 — Start the Streamlit Dashboard (Terminal 2)

```bash
streamlit run streamlit_app.py
```

Streamlit will print a local URL (usually `http://localhost:8501`). Open it in your browser.

## Step 4 — Using the Dashboard

1. **Check API connectivity** — Click "Check API Health" in the sidebar to confirm the FastAPI service is running and the model is loaded.
2. **Load Model Info** — Click "Load Model Info" in the sidebar to see model metadata from the API.
3. **Fill in the applicant form** — Complete all four sections (Financial, Loan, Employment, Personal) using the 20 UCI German Credit feature inputs.
4. **Assess Credit Risk** — Click the "Assess Credit Risk" button. The dashboard sends the form data to `POST /predict` and displays the result.

## Architecture

```
Streamlit UI (streamlit_app.py)
        ↓ HTTP  (httpx)
FastAPI REST API  (api/main.py)
        ↓
CreditRiskPredictor  (src/prediction/predictor.py)
        ↓
preprocessing.joblib  +  credit_risk_model.joblib
        ↓
Risk Policy
        ↓
JSON Response
        ↓
Streamlit Result Panel
```

The Streamlit dashboard does **not** load CSV datasets, model artifacts, or run any training code.
All inference is performed by the FastAPI service.

## Important Disclaimer

> **Demonstration only.** This dashboard uses the historical UCI Statlog German Credit benchmark dataset (1994, Prof. Dr. Hans Hofmann). It is NOT an Indian banking approval decision and should NOT be used for real lending decisions. Risk policy thresholds are demonstration assumptions only.

## Running All Tests

```bash
pytest tests/ -v
```

Expected: all tests across `test_prediction_pipeline.py`, `test_api.py`, and `test_streamlit_app.py` pass.
