---
title: AQI Cross-City Generalization Dashboard
emoji: 🌍
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# AQI Cross-City Generalization — 3-Group Regional Transfer Experiment

Trains and evaluates XGBoost, LightGBM, and Deep Bidirectional LSTM models across three distinct geographic cohorts in India to benchmark cross-city generalizability under spatial covariate shift.

## Regional Transfer Cohorts

| Group | Source Donor Cities (Train/Val) | Unseen Target City (Test) | Regional Regime |
|---|---|---|---|
| **1** | Pune + Chennai | **Mumbai** | Western Coastal Megacity |
| **2** | Delhi + Kanpur | **Lucknow** | Indo-Gangetic Plain |
| **3** | Bengaluru + Mysuru | **Hyderabad** | Deccan Plateau |

## Key Empirical Results (Evaluated on Clean N=835 Unseen Test Days)

- **LightGBM:** Mumbai: **99.40%** (MAE: 5.00) | Lucknow: **99.04%** (MAE: 6.57) | Hyderabad: **98.80%** (MAE: 5.58)
- **XGBoost:** Mumbai: **98.80%** (MAE: 4.49) | Lucknow: **98.44%** (MAE: 7.36) | Hyderabad: **98.92%** (MAE: 4.66)
- **Voting Ensemble:** Mumbai: **98.92%** | Lucknow: **98.68%** | Hyderabad: **98.44%**
- **Hardware Sensor Ablation (Tier 2):** Particulate-only telemetry (PM2.5, PM10, PM_ratio) preserves **>95.81%** cross-city accuracy while eliminating gaseous chemical sensor requirements.

## Setup & Local Run

```bash
pip install -r requirements.txt
python app.py   # http://localhost:5001 (or port 7860 in Docker)
```

## Repository Structure

```
.
├── data/                        # 9 city daily CSV records (841 days each) & live buffers
├── models/                      # Serialized weights, scalers, and empirical JSON benchmarks
├── preprocessing/               # Auxiliary preprocessing utilities
├── templates/index.html         # Interactive web dashboard with Chart.js analytics
├── train.py                     # Leak-free cross-city training pipeline
├── run_reduced_sensors.py       # Hardware sensor degradation benchmark (Experiment E9)
├── app.py                       # Production Flask API & CPCB telemetry fetcher
├── Dockerfile                   # Production container definition (Hugging Face / Cloud)
└── Procfile                     # Gunicorn web deployment definition
```

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | `GET` | Interactive analytics dashboard |
| `/api/predict` | `POST` | Single-sample AQI prediction + real-time TreeSHAP waterfall |
| `/api/fetch_live` | `GET` | Queries official data.gov.in CPCB portal for live city telemetry |
| `/api/live_metrics`| `GET` | Real-time streaming evaluation metrics |
| `/api/all_groups` | `GET` | Empirical benchmark results across all 3 cohorts |
| `/api/model_info` | `GET` | Model file sizes and metadata |
| `/api/status` | `GET` | Verifies models are loaded in memory |
