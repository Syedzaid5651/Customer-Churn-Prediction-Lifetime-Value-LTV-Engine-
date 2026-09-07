# Customer Churn Prediction & LTV Engine

A predictive analytics system for telecom/subscription businesses: identifies customers at high risk of churn, estimates their predicted lifetime value (LTV), and surfaces a prioritized retention target list.

Built over 4 weeks against the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (7,043 customers, IBM/Kaggle).

## Headline results

- **26.5%** overall churn rate; strongest drivers are **contract type**, **tenure**, and **fiber-optic internet service** (confirmed independently by both EDA and SHAP).
- Best churn model: **Random Forest**, F1=0.64, ROC-AUC=0.84 on held-out data.
- LTV regression model: **XGBoost**, R²=0.985, MAE ≈ $185.
- **255 customers** are simultaneously high-churn-risk and high-LTV — the top retention priority list, representing **$2.75M** in predicted lifetime value currently at risk.

## Architecture

```
CSV source data
      |
      v
sql/schema.sql (Postgres DDL)  <---  src/ingest.py (clean + load)
      |
      v
dim_customers (analytics-ready table)
      |
      +--> src/features.py (feature engineering)
      |
      +--> src/train_models.py --> models/best_model.joblib (churn)
      |         |
      |         v
      |    src/explain.py (SHAP)
      |
      +--> src/construct_ltv_target.py (LTV target via hazard heuristic)
      |         |
      |         v
      |    src/train_ltv_model.py --> models/ltv_model.joblib
      |
      v
api/main.py (FastAPI)  <-->  src/batch_score.py --> fact_predictions table
      |                                                    |
      v                                                    v
  internal apps                              reports/dashboard.html / Superset
```

## Project structure

```
project1_churn_ltv/
├── data/                   raw CSV, cleaned CSV, local sqlite dev DB
├── sql/schema.sql           Postgres DDL: raw -> dim -> fact_predictions + BI view
├── src/
│   ├── ingest.py             Week 1: load + clean
│   ├── eda.py                 Week 1: exploratory analysis + charts
│   ├── features.py             Week 2: feature engineering
│   ├── train_models.py          Week 2: churn classifiers
│   ├── explain.py                Week 2: SHAP explainability
│   ├── construct_ltv_target.py    Week 3: LTV target construction
│   ├── train_ltv_model.py          Week 3: LTV regression
│   └── batch_score.py               Week 4: batch scoring job
├── api/
│   ├── main.py               FastAPI app
│   ├── model_service.py       prediction logic
│   └── schemas.py              request/response models
├── models/                  trained model artifacts (.joblib)
├── reports/
│   ├── week1_baseline_report.md
│   ├── week2_modeling_report.md
│   ├── week3_ltv_api_report.md
│   ├── week4_deployment_report.md
│   ├── dashboard.html         interactive dashboard (open in any browser)
│   └── figures/               all EDA + SHAP charts
├── Dockerfile
├── docker-compose.yml        Postgres + API + Superset
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt

# 1. Load and clean data (defaults to local SQLite; set DATABASE_URL for Postgres)
python src/ingest.py

# 2. Run EDA
python src/eda.py

# 3. Train churn models + SHAP explanations
python src/train_models.py
python src/explain.py

# 4. Build LTV target and train LTV model
python src/construct_ltv_target.py
python src/train_ltv_model.py

# 5. Batch-score all customers into fact_predictions
python src/batch_score.py

# 6. Serve predictions via API
uvicorn api.main:app --reload --port 8000
```

Or with Docker (Postgres + API + Superset together):
```bash
docker compose up -d
```

## API quick reference

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Model load status |
| `/predict/churn` | POST | Batch churn probability |
| `/predict/ltv` | POST | Batch LTV prediction |
| `/predict/full` | POST | Batch churn + LTV + priority score |
| `/predict/full/single` | POST | Single-customer convenience endpoint |

## Known limitations (stated plainly)

- **LTV is a derived target**, not observed ground truth — built from a hazard-rate heuristic chained off the churn model (see `week3_ltv_api_report.md` for the full method). Replace with real revenue data if/when available.
- **Docker/Compose configs are written to spec but not build-tested** in the development sandbox (no Docker daemon available there) — verify with `docker compose build` before production use.
- The dashboard (`reports/dashboard.html`) is a static, self-contained snapshot of one batch-scoring run. For a live, filterable BI tool, connect Superset per the instructions in `week4_deployment_report.md`.
- Model performance (ROC-AUC ~0.84) reflects a real ceiling in this feature set's predictive signal, consistent across three very different model families — additional external data (support tickets, usage trends, competitor pricing) would likely be needed to push meaningfully higher.

## Week-by-week reports

- [Week 1 — Data ingestion & baseline EDA](reports/week1_baseline_report.md)
- [Week 2 — Feature engineering, modeling & SHAP](reports/week2_modeling_report.md)
- [Week 3 — LTV modeling & FastAPI service](reports/week3_ltv_api_report.md)
- [Week 4 — Dashboard, batch scoring & deployment](reports/week4_deployment_report.md)
