# Week 4 files — Batch Scoring, Dashboard & Deployment

## Files
- `batch_score.py` — scores all customers via the API's `ModelService` and writes to `fact_predictions`. Needs `api/model_service.py`, `api/schemas.py`, and `models/*.joblib` from earlier weeks to run.
- `dashboard.html` — **open this directly in any browser**, no server needed. Self-contained, real data embedded (7,043 scored customers).
- `dashboard_data.json` — the aggregated data feeding the dashboard.
- `batch_score_summary.json` — run summary (high-risk counts, LTV at risk, etc.)
- `Dockerfile` — builds the FastAPI service image.
- `docker-compose.yml` — orchestrates Postgres + API + Superset.
- `.dockerignore`
- `week4_deployment_report.md` — full write-up, including Superset connection steps.
- `PROJECT_README.md` — the complete 4-week project README, included for context since these files reference the full pipeline.

## Headline numbers (from the actual batch run)
- 7,043 customers scored
- 2,190 high-risk (31.1%)
- **255 customers both high-risk and high-LTV** — the priority retention list
- **$2.75M** in predicted LTV currently at risk

## To run standalone
`batch_score.py` and the Docker files assume the full project structure (`src/`, `api/`, `models/`, `sql/schema.sql`) from Weeks 1–3. `dashboard.html` has no dependencies — just open it.

Docker/Compose were written to spec but not build-tested in the dev sandbox (no Docker daemon there). Verify with `docker compose build` before production use.
