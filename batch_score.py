"""
Week 4, Day 1-3: Batch Scoring Job
---------------------------------------
Scores every customer in `dim_customers` with the churn + LTV models
and writes results into `fact_predictions`, so BI tools (Superset/
Metabase) can read predictions straight from the warehouse instead of
calling the API per-customer.

In production this would run on a schedule (e.g. nightly via Airflow/
cron) after new billing data lands. Reuses the same ModelService as the
FastAPI layer, so batch and real-time scoring never drift apart.

Usage:
    python src/batch_score.py --database-url sqlite:///data/churn_ltv.db
"""
import argparse
import os
import sys
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import create_engine

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))
sys.path.insert(0, os.path.dirname(__file__))
from model_service import get_service  # noqa: E402
from schemas import CustomerFeatures  # noqa: E402

DEFAULT_DB_URL = "sqlite:///data/churn_ltv.db"
MODEL_VERSION = "v1.0.0"


def load_customers(engine) -> pd.DataFrame:
    return pd.read_sql("SELECT * FROM dim_customers", engine)


def to_customer_features(df: pd.DataFrame) -> list:
    records = df.drop(columns=["churn"], errors="ignore").to_dict(orient="records")
    return [CustomerFeatures(**r) for r in records]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL", DEFAULT_DB_URL))
    args = parser.parse_args()

    engine = create_engine(args.database_url)
    print(f"[batch_score] connecting to {engine.url}")

    df = load_customers(engine)
    print(f"[batch_score] scoring {len(df)} customers")

    service = get_service()
    customers = to_customer_features(df)
    predictions = service.predict_full_batch(customers)

    predictions["model_name"] = "churn_xgboost_ltv_xgboost"
    predictions["model_version"] = MODEL_VERSION
    predictions["predicted_at"] = datetime.now(timezone.utc).isoformat()

    out_table = predictions.rename(columns={"predicted_ltv": "predicted_ltv"})[
        ["customer_id", "model_name", "model_version", "churn_probability",
         "churn_prediction", "predicted_ltv", "ltv_segment", "predicted_at"]
    ]
    out_table.to_sql("fact_predictions", engine, if_exists="replace", index=False)

    print(f"[batch_score] wrote {len(out_table)} rows to fact_predictions")
    print(out_table.head(5).to_string())

    # Summary stats useful for a dashboard "as of last run" panel
    summary = {
        "total_customers": int(len(out_table)),
        "high_risk_count": int((out_table["churn_probability"] >= 0.6).sum()),
        "high_risk_pct": float((out_table["churn_probability"] >= 0.6).mean()),
        "high_ltv_count": int((out_table["ltv_segment"] == "High").sum()),
        "high_risk_high_ltv_count": int(
            ((out_table["churn_probability"] >= 0.6) & (out_table["ltv_segment"] == "High")).sum()
        ),
        "total_predicted_ltv_at_risk": float(
            out_table.loc[out_table["churn_probability"] >= 0.6, "predicted_ltv"].sum()
        ),
        "run_at": datetime.now(timezone.utc).isoformat(),
    }
    import json
    os.makedirs("reports", exist_ok=True)
    with open("reports/batch_score_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[batch_score] summary: {json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
