# 🔮 Customer Churn Prediction & Lifetime Value (LTV) Engine

A production-level predictive analytics system designed for telecommunications and subscription-based businesses. This engine ingests historical customer data to identify customers at high risk of cancellation (churn) and calculates their predicted Customer Lifetime Value (LTV) to help marketing teams prioritize high-value retention campaigns.

## 📌 Project Overview

| Feature | Description |
|---------|-------------|
| **Churn Prediction** | ML models to predict which customers will leave |
| **LTV Calculation** | Forecast expected lifetime revenue per customer |
| **REST API** | FastAPI service for real-time predictions |
| **Dashboard** | Interactive visualization of churn risk & LTV segments |
| **Containerized** | Docker-ready for easy deployment |

## 🛠️ Tech Stack

- **Language:** Python 3.10+, SQL
- **Database:** PostgreSQL (Data Warehouse)
- **ML & Analysis:** Pandas, Scikit-Learn, XGBoost, SHAP
- **API:** FastAPI + Uvicorn
- **Visualization:** Apache Superset / Metabase
- **Containerization:** Docker + Docker Compose

## 📊 Dataset

- **Source:** [Telco Customer Churn Dataset (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size:** 7,043 rows × 21 columns
- **Target Variable:** `Churn` (Yes/No)

## 📁 Project Structure

```
customer-churn-ltv-engine/
├── README.md
├── requirements.txt
├── docker-compose.yml
├── Dockerfile
├── .env
├── .gitignore
├── data/
│   ├── raw/                         # Original dataset
│   └── processed/                   # Cleaned & transformed data
├── notebooks/
│   ├── 01_eda.ipynb                 # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb # Feature creation
│   ├── 03_model_training.ipynb      # ML model training & evaluation
│   ├── 04_shap_analysis.ipynb       # Model explainability
│   └── 05_ltv_calculation.ipynb     # LTV modeling
├── src/
│   ├── data_ingestion.py            # CSV → PostgreSQL loader
│   ├── preprocessing.py             # Data cleaning functions
│   ├── feature_engineering.py       # Feature creation
│   ├── model_training.py            # Train & save models
│   ├── ltv_model.py                 # LTV calculation
│   └── utils.py                     # Helper functions
├── api/
│   ├── main.py                      # FastAPI application
│   ├── schemas.py                   # Pydantic models
│   └── endpoints/
│       ├── churn.py                 # Churn prediction endpoints
│       └── ltv.py                   # LTV prediction endpoints
├── models/                          # Saved ML models (.pkl)
├── dashboards/                      # Dashboard configurations
└── docs/
    ├── architecture.md              # System architecture
    └── api_docs.md                  # API reference
```

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/customer-churn-ltv-engine.git
cd customer-churn-ltv-engine
```

### 2. Setup Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 3. Setup PostgreSQL
- Install PostgreSQL
- Create database: `churn_db`
- Update `.env` with your credentials

### 4. Load Dataset
```bash
python src/data_ingestion.py
```

### 5. Run API
```bash
uvicorn api.main:app --reload
```

### 6. Open Dashboard
Visit `http://localhost:8088` for Superset dashboard

## 📅 Development Timeline

| Week | Focus Area |
|------|------------|
| Week 1 | Data Ingestion & Exploratory Data Analysis (EDA) |
| Week 2 | Feature Engineering & Predictive Modeling |
| Week 3 | LTV Calculation & API Development |
| Week 4 | Visualization Dashboard & Deployment |

## 👨‍💻 Author

**Zaalima Development Internship Project**

## 📄 License

This project is confidential and proprietary to Zaalima Development.
