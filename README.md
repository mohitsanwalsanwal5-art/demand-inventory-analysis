# Project FORESIGHT: Demand & Inventory Intelligence Platform
**Enterprise Decision Support System for NorthBay Living**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Deployment-Production--Ready-success.svg)]()

---

## 1. Executive & Project Overview

**Project FORESIGHT** is an enterprise-grade demand forecasting and inventory risk intelligence engine developed for **NorthBay Living** (a multi-category home and living brand). The platform connects statistical machine learning demand projections directly to operational inventory buffers and financial exposure analysis, transforming raw sales and inventory snapshots into proactive procurement and working capital decisions.

### Key Highlights
- **Validated Demand Forecasting:** Multi-step forward demand projections over an 8-week horizon across 50 active commercial SKUs.
- **Model Superiority:** `HistGradientBoostingRegressor` achieves **10.56% WAPE** across rolling-origin backtesting, outperforming the rigorous Seasonal-Naive benchmark (**11.17% WAPE**) by **55 basis points** with near-zero aggregate bias (-1.41%).
- **Dual Working Capital Optimization:** Identifies **₹1.79 Cr (₹17,948,396.12)** total value at stake across active inventory:
  - **Stockout Exposure (6 SKUs):** ₹45.18 Lakhs (₹4,517,903.87) revenue at risk due to buffer breaches during supplier lead time.
  - **Overstock Capital Trapped (20 SKUs):** ₹1.34 Cr (₹13,430,492.25) working capital tied up in stock exceeding 4 weeks of forward demand.
  - **Balanced/Healthy (24 SKUs):** 48% of active commercial catalog operating within optimal inventory parameters.
- **Full Inventory Reconciliation:** Preserves the complete 200-SKU inventory universe (50 active commercial SKUs + 150 warehouse-only SKUs with zero historical sales) without silent data deletion.

---

## 2. System Architecture

```
                                  +---------------------------------------+
                                  |         Raw Data Ingestion            |
                                  | sales_daily, sku_master, calendar,   |
                                  |         inventory_snapshots           |
                                  +-------------------+-------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  |    Data Harmonization & Validation    |
                                  |   Weekly aggregation, SKU split       |
                                  |  (50 Commercial / 150 Warehouse-Only) |
                                  +-------------------+-------------------+
                                                      |
                         +----------------------------+----------------------------+
                         |                                                         |
                         v                                                         v
       +------------------------------------+                    +------------------------------------+
       |   D3 Demand Forecasting Engine     |                    |   D4 Risk & Rupee Impact Engine    |
       |  HistGradientBoostingRegressor     |                    |  Lead-time demand buffer breach    |
       |  8-Week rolling-origin forecast    +------------------->|  Overstock forward demand analysis |
       |  WAPE: 10.56% vs 11.17% baseline   |   Forecast Units   |  Rupee exposure quantification     |
       +-----------------+------------------+                    +-----------------+------------------+
                         |                                                         |
                         +----------------------------+----------------------------+
                                                      |
                         +----------------------------+----------------------------+
                         |                                                         |
                         v                                                         v
       +------------------------------------+                    +------------------------------------+
       |   D5 Streamlit Operations Console  |                    |   D6 FastAPI Production Microservice|
       |  Executive KPI cards & risk table  |                    |  High-throughput REST API          |
       |  Interactive SKU demand curves     |                    |  /health, /forecast, /risk,        |
       |  4-quadrant decision matrix        |                    |  /score/batch                      |
       +------------------------------------+                    +------------------------------------+
```

---

## 3. Data Description & Catalog Scope

The analysis operates on real-world retail enterprise data spanning two calendar years (January 2024 through December 2025):
- **Historical Sales:** Daily transactional sales aggregated to standard Monday-starting calendar weeks through **2025-12-28** (104 complete historical weeks).
- **Forecast Horizon:** 8 forward weeks spanning **2025-12-29 to 2026-02-16**.
- **SKU Catalog Reconciliation:**
  - **Total Tracked Inventory SKUs:** 200 SKUs (10,950 total physical units in warehouse, ₹6.45 Cr valuation).
  - **Active Commercial SKUs:** 50 SKUs with continuous sales history, evaluated for forward demand forecasting and commercial risk scoring.
  - **Warehouse-Only SKUs:** 150 SKUs with zero sales history (holding 6,675 physical units, ₹3.93 Cr valuation), maintained in reporting for audit and physical reconciliation.

---

## 4. Forecasting Model & Validation (Deliverable D3)

Demand is projected using an autoregressive gradient boosting architecture with rolling-origin temporal cross-validation:

| Model / Baseline | Evaluation Metric | Backtest Result | Benchmark Comparison | Model Status |
| :--- | :--- | :--- | :--- | :--- |
| **HistGradientBoostingRegressor** | **WAPE** | **10.56%** | **-0.55% vs Baseline** | **PRODUCTION MODEL** |
| Seasonal-Naive Baseline | WAPE | 11.17% | Reference standard | Baseline Benchmark |
| Aggregate Forecast Bias | Mean Bias | -1.41% | Minimal under-forecasting | Production Validated |

- **Production Model Artifact:** `src/models/forecast_model.joblib` (597 KB).
- **Features Used:** Autoregressive lags (Lag 1, 2, 4, 8, 12, 52), rolling statistics (4-week and 12-week moving averages and standard deviations), calendar features (week of year, month, quarter), and pricing indicators.
- **Zero Leakage:** Strict temporal separation ensures no future information is accessible during feature computation.

---

## 5. Inventory Risk Scoring & Financial Impact (Deliverable D4)

The risk engine connects the 8-week forward forecast to physical inventory parameters (Current Stock, On Order, Supplier Lead Time, Safety Stock):

### Primary Stockout Methodology
$$\text{Lead Time Demand (LTD)} = \text{Forecast demand during Supplier Lead Time (Days)}$$
$$\text{Required Inventory Buffer} = \text{LTD} + \text{Safety Stock}$$
$$\text{Available Inventory} = \text{Current Stock} + \text{On Order}$$
$$\text{Stockout Flag} = 1 \quad \text{if Available Inventory} < \text{Required Buffer, else } 0$$
$$\text{Revenue at Risk} = \text{Buffer Gap (Units)} \times \text{Unit Selling Price}$$

### Primary Overstock Methodology
$$\text{Excess Units} = \max(0, \text{Available Units} - \text{Forward 4-Week Demand})$$
$$\text{Overstock Flag} = 1 \quad \text{if Available Units} > 1.5 \times \text{Forward 4-Week Demand, else } 0$$
$$\text{Capital Tied Up} = \text{Excess Units} \times \text{Unit Cost Price}$$

### Portfolio Risk Breakdown
- **Stockout Risk:** **6 SKUs** (`SKU010`, `SKU012`, `SKU017`, `SKU023`, `SKU031`, `SKU040`) $\rightarrow$ **₹45.18 Lakhs** revenue exposure.
- **Overstock Risk:** **20 SKUs** $\rightarrow$ **₹1.34 Cr** working capital trapped.
- **Healthy SKUs:** **24 SKUs** $\rightarrow$ Operating within normal buffer limits.
- **Total Value at Stake:** **₹1.79 Cr (₹17,948,396.12)**.

---

## 6. Repository Structure

```
.
├── .streamlit/
│   └── config.toml                  # Streamlit headless server configuration
├── app/
│   └── app.py                       # D5 Streamlit Operations Planning Dashboard
├── data/
│   ├── processed/                   # Harmonized datasets, model metrics, risk scores
│   │   ├── backtest_results.csv
│   │   ├── decision_grid.csv
│   │   ├── forecast_results.csv
│   │   ├── latest_inventory_status.csv
│   │   ├── risk_scores.csv
│   │   ├── risk_summary.json
│   │   └── weekly_demand_analysis_ready.csv
│   └── raw/                         # Source CSV files
├── notebooks/
│   ├── 01_eda.ipynb                 # D2 Exploratory Data Analysis
│   ├── 02_baseline.ipynb            # D3 Seasonal-Naive Baseline
│   └── 03_model.ipynb               # D3 Machine Learning Pipeline & Backtesting
├── reports/
│   ├── FORESIGHT_Executive_Readout.pptx # D7 14-slide executive presentation
│   ├── eda_memo.md                  # D2 Technical EDA memo
│   ├── executive_readout.md         # D7 Executive readout document
│   ├── risk_memo.md                 # D4 Inventory risk methodology memo
│   └── figures/                     # 9 high-resolution analytical figures
├── service/
│   └── main.py                      # D6 FastAPI Scoring Microservice
├── src/
│   ├── __init__.py
│   ├── forecast.py                  # DemandForecastingEngine & feature pipeline
│   ├── models/
│   │   └── forecast_model.joblib    # Trained production model (Git-tracked)
│   ├── pipeline.py                  # D1 Data ingestion & reconciliation
│   └── risk.py                      # D4 InventoryRiskEngine
├── Procfile                         # Deployment process specification
├── render.yaml                      # Render Blueprint (Dashboard + API)
├── requirements.txt                 # Frozen production dependencies
└── README.md                        # Documentation
```

---

## 7. Local Installation & Execution

### Step 1: Clone Repository
```bash
git clone https://github.com/Jalagamdolu/Foresight_demand_inventary.git
cd Foresight_demand_inventary
```

### Step 2: Environment Setup
```bash
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### Step 3: Launch Streamlit Dashboard (Port 8501)
```bash
streamlit run app/app.py
```
Access the dashboard at `http://localhost:8501`.

### Step 4: Launch FastAPI Scoring Microservice (Port 8000)
```bash
uvicorn service.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger UI: `http://localhost:8000/docs`.

---

## 8. API Endpoint Documentation

| Method | Endpoint | Description | Sample Response / Status |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | Service & model health check | `200 OK` (Model type, WAPE, Active SKUs) |
| `GET` | `/forecast/{sku}` | 8-Week forecast with P10/P90 intervals | `200 OK` (Weekly units, weekly revenue) |
| `GET` | `/risk/{sku}` | Stockout/overstock risk & Rupee impact | `200 OK` (Buffer gap, revenue at risk) |
| `POST`| `/score/batch` | Batch scoring across multiple SKUs | `200 OK` (Scored results + error handling) |
| `GET` | `/catalog/skus` | Catalog list and summary status | `200 OK` (50 active commercial SKUs) |

### Sample Request: Batch Scoring
```bash
curl -X POST "http://localhost:8000/score/batch" \
     -H "Content-Type: application/json" \
     -d '{"sku_ids": ["SKU010", "SKU012", "SKU999"]}'
```

---

## 9. Cloud Deployment Configuration

The repository contains declarative infrastructure definitions for multi-service cloud hosting:

- **Streamlit Community Cloud:**
  - Repository: `Jalagamdolu/Foresight_demand_inventary`
  - Branch: `main`
  - Main file path: `app/app.py`
- **Render (`render.yaml`):**
  - **Service 1 (Streamlit Web):** `streamlit run app/app.py --server.port $PORT --server.address 0.0.0.0`
  - **Service 2 (FastAPI Microservice):** `uvicorn service.main:app --host 0.0.0.0 --port $PORT`

---

## 10. Operational Scope & Limitations

1. **Decision Support, Not Guaranteed Demand:** Forecasts represent probabilistic mathematical expectations based on historical patterns; they are intended to inform merchandiser decisions rather than automate purchase orders.
2. **Exposure Estimates:** Rupee figures denote financial value at stake (potential gross revenue loss or tied-up working capital), not realized ledger losses.
3. **No Automated PO Placement:** The system flags replenishment urgency ("Reorder Now") but requires human buyer approval before issuing purchase orders to suppliers.
4. **No Dynamic Price Optimization:** Markdown recommendations indicate surplus stock clearance needs; price elasticity curves and discount rates must be set by the merchandising team.
5. **Static Supplier Lead Times:** Model operates on contract lead times defined in master data; live EDI/freight tracking updates are not integrated.
