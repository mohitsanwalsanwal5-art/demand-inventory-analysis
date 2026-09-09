"""
Project FORESIGHT - Inventory Risk Scoring & Demand Forecasting API (Deliverable D6)
Client: NorthBay Living

Production FastAPI service exposing real-time demand forecasts and operational
inventory risk decisioning for the 50 commercial SKUs.

Endpoints:
  - GET  /health           : Health check and model metadata
  - GET  /forecast/{sku_id}: 8-week forward forecast & prediction intervals
  - GET  /risk/{sku_id}    : Dynamic lead-time buffer risk & rupee exposure
  - POST /score/batch      : Multi-SKU batch evaluation with error handling
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Determine project root path (repository-relative)
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"

# Initialize FastAPI application
app = FastAPI(
    title="Project FORESIGHT: Demand & Inventory Scoring API",
    description="Operational scoring API powering dynamic lead-time stockout risk, overstock detection, and rupee exposure for NorthBay Living.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for frontend clients (e.g. Streamlit dashboard or React/Vue apps)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global in-memory cache for fast read latency
_DATA_CACHE: Dict[str, Any] = {}


def load_datasets():
    """Load and index processed forecast and risk datasets at application startup."""
    fc_path = DATA_DIR / "forecast_results.csv"
    risk_path = DATA_DIR / "risk_scores.csv"
    summary_path = DATA_DIR / "risk_summary.json"

    if not fc_path.exists():
        raise FileNotFoundError(f"Missing forecast data file: {fc_path}")
    if not risk_path.exists():
        raise FileNotFoundError(f"Missing risk data file: {risk_path}")

    fc_df = pd.read_csv(fc_path)
    risk_df = pd.read_csv(risk_path)

    summary_meta = {}
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            summary_meta = json.load(f)

    # Index by SKU for O(1) lookup
    fc_by_sku = {}
    for sku, group in fc_df.groupby("SKU"):
        fc_by_sku[sku.upper()] = group.sort_values("Horizon_Step").to_dict(orient="records")

    risk_by_sku = {}
    for _, row in risk_df.iterrows():
        sku_key = str(row["SKU"]).strip().upper()
        risk_by_sku[sku_key] = row.to_dict()

    _DATA_CACHE["fc_df"] = fc_df
    _DATA_CACHE["risk_df"] = risk_df
    _DATA_CACHE["fc_by_sku"] = fc_by_sku
    _DATA_CACHE["risk_by_sku"] = risk_by_sku
    _DATA_CACHE["summary_meta"] = summary_meta


# Load datasets on import/startup
load_datasets()


# =====================================================================
# Pydantic Request & Response Schemas
# =====================================================================

class HealthResponse(BaseModel):
    status: str
    model: str
    model_version: str
    validation_wape: float
    baseline_wape: float
    active_skus_evaluated: int
    data_as_of: str


class WeeklyForecastItem(BaseModel):
    week_number: int
    forecast_week_start: str
    forecast_units: float
    forecast_revenue: float
    interval_p10: float
    interval_p90: float


class ForecastResponse(BaseModel):
    sku: str
    product_name: str
    category: str
    subcategory: str
    forecast_horizon_weeks: int
    forecast_method: str
    total_forecast_units: float
    total_forecast_revenue: float
    weekly_forecast: List[WeeklyForecastItem]


class RiskResponse(BaseModel):
    sku: str
    product_name: str
    category: str
    current_stock: int
    on_order: int
    available_units: int
    lead_time_days: int
    safety_stock: int
    reorder_point: int
    lead_time_demand: float
    required_inventory_buffer: float
    buffer_gap_units: float
    stockout_score: float
    stockout_flag: int
    stockout_risk: bool
    forward_demand_units_4w: float
    excess_units: float
    overstock_score: float
    overstock_flag: int
    overstock_risk: bool
    decision_quadrant: str
    recommended_action: str
    action_rationale: str
    revenue_at_risk: float
    inventory_value_tied_up: float
    value_at_stake: float
    current_stock_below_reorder_point: int
    reorder_gap_units: float


class BatchScoreRequest(BaseModel):
    sku_ids: List[str] = Field(..., description="List of SKU identifiers to evaluate (e.g., ['SKU010', 'SKU012'])")


class BatchScoreResultItem(BaseModel):
    sku: str
    product_name: str
    category: str
    decision_quadrant: str
    stockout_risk: bool
    overstock_risk: bool
    stockout_score: float
    overstock_score: float
    available_units: int
    required_inventory_buffer: float
    buffer_gap_units: float
    revenue_at_risk: float
    inventory_value_tied_up: float
    value_at_stake: float
    recommended_action: str
    total_8w_forecast_units: float
    total_8w_forecast_revenue: float


class BatchErrorItem(BaseModel):
    sku: str
    error: str


class BatchScoreResponse(BaseModel):
    requested_count: int
    scored_count: int
    error_count: int
    results: List[BatchScoreResultItem]
    errors: List[BatchErrorItem]


# =====================================================================
# API Endpoints
# =====================================================================

@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def get_health():
    """
    Check API operational health and inspect model validation metadata.
    """
    return HealthResponse(
        status="healthy",
        model="HistGradientBoostingRegressor",
        model_version="1.0.0",
        validation_wape=0.1056,  # 10.56%
        baseline_wape=0.1117,    # 11.17%
        active_skus_evaluated=len(_DATA_CACHE["risk_by_sku"]),
        data_as_of="2025-12-01"
    )


@app.get("/forecast/{sku_id}", response_model=ForecastResponse, tags=["Forecast"])
def get_forecast(sku_id: str):
    """
    Retrieve the 8-week forward demand forecast and P10/P90 prediction intervals for a specific SKU.
    """
    clean_sku = sku_id.strip().upper()
    fc_records = _DATA_CACHE["fc_by_sku"].get(clean_sku)

    if not fc_records:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"SKU '{sku_id}' not found in active commercial forecasting catalog. Available SKUs: SKU001 through SKU050."
        )

    first = fc_records[0]
    total_units = round(sum(r["Forecast_Units"] for r in fc_records), 1)
    total_rev = round(sum(r["Forecast_Revenue"] for r in fc_records), 2)

    weekly_items = [
        WeeklyForecastItem(
            week_number=int(r["Horizon_Step"]),
            forecast_week_start=str(r["Forecast_Week"]),
            forecast_units=float(r["Forecast_Units"]),
            forecast_revenue=float(r["Forecast_Revenue"]),
            interval_p10=float(r["Interval_P10"]),
            interval_p90=float(r["Interval_P90"])
        )
        for r in fc_records
    ]

    return ForecastResponse(
        sku=clean_sku,
        product_name=str(first["Product_Name"]),
        category=str(first["Category"]),
        subcategory=str(first["Subcategory"]),
        forecast_horizon_weeks=len(fc_records),
        forecast_method=str(first["Forecast_Method"]),
        total_forecast_units=total_units,
        total_forecast_revenue=total_rev,
        weekly_forecast=weekly_items
    )


@app.get("/risk/{sku_id}", response_model=RiskResponse, tags=["Risk"])
def get_risk(sku_id: str):
    """
    Retrieve inventory risk scores, lead-time buffer breach status, decision quadrant, and rupee exposure.
    """
    clean_sku = sku_id.strip().upper()
    r = _DATA_CACHE["risk_by_sku"].get(clean_sku)

    if not r:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"SKU '{sku_id}' not found in active risk catalog. Available SKUs: SKU001 through SKU050."
        )

    return RiskResponse(
        sku=clean_sku,
        product_name=str(r["Product_Name"]),
        category=str(r["Category"]),
        current_stock=int(r["Current_Stock"]),
        on_order=int(r["On_Order"]),
        available_units=int(r["available_units"]),
        lead_time_days=int(r["Lead_Time_Days"]),
        safety_stock=int(r["Safety_Stock"]),
        reorder_point=int(r["Reorder_Point"]),
        lead_time_demand=float(r["lead_time_demand"]),
        required_inventory_buffer=float(r["required_inventory_buffer"]),
        buffer_gap_units=float(r["buffer_gap_units"]),
        stockout_score=float(r["stockout_score"]),
        stockout_flag=int(r["stockout_flag"]),
        stockout_risk=bool(r["stockout_flag"] == 1),
        forward_demand_units_4w=float(r["forward_demand_units"]),
        excess_units=float(r["excess_units"]),
        overstock_score=float(r["overstock_score"]),
        overstock_flag=int(r["overstock_flag"]),
        overstock_risk=bool(r["overstock_flag"] == 1),
        decision_quadrant=str(r["decision_quadrant"]),
        recommended_action=str(r["recommended_action"]),
        action_rationale=str(r["action_rationale"]),
        revenue_at_risk=float(r["revenue_at_risk"]),
        inventory_value_tied_up=float(r["inventory_value_tied_up"]),
        value_at_stake=float(r["value_at_stake"]),
        current_stock_below_reorder_point=int(r["current_stock_below_reorder_point"]),
        reorder_gap_units=float(r["reorder_gap_units"])
    )


@app.post("/score/batch", response_model=BatchScoreResponse, tags=["Scoring"])
def score_batch(request: BatchScoreRequest):
    """
    Score a batch list of SKU IDs. Returns combined forecast and risk metrics for each valid SKU,
    and isolates invalid or uncataloged SKUs in the error response.
    """
    results: List[BatchScoreResultItem] = []
    errors: List[BatchErrorItem] = []

    for raw_sku in request.sku_ids:
        clean_sku = str(raw_sku).strip().upper()
        risk_row = _DATA_CACHE["risk_by_sku"].get(clean_sku)
        fc_list = _DATA_CACHE["fc_by_sku"].get(clean_sku)

        if not risk_row or not fc_list:
            errors.append(BatchErrorItem(
                sku=raw_sku,
                error=f"SKU '{raw_sku}' not found in active commercial catalog."
            ))
            continue

        tot_units = round(sum(f["Forecast_Units"] for f in fc_list), 1)
        tot_rev = round(sum(f["Forecast_Revenue"] for f in fc_list), 2)

        results.append(BatchScoreResultItem(
            sku=clean_sku,
            product_name=str(risk_row["Product_Name"]),
            category=str(risk_row["Category"]),
            decision_quadrant=str(risk_row["decision_quadrant"]),
            stockout_risk=bool(risk_row["stockout_flag"] == 1),
            overstock_risk=bool(risk_row["overstock_flag"] == 1),
            stockout_score=float(risk_row["stockout_score"]),
            overstock_score=float(risk_row["overstock_score"]),
            available_units=int(risk_row["available_units"]),
            required_inventory_buffer=float(risk_row["required_inventory_buffer"]),
            buffer_gap_units=float(risk_row["buffer_gap_units"]),
            revenue_at_risk=float(risk_row["revenue_at_risk"]),
            inventory_value_tied_up=float(risk_row["inventory_value_tied_up"]),
            value_at_stake=float(risk_row["value_at_stake"]),
            recommended_action=str(risk_row["recommended_action"]),
            total_8w_forecast_units=tot_units,
            total_8w_forecast_revenue=tot_rev
        ))

    return BatchScoreResponse(
        requested_count=len(request.sku_ids),
        scored_count=len(results),
        error_count=len(errors),
        results=results,
        errors=errors
    )
