"""
Project FORESIGHT - Inventory Risk Scoring & Decisioning Engine (Deliverable D4)
Client: NorthBay Living

This module converts the 8-week SKU demand forecasts and current warehouse inventory
positions into operational stockout and overstock risk scores, maps every active SKU
into the 4 decision quadrants (Reorder Now, Markdown/Clear, Watch/Volatile, Healthy),
and quantifies the rupee value at stake (revenue at risk and inventory value tied up).

Acceptance Criteria Addressed:
1. Scores stockout and overstock risk for every SKU over the forecast horizon.
2. Attaches a concrete recommended action and exact rupee value at stake.
3. Transparent, explainable, causal supply-chain logic (not a black box).
4. Fully reconciles with the 4-quadrant decisioning grid in Section 08 of the brief.
5. Preserves all 200 SKUs for warehouse inventory reconciliation.
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("FORESIGHT_Risk")


class InventoryRiskEngine:
    """
    Production risk scoring engine that evaluates forward demand vs inventory positions
    and quantifies financial exposure.
    """

    def __init__(
        self,
        processed_dir: str = "data/processed",
        figures_dir: str = "reports/figures"
    ):
        self.processed_dir = Path(processed_dir)
        self.figures_dir = Path(figures_dir)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        self.inv_path = self.processed_dir / "latest_inventory_status.csv"
        self.fc_path = self.processed_dir / "forecast_results.csv"

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Step 1: Load latest inventory status and forward forecast outputs.
        """
        logger.info("Loading latest inventory status and forecast results...")
        if not self.inv_path.exists():
            raise FileNotFoundError(f"Missing inventory file: {self.inv_path.resolve()}")
        if not self.fc_path.exists():
            raise FileNotFoundError(f"Missing forecast file: {self.fc_path.resolve()}")

        inv_df = pd.read_csv(self.inv_path)
        fc_df = pd.read_csv(self.fc_path)
        
        logger.info("Loaded inventory: %d SKUs | Forecast: %d records", len(inv_df), len(fc_df))
        return inv_df, fc_df

    def compute_risk_scores(self, inv_df: pd.DataFrame, fc_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Step 2: Calculate lead-time demand, available stock, stockout/overstock scores,
        decision quadrants, and rupee impact.
        """
        logger.info("Computing lead-time demand, inventory gaps, and risk scores...")

        # Separate 50 active commercial SKUs from 150 warehouse-only SKUs
        active_df = inv_df[inv_df["is_active_sales_sku"]].copy().reset_index(drop=True)
        inactive_df = inv_df[~inv_df["is_active_sales_sku"]].copy().reset_index(drop=True)

        # Aggregate forecast horizons per active SKU
        fc_w1 = fc_df[fc_df["Horizon_Step"] == 1].set_index("SKU")["Forecast_Units"]
        fc_w2 = fc_df[fc_df["Horizon_Step"] == 2].set_index("SKU")["Forecast_Units"]
        fc_4w = fc_df[fc_df["Horizon_Step"] <= 4].groupby("SKU")["Forecast_Units"].sum()
        fc_8w = fc_df[fc_df["Horizon_Step"] <= 8].groupby("SKU")["Forecast_Units"].sum()
        fc_avg = fc_df.groupby("SKU")["Forecast_Units"].mean()

        active_df["fc_w1"] = active_df["SKU"].map(fc_w1)
        active_df["fc_w2"] = active_df["SKU"].map(fc_w2)
        active_df["forward_demand_units"] = active_df["SKU"].map(fc_4w).round(1)
        active_df["forward_demand_8w"] = active_df["SKU"].map(fc_8w).round(1)
        active_df["fc_avg_weekly"] = active_df["SKU"].map(fc_avg).round(1)

        # 1. Lead-time demand calculation (dynamically handles lead times of 3 to 14 days)
        def calc_lead_time_demand(row: pd.Series) -> float:
            lt = int(row["Lead_Time_Days"])
            w1 = float(row["fc_w1"])
            w2 = float(row["fc_w2"])
            if lt <= 7:
                return round((w1 / 7.0) * lt, 1)
            else:
                return round(w1 + (w2 / 7.0) * (lt - 7), 1)

        active_df["lead_time_demand"] = active_df.apply(calc_lead_time_demand, axis=1)

        # 2. Available inventory & Required buffer
        active_df["available_units"] = active_df["Current_Stock"] + active_df["On_Order"]
        active_df["required_inventory_buffer"] = (active_df["lead_time_demand"] + active_df["Safety_Stock"]).round(1)
        active_df["required_buffer"] = active_df["required_inventory_buffer"]  # compatibility alias

        # Primary stockout flag (forecast-driven buffer breach)
        # stockout_flag = 1 if available_units < required_inventory_buffer else 0
        active_df["stockout_flag"] = (active_df["available_units"] < active_df["required_inventory_buffer"]).astype(int)

        # Separate explanatory indicators (preserved for operational transparency)
        active_df["current_stock_below_reorder_point"] = (active_df["Current_Stock"] <= active_df["Reorder_Point"]).astype(int)
        active_df["reorder_gap_units"] = (active_df["Reorder_Point"] - active_df["Current_Stock"]).apply(lambda x: max(0.0, round(x, 1)))
        active_df["lead_time_stock_gap"] = (active_df["lead_time_demand"] - active_df["available_units"]).apply(lambda x: max(0.0, round(x, 1)))
        active_df["safety_stock_gap"] = (active_df["Safety_Stock"] - active_df["available_units"]).apply(lambda x: max(0.0, round(x, 1)))
        active_df["buffer_gap_units"] = (active_df["required_inventory_buffer"] - active_df["available_units"]).apply(lambda x: max(0.0, round(x, 1)))
        active_df["stock_gap_units"] = active_df["buffer_gap_units"]  # compatibility alias

        # 3. Explainable continuous monotonic Stockout Risk Score [0.0, 1.0]
        # Score increases monotonically as available inventory falls below required buffer.
        # Score = 1.0 - 0.5 * (available_units / required_inventory_buffer), clipped to [0.0, 1.0].
        # Score > 0.50 strictly coincides with available_units < required_inventory_buffer (boundary score = 0.50).
        active_df["stockout_score"] = np.clip(
            1.0 - 0.5 * (active_df["available_units"] / active_df["required_inventory_buffer"]),
            0.0, 1.0
        ).round(3)

        # 4. Overstock Risk Score & Flag (Forecast-driven 4-week methodology)
        active_df["excess_units"] = np.where(
            active_df["Current_Stock"] > active_df["forward_demand_units"],
            (active_df["Current_Stock"] - active_df["forward_demand_units"]).round(1),
            0.0
        )
        active_df["overstock_score"] = np.clip(
            0.5 * (active_df["Current_Stock"] / active_df["forward_demand_units"]),
            0.0, 1.0
        ).round(3)
        active_df["overstock_flag"] = (active_df["Current_Stock"] > active_df["forward_demand_units"]).astype(int)

        # 5. Four Decision Quadrants mapping
        def assign_quadrant(row: pd.Series) -> str:
            so = row["stockout_flag"]
            os = row["overstock_flag"]
            if so == 1 and os == 0:
                return "Reorder Now"
            elif so == 0 and os == 1:
                return "Markdown / Clear"
            elif so == 1 and os == 1:
                return "Watch / Volatile"
            else:
                return "Healthy"

        active_df["decision_quadrant"] = active_df.apply(assign_quadrant, axis=1)

        # 6. Action and Operational Rationale
        def assign_recommendation(row: pd.Series) -> Tuple[str, str]:
            q = row["decision_quadrant"]
            if q == "Reorder Now":
                return (
                    "Prioritize replenishment purchase order review",
                    f"Available inventory ({row['available_units']}) breaches required buffer ({row['required_inventory_buffer']} = {row['lead_time_demand']} LTD + {row['Safety_Stock']} SS); buffer gap is {row['buffer_gap_units']} units."
                )
            elif q == "Markdown / Clear":
                return (
                    "Review clearance and promotional markdown strategy",
                    f"Holding {row['Current_Stock']} units vs 4-week forecast demand of {row['forward_demand_units']}; {row['excess_units']} excess units."
                )
            elif q == "Watch / Volatile":
                return (
                    "Investigate demand volatility and inventory positioning",
                    f"Buffer breach stockout risk coexists with excess holding above 4-week forecast demand."
                )
            else:
                return (
                    "Maintain current inventory position and monitor",
                    f"Available inventory ({row['available_units']}) safely covers required buffer ({row['required_inventory_buffer']}) and stock is within 4-week demand."
                )

        rec_tuples = active_df.apply(assign_recommendation, axis=1)
        active_df["recommended_action"] = [t[0] for t in rec_tuples]
        active_df["action_rationale"] = [t[1] for t in rec_tuples]

        # 7. Rupee Impact Calculations
        # Stockout exposure: Revenue at risk = units_at_stockout_risk * Selling_Price
        # Note: revenue_at_risk is an exposure estimate, NOT guaranteed lost revenue.
        # Overstock exposure: Inventory value tied up = excess_units * Cost_Price
        # Note: inventory_value_tied_up is capital tied up, NOT guaranteed loss.
        # Value at stake: Decision-support exposure metric (revenue_at_risk + inventory_value_tied_up).
        active_df["units_at_stockout_risk"] = np.where(
            active_df["stockout_flag"] == 1,
            active_df["buffer_gap_units"],
            0.0
        )
        active_df["revenue_at_risk"] = (active_df["units_at_stockout_risk"] * active_df["Selling_Price"]).round(2)
        active_df["inventory_value_tied_up"] = (active_df["excess_units"] * active_df["Cost_Price"]).round(2)
        active_df["value_at_stake"] = (active_df["revenue_at_risk"] + active_df["inventory_value_tied_up"]).round(2)

        # 8. Reconciled Full 200-SKU Decision Grid
        # Ensure 150 warehouse-only SKUs are preserved with explicit status
        inactive_df["lead_time_demand"] = 0.0
        inactive_df["available_units"] = inactive_df["Current_Stock"] + inactive_df["On_Order"]
        inactive_df["required_inventory_buffer"] = inactive_df["Safety_Stock"].astype(float).round(1)
        inactive_df["required_buffer"] = inactive_df["required_inventory_buffer"]
        inactive_df["current_stock_below_reorder_point"] = 0
        inactive_df["reorder_gap_units"] = 0.0
        inactive_df["lead_time_stock_gap"] = 0.0
        inactive_df["safety_stock_gap"] = 0.0
        inactive_df["buffer_gap_units"] = 0.0
        inactive_df["stock_gap_units"] = 0.0
        inactive_df["units_at_stockout_risk"] = 0.0
        inactive_df["stockout_score"] = 0.0
        inactive_df["stockout_flag"] = 0
        inactive_df["forward_demand_units"] = 0.0
        inactive_df["forward_demand_8w"] = 0.0
        inactive_df["fc_avg_weekly"] = 0.0
        inactive_df["excess_units"] = inactive_df["Current_Stock"].astype(float)
        inactive_df["overstock_score"] = 1.0  # 100% overstock (0 sales)
        inactive_df["overstock_flag"] = 1
        inactive_df["decision_quadrant"] = "Warehouse Only (Inactive Catalog)"
        inactive_df["recommended_action"] = "Freeze purchase orders and audit for catalog launch or bulk liquidation"
        inactive_df["action_rationale"] = "No commercial sales records in 2 years; 100% of warehouse stock is idle capital."
        inactive_df["revenue_at_risk"] = 0.0
        inactive_df["inventory_value_tied_up"] = inactive_df["Inventory_Value"]
        inactive_df["value_at_stake"] = inactive_df["Inventory_Value"]
        inactive_df["fc_w1"] = 0.0
        inactive_df["fc_w2"] = 0.0

        full_grid_df = pd.concat([active_df, inactive_df], ignore_index=True)

        # Summary Metrics
        q_counts = active_df["decision_quadrant"].value_counts().to_dict()
        q_percentages = {k: round(v / len(active_df) * 100, 1) for k, v in q_counts.items()}

        summary_metrics = {
            "active_skus_evaluated": len(active_df),
            "total_warehouse_skus_reconciled": len(full_grid_df),
            "quadrant_counts": q_counts,
            "quadrant_percentages": q_percentages,
            "stockout_risk_count": int(active_df["stockout_flag"].sum()),
            "stockout_risk_pct": round(float(active_df["stockout_flag"].sum() / len(active_df) * 100), 1),
            "total_units_at_stockout_risk": float(round(active_df["units_at_stockout_risk"].sum(), 1)),
            "total_revenue_at_risk": float(round(active_df["revenue_at_risk"].sum(), 2)),
            "overstock_risk_count": int(active_df["overstock_flag"].sum()),
            "overstock_risk_pct": round(float(active_df["overstock_flag"].sum() / len(active_df) * 100), 1),
            "total_excess_inventory_units": float(round(active_df["excess_units"].sum(), 1)),
            "total_active_inventory_value_tied_up": float(round(active_df["inventory_value_tied_up"].sum(), 2)),
            "total_value_at_stake": float(round(active_df["value_at_stake"].sum(), 2)),
            "inactive_warehouse_value_tied_up": float(round(inactive_df["Inventory_Value"].sum(), 2)),
            "top_stockout_skus": active_df[active_df["stockout_flag"] == 1].sort_values(by="revenue_at_risk", ascending=False)[
                ["SKU", "Product_Name", "revenue_at_risk", "units_at_stockout_risk", "decision_quadrant"]
            ].head(6).to_dict(orient="records"),
            "top_overstock_skus": active_df[active_df["overstock_flag"] == 1].sort_values(by="inventory_value_tied_up", ascending=False)[
                ["SKU", "Product_Name", "inventory_value_tied_up", "excess_units", "decision_quadrant"]
            ].head(5).to_dict(orient="records"),
            "top_value_at_stake_skus": active_df.sort_values(by="value_at_stake", ascending=False)[
                ["SKU", "Product_Name", "value_at_stake", "decision_quadrant"]
            ].head(5).to_dict(orient="records")
        }

        return active_df, full_grid_df, summary_metrics

    def create_decision_grid_chart(self, active_df: pd.DataFrame):
        """
        Step 3: Generate the official 2D stockout-vs-overstock decision grid chart (Figure 6 in brief).
        """
        logger.info("Generating decisioning grid scatter/bubble chart...")
        
        fig, ax = plt.subplots(figsize=(12, 9))
        
        quad_colors = {
            "Reorder Now": "#e74c3c",      # Red
            "Markdown / Clear": "#3498db",  # Blue
            "Watch / Volatile": "#f39c12",  # Amber/Orange
            "Healthy": "#2ecc71"           # Green
        }
        
        # Quadrant backgrounds
        ax.axhspan(0.5, 1.0, 0.0, 0.5, color="#e74c3c", alpha=0.08)  # Reorder Now
        ax.axhspan(0.0, 0.5, 0.5, 1.0, color="#3498db", alpha=0.08)  # Markdown/Clear
        ax.axhspan(0.5, 1.0, 0.5, 1.0, color="#f39c12", alpha=0.08)  # Watch/Volatile
        ax.axhspan(0.0, 0.5, 0.0, 0.5, color="#2ecc71", alpha=0.08)  # Healthy
        
        # Quadrant labels
        ax.text(0.12, 0.94, "REORDER NOW\n(High Stockout, Low Overstock)", color="#c0392b", fontsize=11, fontweight="bold", ha="center")
        ax.text(0.85, 0.94, "WATCH / VOLATILE\n(High on Both - Erratic Demand)", color="#d35400", fontsize=11, fontweight="bold", ha="center")
        ax.text(0.85, 0.06, "MARKDOWN / CLEAR\n(High Overstock, Low Stockout)", color="#2980b9", fontsize=11, fontweight="bold", ha="center")
        ax.text(0.12, 0.06, "HEALTHY\n(Low on Both - Balanced)", color="#27ae60", fontsize=11, fontweight="bold", ha="center")
        
        # Dividers
        ax.axhline(0.5, color="#7f8c8d", lw=1.2, ls="--")
        ax.axvline(0.5, color="#7f8c8d", lw=1.2, ls="--")
        
        # Bubble plot
        for q_name, color in quad_colors.items():
            sub = active_df[active_df["decision_quadrant"] == q_name]
            # Size proportional to value_at_stake
            sizes = np.clip(sub["value_at_stake"] / 15000, 60, 900)
            ax.scatter(
                sub["overstock_score"],
                sub["stockout_score"],
                s=sizes,
                c=color,
                label=f"{q_name} ({len(sub)} SKUs)",
                alpha=0.75,
                edgecolors="#2c3e50",
                linewidth=0.8
            )
            
            # Annotate top revenue SKUs
            for _, r in sub.sort_values(by="value_at_stake", ascending=False).head(2).iterrows():
                ax.annotate(
                    f"{r['SKU']}\n(₹{r['value_at_stake']/1e5:.1f}L)",
                    (r["overstock_score"], r["stockout_score"]),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                    fontsize=8,
                    fontweight="bold"
                )

        ax.set_xlim(-0.02, 1.05)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlabel("Overstock Risk Score [0.0 = Low, 1.0 = High]", fontsize=12, fontweight="bold", labelpad=8)
        ax.set_ylabel("Stockout Risk Score [0.0 = Low, 1.0 = High]", fontsize=12, fontweight="bold", labelpad=8)
        ax.set_title("Project FORESIGHT: 4-Quadrant Inventory Decisioning Grid\n(Bubble size = Rupee Value at Stake)", fontsize=13, fontweight="bold", pad=12)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=4, frameon=True, fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        chart_path = self.figures_dir / "09_risk_decisioning_grid.png"
        plt.savefig(chart_path, dpi=300)
        plt.close()
        
        logger.info("Saved decision grid chart to %s", chart_path)

    def validate_and_save(
        self,
        active_df: pd.DataFrame,
        full_grid_df: pd.DataFrame,
        summary_metrics: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Step 4: Execute QA assertions and persist outputs.
        """
        logger.info("Running QA validation assertions on D4 outputs...")
        
        # Validation checks
        assert len(active_df) == 50, f"Expected 50 active SKUs, got {len(active_df)}"
        assert len(full_grid_df) == 200, f"Expected 200 reconciled SKUs, got {len(full_grid_df)}"
        assert active_df["SKU"].duplicated().sum() == 0, "Duplicate active SKUs found!"
        assert full_grid_df["SKU"].duplicated().sum() == 0, "Duplicate SKUs in decision grid!"
        
        # Quadrant reconciliation
        valid_quadrants = {"Reorder Now", "Markdown / Clear", "Watch / Volatile", "Healthy"}
        assert set(active_df["decision_quadrant"]).issubset(valid_quadrants), "Invalid active quadrant found!"
        assert sum(summary_metrics["quadrant_counts"].values()) == 50, "Quadrant counts do not reconcile to 50!"
        assert active_df["decision_quadrant"].notnull().all(), "Null quadrant detected!"

        # Supply-chain logic assertions
        # 1. On_Order incorporated in available inventory
        assert ((active_df["Current_Stock"] + active_df["On_Order"]) == active_df["available_units"]).all(), "Available units must equal Current_Stock + On_Order!"
        # 2. Safety_Stock incorporated in required buffer
        assert (np.isclose(active_df["required_inventory_buffer"], active_df["lead_time_demand"] + active_df["Safety_Stock"], atol=0.15)).all(), "Required buffer must equal lead_time_demand + Safety_Stock!"
        # 3. Stockout flag consistency with buffer breach
        assert ((active_df["available_units"] < active_df["required_inventory_buffer"]) == (active_df["stockout_flag"] == 1)).all(), "Stockout flag must strictly equal (available_units < required_inventory_buffer)!"
        # 4. Stockout score threshold consistency (score > 0.50 coincides with stockout_flag == 1)
        assert ((active_df["stockout_score"] > 0.50) == (active_df["stockout_flag"] == 1)).all(), "Stockout score > 0.50 must coincide with stockout_flag == 1!"
        # 5. Stockout units at risk derive from buffer gap
        assert (active_df.loc[active_df["stockout_flag"] == 1, "units_at_stockout_risk"] == active_df.loc[active_df["stockout_flag"] == 1, "buffer_gap_units"]).all(), "Units at stockout risk must equal buffer_gap_units for flagged SKUs!"
        assert (active_df.loc[active_df["stockout_flag"] == 0, "units_at_stockout_risk"] == 0.0).all(), "Units at stockout risk must be 0 for unflagged SKUs!"
        # 6. Monetary calculations reconcile
        assert np.isclose(active_df["revenue_at_risk"].sum(), (active_df["units_at_stockout_risk"] * active_df["Selling_Price"]).sum(), atol=1.0), "Revenue at risk does not reconcile!"
        assert np.isclose(active_df["inventory_value_tied_up"].sum(), (active_df["excess_units"] * active_df["Cost_Price"]).sum(), atol=1.0), "Inventory value tied up does not reconcile!"
        assert np.isclose(active_df["value_at_stake"].sum(), (active_df["revenue_at_risk"] + active_df["inventory_value_tied_up"]).sum(), atol=1.0), "Value at stake does not reconcile!"

        required_cols = [
            "SKU", "Product_Name", "Category", "Current_Stock", "On_Order",
            "Lead_Time_Days", "Safety_Stock", "Reorder_Point", "lead_time_demand",
            "available_units", "required_inventory_buffer",
            "current_stock_below_reorder_point", "reorder_gap_units",
            "lead_time_stock_gap", "safety_stock_gap", "buffer_gap_units",
            "units_at_stockout_risk", "stockout_score", "stockout_flag",
            "forward_demand_units", "excess_units", "overstock_score", "overstock_flag",
            "decision_quadrant", "recommended_action", "action_rationale",
            "revenue_at_risk", "inventory_value_tied_up", "value_at_stake"
        ]
        
        for col in required_cols:
            assert col in active_df.columns, f"Missing required column: {col}"
            assert active_df[col].isnull().sum() == 0, f"Unexpected nulls in column {col}"
            
        assert (active_df["revenue_at_risk"] >= 0).all(), "Negative revenue_at_risk detected!"
        assert (active_df["inventory_value_tied_up"] >= 0).all(), "Negative inventory_value_tied_up detected!"
        
        logger.info("All QA assertions passed successfully.")
        
        # Persist CSVs and JSON
        risk_scores_path = self.processed_dir / "risk_scores.csv"
        decision_grid_path = self.processed_dir / "decision_grid.csv"
        summary_path = self.processed_dir / "risk_summary.json"
        
        active_df[required_cols].to_csv(risk_scores_path, index=False)
        full_grid_df.to_csv(decision_grid_path, index=False)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_metrics, f, indent=2)
            
        logger.info("Saved active risk scores to %s (50 SKUs)", risk_scores_path)
        logger.info("Saved full decision grid to %s (200 SKUs)", decision_grid_path)
        logger.info("Saved risk summary metrics to %s", summary_path)
        
        return {
            "risk_scores": str(risk_scores_path.resolve()),
            "decision_grid": str(decision_grid_path.resolve()),
            "risk_summary": str(summary_path.resolve())
        }

    def run(self) -> Dict[str, Any]:
        """
        Run the complete D4 Risk Scoring Engine end-to-end.
        """
        logger.info("Starting Project FORESIGHT Risk Scoring Engine (D4)...")
        inv_df, fc_df = self.load_data()
        active_df, full_grid_df, summary_metrics = self.compute_risk_scores(inv_df, fc_df)
        self.create_decision_grid_chart(active_df)
        output_files = self.validate_and_save(active_df, full_grid_df, summary_metrics)
        
        logger.info("Risk Scoring Engine (D4) execution COMPLETE.")
        return {
            "outputs": output_files,
            "summary": summary_metrics
        }


def main():
    parser = argparse.ArgumentParser(description="Run Project FORESIGHT Inventory Risk Scoring Engine (D4)")
    parser.add_argument("--processed-dir", type=str, default="data/processed")
    parser.add_argument("--figures-dir", type=str, default="reports/figures")
    args = parser.parse_args()

    engine = InventoryRiskEngine(
        processed_dir=args.processed_dir,
        figures_dir=args.figures_dir
    )
    engine.run()


if __name__ == "__main__":
    main()
