"""
Project FORESIGHT - Demand Forecasting & Rolling-Origin Backtesting (Deliverable D3)
Client: NorthBay Living

This module provides:
1. Data preparation: Standardizes weekly timeline to 104 full 7-day weeks.
2. Seasonal-naive baseline benchmark (52-week seasonal period).
3. Leakage-free feature engineering: Lags (1, 2, 4, 8, 12, 52), rolling stats (4 & 12 weeks), calendar/promo flags.
4. Tree-based regression model (HistGradientBoostingRegressor).
5. Chronological rolling-origin cross-validation over 4 independent 8-week test origins.
6. Honest comparison using global Weighted Absolute Percentage Error (WAPE) and Signed Bias.
7. 8-week multi-step forward demand forecast generation for all 50 active commercial SKUs.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# Structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("FORESIGHT_Forecast")


class DemandForecastingEngine:
    """
    Production forecasting engine implementing seasonal-naive baseline,
    leakage-free tree regression, rolling-origin CV, and forward inference.
    """

    def __init__(
        self,
        processed_dir: str = "data/processed",
        models_dir: str = "src/models",
        figures_dir: str = "reports/figures",
        horizon: int = 8,
        season_period: int = 52
    ):
        self.processed_dir = Path(processed_dir)
        self.models_dir = Path(models_dir)
        self.figures_dir = Path(figures_dir)
        self.horizon = horizon
        self.season_period = season_period
        
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        self.data_path = self.processed_dir / "weekly_demand_analysis_ready.csv"
        self.cat_cols = ["Category", "Subcategory", "Season"]
        self.num_cols = [
            "Year", "Month", "Week_Number",
            "Promo_Days_Cal", "Holiday_Days",
            "Cost_Price", "Selling_Price", "Gross_Margin_Per_Unit",
            "lag_1", "lag_2", "lag_4", "lag_8", "lag_12", "lag_52",
            "rolling_mean_4", "rolling_mean_12", "rolling_std_4", "rolling_std_12",
            "rolling_min_4", "rolling_max_4"
        ]
        self.feature_cols = self.cat_cols + self.num_cols

    def load_and_prepare_weekly_data(self) -> Tuple[pd.DataFrame, List[pd.Timestamp]]:
        """
        Step 1: Load weekly dataset, sort strictly by SKU and Week_Start,
        and resolve the final partial week issue.
        """
        logger.info("Loading weekly dataset from %s...", self.data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Missing processed data: {self.data_path.resolve()}")
            
        df = pd.read_csv(self.data_path)
        df["Week_Start"] = pd.to_datetime(df["Week_Start"])
        df["Launch_Date"] = pd.to_datetime(df["Launch_Date"])
        
        # Sort strictly by SKU and Week_Start
        df = df.sort_values(by=["SKU", "Week_Start"]).reset_index(drop=True)
        
        # Partial week handling: Week 105 (commencing 2025-12-29) contains only 3 days.
        # Standardize to the 104 full 7-day calendar weeks (Jan 1, 2024 to Dec 28, 2025).
        full_weeks_df = df[df["Week_Start"] < "2025-12-29"].copy().reset_index(drop=True)
        unique_weeks = sorted(full_weeks_df["Week_Start"].unique())
        
        logger.info("Dataset standardized to %d full 7-day weeks across %d active SKUs (%d rows)",
                    len(unique_weeks), full_weeks_df["SKU"].nunique(), len(full_weeks_df))
        
        return full_weeks_df, unique_weeks

    @staticmethod
    def engineer_lags_and_rolling(df_input: pd.DataFrame) -> pd.DataFrame:
        """
        Step 2: Construct temporal features with strict past-only information (zero leakage).
        """
        df_feat = df_input.copy()
        df_feat = df_feat.sort_values(by=["SKU", "Week_Start"]).reset_index(drop=True)
        
        # Shifted historical lags
        for lag in [1, 2, 4, 8, 12, 52]:
            df_feat[f"lag_{lag}"] = df_feat.groupby("SKU")["Weekly_Units"].shift(lag)
            
        grouped = df_feat.groupby("SKU")["Weekly_Units"]
        df_feat["rolling_mean_4"] = grouped.apply(lambda s: s.shift(1).rolling(4).mean()).reset_index(level=0, drop=True)
        df_feat["rolling_mean_12"] = grouped.apply(lambda s: s.shift(1).rolling(12).mean()).reset_index(level=0, drop=True)
        df_feat["rolling_std_4"] = grouped.apply(lambda s: s.shift(1).rolling(4).std()).reset_index(level=0, drop=True).fillna(0)
        df_feat["rolling_std_12"] = grouped.apply(lambda s: s.shift(1).rolling(12).std()).reset_index(level=0, drop=True).fillna(0)
        df_feat["rolling_min_4"] = grouped.apply(lambda s: s.shift(1).rolling(4).min()).reset_index(level=0, drop=True)
        df_feat["rolling_max_4"] = grouped.apply(lambda s: s.shift(1).rolling(4).max()).reset_index(level=0, drop=True)
        
        return df_feat

    def run_rolling_origin_backtest(
        self,
        df_clean: pd.DataFrame,
        unique_weeks: List[pd.Timestamp]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Step 3: Execute chronological rolling-origin cross-validation.
        Evaluates both Seasonal-Naive Baseline and HistGradientBoostingRegressor on 4 distinct origins.
        """
        logger.info("Executing rolling-origin cross-validation (Horizon = %d weeks)...", self.horizon)
        
        # 4 non-overlapping test origins (indices 71, 79, 87, 95 in 104-week series)
        # Each origin evaluates 8 future weeks (4 * 8 * 50 = 1,600 predictions)
        origins = [71, 79, 87, 95]
        backtest_records = []
        
        df_with_features = self.engineer_lags_and_rolling(df_clean)
        for col in self.cat_cols:
            df_with_features[col] = df_with_features[col].astype("category")
            
        for orig_idx in origins:
            cutoff_date = unique_weeks[orig_idx]
            test_weeks = unique_weeks[orig_idx + 1 : orig_idx + self.horizon + 1]
            orig_date_str = cutoff_date.strftime("%Y-%m-%d")
            
            logger.info("Evaluating Origin index %d (%s) -> predicting %s to %s",
                        orig_idx, orig_date_str,
                        test_weeks[0].strftime("%Y-%m-%d"),
                        test_weeks[-1].strftime("%Y-%m-%d"))
                        
            # Train set: all observations <= cutoff_date with valid seasonal lag (week_idx >= 52)
            train_mask = (df_with_features["Week_Start"] <= cutoff_date) & df_with_features["lag_52"].notna()
            train_df = df_with_features[train_mask]
            
            X_train = train_df[self.feature_cols]
            y_train = train_df["Weekly_Units"]
            
            # Fit ML Model (strong simple tree ensemble)
            ml_model = HistGradientBoostingRegressor(
                max_iter=150,
                learning_rate=0.08,
                max_leaf_nodes=31,
                min_samples_leaf=20,
                random_state=42,
                categorical_features=self.cat_cols
            )
            ml_model.fit(X_train, y_train)
            
            # Recursive multi-step forward inference
            history_buffer = df_clean[df_clean["Week_Start"] <= cutoff_date].copy()
            
            for h, tw in enumerate(test_weeks, 1):
                target_week_idx = orig_idx + h
                seasonal_week_idx = target_week_idx - self.season_period
                seasonal_date = unique_weeks[seasonal_week_idx]
                
                # Actual observations
                actual_step = df_clean[df_clean["Week_Start"] == tw].copy()
                
                # Baseline prediction: exactly the seasonal lag 52 weeks prior
                baseline_lookup = df_clean[df_clean["Week_Start"] == seasonal_date][["SKU", "Weekly_Units"]].rename(
                    columns={"Weekly_Units": "Pred_Baseline"}
                )
                
                # ML Feature construction for step h (strictly using history_buffer)
                comb = pd.concat([history_buffer, actual_step], ignore_index=True)
                comb_feat = self.engineer_lags_and_rolling(comb)
                curr_test = comb_feat[comb_feat["Week_Start"] == tw].copy()
                for col in self.cat_cols:
                    curr_test[col] = curr_test[col].astype("category")
                    
                X_test = curr_test[self.feature_cols]
                preds_ml = np.clip(ml_model.predict(X_test), 0, None).round(1)
                
                # Merge step results
                step_res = actual_step[["Week_Start", "SKU", "Product_Name", "Category", "Weekly_Units"]].rename(
                    columns={"Weekly_Units": "Actual_Units"}
                )
                step_res["Origin_Index"] = orig_idx
                step_res["Origin_Date"] = orig_date_str
                step_res["Horizon_Step"] = h
                step_res = step_res.merge(baseline_lookup, on="SKU", how="left")
                step_res["Pred_ML"] = preds_ml
                
                backtest_records.append(step_res)
                
                # Append ML prediction to history buffer for next recursive step
                rec_append = actual_step.copy()
                rec_append["Weekly_Units"] = preds_ml
                history_buffer = pd.concat([history_buffer, rec_append], ignore_index=True)

        backtest_df = pd.concat(backtest_records, ignore_index=True)
        
        # Calculate global evaluation metrics
        # WAPE = sum(|Actual - Pred|) / sum(Actual)
        sum_actual = backtest_df["Actual_Units"].sum()
        
        base_wape = float(np.sum(np.abs(backtest_df["Actual_Units"] - backtest_df["Pred_Baseline"])) / sum_actual)
        base_bias = float(np.mean(backtest_df["Pred_Baseline"] - backtest_df["Actual_Units"]))
        base_mae = float(mean_absolute_error(backtest_df["Actual_Units"], backtest_df["Pred_Baseline"]))
        base_rmse = float(root_mean_squared_error(backtest_df["Actual_Units"], backtest_df["Pred_Baseline"]))
        
        ml_wape = float(np.sum(np.abs(backtest_df["Actual_Units"] - backtest_df["Pred_ML"])) / sum_actual)
        ml_bias = float(np.mean(backtest_df["Pred_ML"] - backtest_df["Actual_Units"]))
        ml_mae = float(mean_absolute_error(backtest_df["Actual_Units"], backtest_df["Pred_ML"]))
        ml_rmse = float(root_mean_squared_error(backtest_df["Actual_Units"], backtest_df["Pred_ML"]))
        
        # Per-origin performance table
        origin_summary = []
        for orig_idx in origins:
            sub = backtest_df[backtest_df["Origin_Index"] == orig_idx]
            sub_actual = sub["Actual_Units"].sum()
            origin_summary.append({
                "origin_index": int(orig_idx),
                "origin_date": sub["Origin_Date"].iloc[0],
                "baseline_wape": float(np.sum(np.abs(sub["Actual_Units"] - sub["Pred_Baseline"])) / sub_actual),
                "ml_wape": float(np.sum(np.abs(sub["Actual_Units"] - sub["Pred_ML"])) / sub_actual),
                "baseline_bias": float(np.mean(sub["Pred_Baseline"] - sub["Actual_Units"])),
                "ml_bias": float(np.mean(sub["Pred_ML"] - sub["Actual_Units"]))
            })

        # Model selection logic
        ml_beats_baseline = bool(ml_wape < base_wape)
        selected_model = "HistGradientBoostingRegressor" if ml_beats_baseline else "Seasonal_Naive"
        
        metrics_summary = {
            "backtest_origins_count": len(origins),
            "forecast_horizon_weeks": self.horizon,
            "predictions_evaluated": len(backtest_df),
            "skus_evaluated": backtest_df["SKU"].nunique(),
            "baseline_metrics": {
                "wape": round(base_wape, 4),
                "wape_pct": round(base_wape * 100, 2),
                "bias": round(base_bias, 4),
                "mae": round(base_mae, 2),
                "rmse": round(base_rmse, 2)
            },
            "ml_metrics": {
                "wape": round(ml_wape, 4),
                "wape_pct": round(ml_wape * 100, 2),
                "bias": round(ml_bias, 4),
                "mae": round(ml_mae, 2),
                "rmse": round(ml_rmse, 2)
            },
            "ml_beats_baseline": ml_beats_baseline,
            "selected_final_method": selected_model,
            "origin_breakdown": origin_summary
        }
        
        logger.info("=== BACKTEST EVALUATION SUMMARY ===")
        logger.info("Seasonal-Naive Baseline: WAPE = %.2f%%, Bias = %.2f, MAE = %.2f, RMSE = %.2f",
                    base_wape * 100, base_bias, base_mae, base_rmse)
        logger.info("HistGradientBoosting ML:  WAPE = %.2f%%, Bias = %.2f, MAE = %.2f, RMSE = %.2f",
                    ml_wape * 100, ml_bias, ml_mae, ml_rmse)
        logger.info("Outcome: ML beats Baseline = %s -> Selected Final Method = %s",
                    ml_beats_baseline, selected_model)
                    
        return backtest_df, metrics_summary

    def train_final_production_model(
        self,
        df_clean: pd.DataFrame
    ) -> Tuple[HistGradientBoostingRegressor, np.ndarray]:
        """
        Step 4: Train final model on all available 104 weeks for forward production inference.
        """
        logger.info("Training final production forecasting model on all 104 full weeks...")
        df_feat = self.engineer_lags_and_rolling(df_clean)
        for col in self.cat_cols:
            df_feat[col] = df_feat[col].astype("category")
            
        train_df = df_feat[df_feat["lag_52"].notna()]
        X_train = train_df[self.feature_cols]
        y_train = train_df["Weekly_Units"]
        
        final_model = HistGradientBoostingRegressor(
            max_iter=150,
            learning_rate=0.08,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            random_state=42,
            categorical_features=self.cat_cols
        )
        final_model.fit(X_train, y_train)
        
        # Compute in-sample/backtest residuals for 80% prediction interval estimation
        residuals = y_train.values - final_model.predict(X_train)
        
        # Save model artifact
        model_artifact_path = self.models_dir / "forecast_model.joblib"
        joblib.dump(final_model, model_artifact_path)
        logger.info("Saved trained model artifact to %s", model_artifact_path)
        
        return final_model, residuals

    def generate_future_forecast(
        self,
        df_clean: pd.DataFrame,
        unique_weeks: List[pd.Timestamp],
        final_model: HistGradientBoostingRegressor,
        residuals: np.ndarray,
        selected_method: str
    ) -> pd.DataFrame:
        """
        Step 5: Generate forward 8-week SKU-level demand forecasts starting from production origin.
        """
        last_history_date = unique_weeks[-1]  # 2025-12-22
        logger.info("Generating forward %d-week forecast from production origin %s...",
                    self.horizon, last_history_date.strftime("%Y-%m-%d"))
                    
        # 8 forward Mondays
        future_dates = [last_history_date + pd.Timedelta(days=7 * i) for i in range(1, self.horizon + 1)]
        
        # Empirical 80% prediction interval bounds from residuals
        err_lower = np.percentile(residuals, 10)
        err_upper = np.percentile(residuals, 90)
        
        history_buffer = df_clean.copy()
        sku_meta = df_clean[["SKU", "Product_Name", "Category", "Subcategory", "Selling_Price", "Cost_Price"]].drop_duplicates()
        
        forecast_records = []
        
        for h, f_date in enumerate(future_dates, 1):
            f_date_str = f_date.strftime("%Y-%m-%d")
            
            # Seasonal date for baseline lookup: 52 weeks prior
            seasonal_date = f_date - pd.Timedelta(weeks=52)
            # Match closest available week in 2024/2025
            seasonal_match = df_clean[df_clean["Week_Start"] == seasonal_date]
            if seasonal_match.empty:
                # Fallback to week 52 weeks before last history date
                closest_date = unique_weeks[-52 + (h - 1)]
                seasonal_match = df_clean[df_clean["Week_Start"] == closest_date]
                
            baseline_dict = dict(zip(seasonal_match["SKU"], seasonal_match["Weekly_Units"]))
            
            # Build mock record for future date with calendar attributes
            mock_step = sku_meta.copy()
            mock_step["Week_Start"] = f_date
            mock_step["Year"] = f_date.year
            mock_step["Month"] = f_date.month
            mock_step["Week_Number"] = f_date.isocalendar()[1]
            mock_step["Season"] = "Winter" if f_date.month in [12, 1, 2] else ("Spring" if f_date.month in [3, 4, 5] else "Summer")
            mock_step["Holiday_Days"] = 1 if f_date.month == 1 and f_date.day == 26 else 0
            mock_step["Promo_Days_Cal"] = 0
            mock_step["Weekly_Units"] = 0
            mock_step["Weekly_Revenue"] = 0.0
            mock_step["Avg_Price"] = mock_step["Selling_Price"]
            mock_step["Launch_Date"] = pd.to_datetime("2024-01-01")
            mock_step["Gross_Margin_Per_Unit"] = mock_step["Selling_Price"] - mock_step["Cost_Price"]
            
            # Merge with history buffer to construct lag features
            comb = pd.concat([history_buffer, mock_step], ignore_index=True)
            comb_feat = self.engineer_lags_and_rolling(comb)
            curr_future = comb_feat[comb_feat["Week_Start"] == f_date].copy()
            for col in self.cat_cols:
                curr_future[col] = curr_future[col].astype("category")
                
            X_future = curr_future[self.feature_cols]
            preds_ml = np.clip(final_model.predict(X_future), 0, None).round(1)
            
            # Assemble output rows
            for idx, (_, row) in enumerate(curr_future.iterrows()):
                sku = row["SKU"]
                base_pred = float(baseline_dict.get(sku, preds_ml[idx]))
                ml_pred = float(preds_ml[idx])
                
                # Selected primary forecast
                chosen_forecast = ml_pred if selected_method == "HistGradientBoostingRegressor" else base_pred
                
                p10 = max(0.0, round(chosen_forecast + err_lower, 1))
                p90 = max(0.0, round(chosen_forecast + err_upper, 1))
                
                forecast_records.append({
                    "Forecast_Week": f_date_str,
                    "Horizon_Step": h,
                    "SKU": sku,
                    "Product_Name": row["Product_Name"],
                    "Category": row["Category"],
                    "Subcategory": row["Subcategory"],
                    "Forecast_Units": round(chosen_forecast, 1),
                    "Baseline_Units": round(base_pred, 1),
                    "ML_Units": round(ml_pred, 1),
                    "Interval_P10": p10,
                    "Interval_P90": p90,
                    "Selling_Price": row["Selling_Price"],
                    "Cost_Price": row["Cost_Price"],
                    "Forecast_Revenue": round(chosen_forecast * row["Selling_Price"], 2),
                    "Forecast_Method": selected_method
                })
                
            # Append predictions to buffer for recursive multi-step forecasting
            step_append = mock_step.copy()
            step_append["Weekly_Units"] = preds_ml
            history_buffer = pd.concat([history_buffer, step_append], ignore_index=True)

        forecast_df = pd.DataFrame(forecast_records)
        return forecast_df

    def create_evaluation_visualizations(
        self,
        backtest_df: pd.DataFrame,
        forecast_df: pd.DataFrame,
        df_clean: pd.DataFrame
    ):
        """
        Step 6: Generate clear evaluation and forecast charts for high-priority SKUs.
        """
        logger.info("Generating D3 evaluation and forecasting visualization charts...")
        
        # Representative SKUs: Top Volume (#1 SKU012), Top Revenue (#1 SKU026), Slow Mover (#1 SKU011), Storage Leader (#2 SKU045)
        rep_skus = ["SKU012", "SKU026", "SKU045", "SKU011"]
        
        # Chart 1: Backtest Evaluation Across Origins
        fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=False)
        axes = axes.flatten()
        
        for i, sku in enumerate(rep_skus):
            ax = axes[i]
            sku_backtest = backtest_df[backtest_df["SKU"] == sku].sort_values(by="Week_Start")
            sku_name = sku_backtest["Product_Name"].iloc[0]
            sku_cat = sku_backtest["Category"].iloc[0]
            
            ax.plot(pd.to_datetime(sku_backtest["Week_Start"]), sku_backtest["Actual_Units"],
                    label="Actual Demand", color="#2c3e50", lw=2, marker="o", ms=4)
            ax.plot(pd.to_datetime(sku_backtest["Week_Start"]), sku_backtest["Pred_Baseline"],
                    label="Seasonal-Naive Baseline", color="#e67e22", ls="--", lw=1.5)
            ax.plot(pd.to_datetime(sku_backtest["Week_Start"]), sku_backtest["Pred_ML"],
                    label="HistGradientBoosting ML", color="#27ae60", lw=2)
                    
            ax.set_title(f"{sku} - {sku_name} ({sku_cat}) [Backtest]", fontsize=11, fontweight="bold")
            ax.set_ylabel("Weekly Units")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right", fontsize=9)
            
        plt.tight_layout()
        backtest_fig_path = self.figures_dir / "07_forecast_backtest_eval.png"
        plt.savefig(backtest_fig_path, dpi=300)
        plt.close()
        
        # Chart 2: 8-Week Forward Forecast with Prediction Intervals
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        axes = axes.flatten()
        
        # Recent history (last 16 weeks)
        recent_weeks = sorted(df_clean["Week_Start"].unique())[-16:]
        
        for i, sku in enumerate(rep_skus):
            ax = axes[i]
            sku_hist = df_clean[(df_clean["SKU"] == sku) & (df_clean["Week_Start"].isin(recent_weeks))].sort_values(by="Week_Start")
            sku_fc = forecast_df[forecast_df["SKU"] == sku].sort_values(by="Forecast_Week")
            
            hist_dates = pd.to_datetime(sku_hist["Week_Start"])
            fc_dates = pd.to_datetime(sku_fc["Forecast_Week"])
            
            ax.plot(hist_dates, sku_hist["Weekly_Units"], label="Recent History (Actual)", color="#2c3e50", lw=2, marker="o")
            ax.plot(fc_dates, sku_fc["Forecast_Units"], label="8-Week Forecast (Selected Method)", color="#2980b9", lw=2.5, marker="s")
            ax.plot(fc_dates, sku_fc["Baseline_Units"], label="Seasonal Baseline Reference", color="#f39c12", ls="--", lw=1.5)
            
            # Prediction intervals (80%)
            ax.fill_between(fc_dates, sku_fc["Interval_P10"], sku_fc["Interval_P90"],
                            color="#2980b9", alpha=0.2, label="80% Prediction Interval")
                            
            ax.axvline(hist_dates.iloc[-1], color="red", ls=":", lw=1.5, label="Forecast Cutoff")
            ax.set_title(f"{sku} - {sku_fc['Product_Name'].iloc[0]} (Forward 8-Week Forecast)", fontsize=11, fontweight="bold")
            ax.set_ylabel("Weekly Demand Units")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper left", fontsize=8)
            
        plt.tight_layout()
        forecast_fig_path = self.figures_dir / "08_future_forecast_8weeks.png"
        plt.savefig(forecast_fig_path, dpi=300)
        plt.close()
        
        logger.info("Saved forecast charts to %s and %s", backtest_fig_path, forecast_fig_path)

    def run(self) -> Dict[str, Any]:
        """
        Run the complete D3 demand forecasting pipeline end-to-end.
        """
        logger.info("Starting Project FORESIGHT Forecasting Engine (D3)...")
        
        # Step 1: Load and clean weekly series
        df_clean, unique_weeks = self.load_and_prepare_weekly_data()
        
        # Step 2 & 3: Rolling-origin backtest
        backtest_df, metrics_summary = self.run_rolling_origin_backtest(df_clean, unique_weeks)
        
        # Step 4: Train final model
        final_model, residuals = self.train_final_production_model(df_clean)
        
        # Step 5: Forward inference
        selected_method = metrics_summary["selected_final_method"]
        forecast_df = self.generate_future_forecast(
            df_clean, unique_weeks, final_model, residuals, selected_method
        )
        
        # Step 6: Visualizations
        self.create_evaluation_visualizations(backtest_df, forecast_df, df_clean)
        
        # Step 7: Persist outputs
        backtest_path = self.processed_dir / "backtest_results.csv"
        forecast_path = self.processed_dir / "forecast_results.csv"
        metrics_path = self.processed_dir / "model_metrics.json"
        
        backtest_df.to_csv(backtest_path, index=False)
        forecast_df.to_csv(forecast_path, index=False)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics_summary, f, indent=2)
            
        logger.info("Saved backtest results to %s (%d rows)", backtest_path, len(backtest_df))
        logger.info("Saved 8-week forecast results to %s (%d rows)", forecast_path, len(forecast_df))
        logger.info("Saved model metrics to %s", metrics_path)
        logger.info("Deliverable D3 Forecasting Engine execution COMPLETE.")
        
        return {
            "metrics": metrics_summary,
            "backtest_rows": len(backtest_df),
            "forecast_rows": len(forecast_df)
        }


def main():
    parser = argparse.ArgumentParser(description="Run Project FORESIGHT Forecasting Engine (D3)")
    parser.add_argument("--processed-dir", type=str, default="data/processed")
    parser.add_argument("--models-dir", type=str, default="src/models")
    parser.add_argument("--figures-dir", type=str, default="reports/figures")
    parser.add_argument("--horizon", type=int, default=8)
    args = parser.parse_args()

    engine = DemandForecastingEngine(
        processed_dir=args.processed_dir,
        models_dir=args.models_dir,
        figures_dir=args.figures_dir,
        horizon=args.horizon
    )
    engine.run()


if __name__ == "__main__":
    main()
