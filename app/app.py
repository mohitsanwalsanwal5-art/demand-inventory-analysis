"""
Project FORESIGHT - Inventory Intelligence & Planning Dashboard (Deliverable D5)
Client: NorthBay Living

Executive decision-support dashboard designed for non-technical stakeholders:
- Head of Operations
- Merchandising Team
- Finance Leadership

Provides interactive 4-quadrant risk navigation, 8-week demand projections with
prediction intervals, prioritized action queues, and explainable supply-chain rationale.
"""

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Project FORESIGHT | Inventory Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Determine project root path (repository-relative)
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"


@st.cache_data(ttl=3600)
def load_all_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """Load all processed data assets into memory with caching."""
    risk_path = DATA_DIR / "risk_scores.csv"
    fc_path = DATA_DIR / "forecast_results.csv"
    hist_path = DATA_DIR / "weekly_demand_analysis_ready.csv"
    grid_path = DATA_DIR / "decision_grid.csv"
    summary_path = DATA_DIR / "risk_summary.json"

    if not risk_path.exists():
        st.error(f"Missing risk scores file: {risk_path}")
        st.stop()
    if not fc_path.exists():
        st.error(f"Missing forecast file: {fc_path}")
        st.stop()
    if not hist_path.exists():
        st.error(f"Missing historical data file: {hist_path}")
        st.stop()

    risk_df = pd.read_csv(risk_path)
    fc_df = pd.read_csv(fc_path)
    hist_df = pd.read_csv(hist_path)
    grid_df = pd.read_csv(grid_path) if grid_path.exists() else pd.DataFrame()

    summary_meta = {}
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            summary_meta = json.load(f)

    # Convert dates
    fc_df["Forecast_Week"] = pd.to_datetime(fc_df["Forecast_Week"])
    hist_df["Week_Start"] = pd.to_datetime(hist_df["Week_Start"])

    return risk_df, fc_df, hist_df, grid_df, summary_meta


# Load datasets
risk_df, fc_df, hist_df, grid_df, summary_meta = load_all_data()

# =====================================================================
# SIDEBAR & FILTERS
# =====================================================================
st.sidebar.title("📦 Project FORESIGHT")
st.sidebar.caption("Executive Inventory & Demand Intelligence\nClient: **NorthBay Living**")
st.sidebar.markdown("---")

st.sidebar.subheader("🔍 Decision Filters")

# 1. Category Filter
all_categories = sorted(risk_df["Category"].dropna().unique().tolist())
selected_categories = st.sidebar.multiselect(
    "Product Category",
    options=all_categories,
    default=all_categories,
    help="Filter active SKUs by merchandising category"
)

# 2. Decision Quadrant Filter
all_quadrants = ["Reorder Now", "Markdown / Clear", "Watch / Volatile", "Healthy"]
selected_quadrants = st.sidebar.multiselect(
    "Decision Quadrant",
    options=all_quadrants,
    default=all_quadrants,
    help="Filter by inventory decisioning quadrant"
)

# 3. Stockout Risk Filter
stockout_filter = st.sidebar.selectbox(
    "Stockout Risk",
    options=["All Active SKUs", "At Risk Only (Buffer Breach)", "Adequately Buffered"],
    index=0
)

# 4. Overstock Risk Filter
overstock_filter = st.sidebar.selectbox(
    "Overstock Risk",
    options=["All Active SKUs", "Overstocked (>4-Wk Supply)", "Within 4-Wk Demand"],
    index=0
)

# 5. SKU Search / Direct Filter
sku_list = ["(All SKUs)"] + sorted(risk_df["SKU"].unique().tolist())
selected_sku_filter = st.sidebar.selectbox(
    "Filter by SKU Identifier",
    options=sku_list,
    index=0
)

# Apply filters
filtered_risk = risk_df.copy()

if selected_categories:
    filtered_risk = filtered_risk[filtered_risk["Category"].isin(selected_categories)]
else:
    filtered_risk = filtered_risk.iloc[0:0]

if selected_quadrants:
    filtered_risk = filtered_risk[filtered_risk["decision_quadrant"].isin(selected_quadrants)]
else:
    filtered_risk = filtered_risk.iloc[0:0]

if stockout_filter == "At Risk Only (Buffer Breach)":
    filtered_risk = filtered_risk[filtered_risk["stockout_flag"] == 1]
elif stockout_filter == "Adequately Buffered":
    filtered_risk = filtered_risk[filtered_risk["stockout_flag"] == 0]

if overstock_filter == "Overstocked (>4-Wk Supply)":
    filtered_risk = filtered_risk[filtered_risk["overstock_flag"] == 1]
elif overstock_filter == "Within 4-Wk Demand":
    filtered_risk = filtered_risk[filtered_risk["overstock_flag"] == 0]

if selected_sku_filter != "(All SKUs)":
    filtered_risk = filtered_risk[filtered_risk["SKU"] == selected_sku_filter]

# Sidebar About Section
st.sidebar.markdown("---")
st.sidebar.subheader("ℹ️ System Metadata")
st.sidebar.markdown(
    """
    **Client:** NorthBay Living  
    **Role:** Data Scientist & Inventory Analytics  
    **Forecast Model:** `HistGradientBoostingRegressor`  
    **Horizon:** 8-Week Forward Rolling  
    **Validation WAPE:** **10.56%** *(beats 11.17% seasonal-naive baseline)*  
    **Backtest Design:** 4 Chronological Rolling Origins  
    **Catalog Universe:** 50 Active Commercial SKUs  
    *(150 warehouse-only lines reconciled)*  
    
    *Notice: Exposure values represent decision-support estimates, not guaranteed accounting write-offs or lost sales.*
    """
)

# =====================================================================
# SECTION 1: EXECUTIVE SUMMARY & KPI CARDS
# =====================================================================
st.title("Project FORESIGHT: Inventory Planning Dashboard")
st.markdown(
    """
    *Dynamic, forecast-driven inventory decisioning connecting machine learning demand forecasts,
    supplier lead-time buffers, and working capital exposure.*
    """
)

# Display KPI banner dynamically from filtered data
total_active = len(filtered_risk)
stockout_count = int(filtered_risk["stockout_flag"].sum()) if total_active > 0 else 0
overstock_count = int(filtered_risk["overstock_flag"].sum()) if total_active > 0 else 0
rev_at_risk = float(filtered_risk["revenue_at_risk"].sum()) if total_active > 0 else 0.0
cap_tied_up = float(filtered_risk["inventory_value_tied_up"].sum()) if total_active > 0 else 0.0
tot_stake = float(filtered_risk["value_at_stake"].sum()) if total_active > 0 else 0.0

col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.metric(
        label="Active SKUs Scored",
        value=f"{total_active}",
        delta=f"{total_active / len(risk_df) * 100:.0f}% of Catalog" if len(risk_df) > 0 else "0%"
    )

with col2:
    st.metric(
        label="Stockout Risk SKUs",
        value=f"{stockout_count}",
        delta=f"{stockout_count / total_active * 100:.1f}% rate" if total_active > 0 else "0%",
        delta_color="inverse"
    )

with col3:
    st.metric(
        label="Overstock Risk SKUs",
        value=f"{overstock_count}",
        delta=f"{overstock_count / total_active * 100:.1f}% rate" if total_active > 0 else "0%",
        delta_color="inverse"
    )

with col4:
    st.metric(
        label="Revenue at Risk",
        value=f"₹{rev_at_risk / 1e5:.2f} L",
        help="Estimated gross unfulfilled customer sales during supplier replenishment if POs are delayed."
    )

with col5:
    st.metric(
        label="Capital Tied Up",
        value=f"₹{cap_tied_up / 1e5:.2f} L",
        help="Working capital locked in on-hand inventory exceeding the 4-week operating demand window."
    )

with col6:
    st.metric(
        label="Total Value at Stake",
        value=f"₹{tot_stake / 1e7:.2f} Cr" if tot_stake >= 1e7 else f"₹{tot_stake / 1e5:.2f} L",
        help="Combined exposure metric (Revenue at Risk + Inventory Capital Tied Up) prioritizing managerial focus."
    )

st.markdown("---")

# =====================================================================
# SECTION 2 & 3: INVENTORY RISK MATRIX (4-QUADRANT VISUALIZATION)
# =====================================================================
st.subheader("🎯 Inventory Decisioning Matrix (4-Quadrant Grid)")

if total_active == 0:
    st.warning("⚠️ No active SKUs match the current filter selection. Please broaden your filters in the sidebar.")
else:
    # Color palette
    color_map = {
        "Reorder Now": "#EF4444",       # Red
        "Markdown / Clear": "#3B82F6",   # Blue
        "Watch / Volatile": "#F59E0B",   # Amber
        "Healthy": "#10B981"            # Green
    }

    fig_quad = go.Figure()

    # Background quadrant shading
    # 1. Reorder Now (x: 0 to 0.5, y: 0.5 to 1.05)
    fig_quad.add_shape(
        type="rect", x0=-0.02, y0=0.5, x1=0.5, y1=1.05,
        fillcolor="rgba(239, 68, 68, 0.08)", line=dict(width=0), layer="below"
    )
    # 2. Markdown / Clear (x: 0.5 to 1.05, y: -0.02 to 0.5)
    fig_quad.add_shape(
        type="rect", x0=0.5, y0=-0.02, x1=1.05, y1=0.5,
        fillcolor="rgba(59, 130, 246, 0.08)", line=dict(width=0), layer="below"
    )
    # 3. Watch / Volatile (x: 0.5 to 1.05, y: 0.5 to 1.05)
    fig_quad.add_shape(
        type="rect", x0=0.5, y0=0.5, x1=1.05, y1=1.05,
        fillcolor="rgba(245, 158, 11, 0.08)", line=dict(width=0), layer="below"
    )
    # 4. Healthy (x: -0.02 to 0.5, y: -0.02 to 0.5)
    fig_quad.add_shape(
        type="rect", x0=-0.02, y0=-0.02, x1=0.5, y1=0.5,
        fillcolor="rgba(16, 185, 129, 0.08)", line=dict(width=0), layer="below"
    )

    # Dividing threshold lines
    fig_quad.add_shape(type="line", x0=0.5, y0=-0.02, x1=0.5, y1=1.05, line=dict(color="#94A3B8", dash="dash", width=1.5))
    fig_quad.add_shape(type="line", x0=-0.02, y0=0.5, x1=1.05, y1=0.5, line=dict(color="#94A3B8", dash="dash", width=1.5))

    # Quadrant Text Banners
    fig_quad.add_annotation(x=0.15, y=0.98, text="🔴 <b>REORDER NOW</b><br>Buffer Breach | Replenish", showarrow=False, font=dict(color="#DC2626", size=12))
    fig_quad.add_annotation(x=0.85, y=0.98, text="🟠 <b>WATCH / VOLATILE</b><br>High Buffer Gap & Holdings", showarrow=False, font=dict(color="#D97706", size=12))
    fig_quad.add_annotation(x=0.85, y=0.04, text="🔵 <b>MARKDOWN / CLEAR</b><br>Excess Stock > 4-Wk Demand", showarrow=False, font=dict(color="#2563EB", size=12))
    fig_quad.add_annotation(x=0.15, y=0.04, text="🟢 <b>HEALTHY</b><br>Balanced Inventory Buffer", showarrow=False, font=dict(color="#059669", size=12))

    # Add SKU scatter points grouped by quadrant
    for q_name in ["Reorder Now", "Markdown / Clear", "Watch / Volatile", "Healthy"]:
        q_subset = filtered_risk[filtered_risk["decision_quadrant"] == q_name]
        if len(q_subset) == 0:
            continue

        # Bubble size proportional to Value at Stake (clipped for clean visualization)
        marker_sizes = np.clip(np.sqrt(q_subset["value_at_stake"] + 1000) / 12, 10, 38)

        custom_text = [
            f"<b>{row['SKU']}</b> - {row['Product_Name']}<br>"
            f"Category: {row['Category']}<br>"
            f"Quadrant: <b>{row['decision_quadrant']}</b><br>"
            f"Stockout Score: {row['stockout_score']:.3f} | Overstock Score: {row['overstock_score']:.3f}<br>"
            f"Available: {row['available_units']} (Stock: {row['Current_Stock']}, On Order: {row['On_Order']})<br>"
            f"Required Buffer: {row['required_inventory_buffer']} (LTD: {row['lead_time_demand']} + SS: {row['Safety_Stock']})<br>"
            f"Buffer Deficit: {row['buffer_gap_units']} units<br>"
            f"Revenue at Risk: ₹{row['revenue_at_risk']:,.2f}<br>"
            f"Capital Tied Up: ₹{row['inventory_value_tied_up']:,.2f}<br>"
            f"Value at Stake: ₹{row['value_at_stake']:,.2f}"
            for _, row in q_subset.iterrows()
        ]

        fig_quad.add_trace(go.Scatter(
            x=q_subset["overstock_score"],
            y=q_subset["stockout_score"],
            mode="markers+text",
            name=f"{q_name} ({len(q_subset)})",
            text=q_subset["SKU"],
            textposition="top center",
            textfont=dict(size=9, color="#E2E8F0"),
            hoverinfo="text",
            hovertext=custom_text,
            marker=dict(
                size=marker_sizes,
                color=color_map.get(q_name, "#94A3B8"),
                opacity=0.85,
                line=dict(color="#FFFFFF", width=1)
            )
        ))

    fig_quad.update_layout(
        title="<b>Four-Quadrant Inventory Position Matrix</b> (Bubble Size = Rupee Value at Stake)",
        xaxis=dict(title="<b>Overstock Risk Score</b> [0.0 = Balanced, 1.0 = Excess Holdings]", range=[-0.02, 1.05], dtick=0.2),
        yaxis=dict(title="<b>Stockout Risk Score</b> [0.0 = Well-Buffered, 1.0 = Buffer Breach]", range=[-0.02, 1.05], dtick=0.2),
        height=540,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_dark"
    )

    st.plotly_chart(fig_quad, use_container_width=True)

# =====================================================================
# SECTION 4: PRIORITIZED ACTION TABLES
# =====================================================================
st.markdown("---")
st.subheader("📋 Prioritized Operational Action Queues")

tab_stockout, tab_overstock, tab_all = st.tabs([
    "🚨 Reorder Priority (Stockout Buffer Breach)",
    "🏷️ Clearance Priority (Excess Working Capital)",
    "📑 Filtered Active Catalog (50 SKUs)"
])

with tab_stockout:
    st.markdown("##### SKUs Facing Buffer Breach During Supplier Lead Time (Ranked by Revenue at Risk)")
    st.caption("Immediate purchase order review required: Available inventory fails to cover expected lead-time demand + safety stock.")

    so_df = filtered_risk[filtered_risk["stockout_flag"] == 1].sort_values(by="revenue_at_risk", ascending=False)

    if len(so_df) == 0:
        st.success("✅ No stockout-risk SKUs in the current filtered view.")
    else:
        display_so = so_df[[
            "SKU", "Product_Name", "Category", "Current_Stock", "On_Order", "available_units",
            "Lead_Time_Days", "lead_time_demand", "Safety_Stock", "required_inventory_buffer",
            "buffer_gap_units", "revenue_at_risk", "recommended_action"
        ]].copy()

        display_so.columns = [
            "SKU", "Product", "Category", "Current Stock", "On Order", "Available Units",
            "Lead Time (Days)", "Lead-Time Demand", "Safety Stock", "Required Buffer",
            "Buffer Deficit", "Revenue at Risk (₹)", "Recommended Action"
        ]

        display_so["Revenue at Risk (₹)"] = display_so["Revenue at Risk (₹)"].apply(lambda x: f"₹{x:,.2f}")
        st.dataframe(display_so, use_container_width=True, hide_index=True)

with tab_overstock:
    st.markdown("##### SKUs Holding Inventory Beyond 4-Week Forward Demand (Ranked by Capital Tied Up)")
    st.caption("Clearance / markdown review recommended: Holding excess inventory locks working capital and increases storage expense.")

    os_df = filtered_risk[filtered_risk["overstock_flag"] == 1].sort_values(by="inventory_value_tied_up", ascending=False)

    if len(os_df) == 0:
        st.success("✅ No overstock-risk SKUs in the current filtered view.")
    else:
        display_os = os_df[[
            "SKU", "Product_Name", "Category", "Current_Stock", "forward_demand_units",
            "excess_units", "Cost_Price", "inventory_value_tied_up", "recommended_action"
        ]].copy()

        display_os.columns = [
            "SKU", "Product", "Category", "Current Stock", "4-Wk Forecast",
            "Excess Units", "Cost Price (₹)", "Capital Tied Up (₹)", "Recommended Action"
        ]

        display_os["Cost Price (₹)"] = display_os["Cost Price (₹)"].apply(lambda x: f"₹{x:,.2f}")
        display_os["Capital Tied Up (₹)"] = display_os["Capital Tied Up (₹)"].apply(lambda x: f"₹{x:,.2f}")
        st.dataframe(display_os, use_container_width=True, hide_index=True)

with tab_all:
    st.markdown("##### Full Filtered Active Inventory Roster")
    display_all = filtered_risk[[
        "SKU", "Product_Name", "Category", "Current_Stock", "On_Order", "Lead_Time_Days",
        "stockout_score", "stockout_flag", "overstock_score", "overstock_flag",
        "decision_quadrant", "value_at_stake", "recommended_action"
    ]].copy()
    display_all["value_at_stake"] = display_all["value_at_stake"].apply(lambda x: f"₹{x:,.2f}")
    st.dataframe(display_all, use_container_width=True, hide_index=True)

# =====================================================================
# SECTION 5, 6, 7: SKU DETAIL, 8-WEEK FORECAST & BUSINESS EXPLANATION
# =====================================================================
st.markdown("---")
st.subheader("🔍 Single-SKU Diagnostic & 8-Week Forward Demand Forecast")

# SKU Selector
available_skus = sorted(risk_df["SKU"].unique().tolist())
default_sku_idx = 9  # Default to SKU010 or SKU012 if available
if "SKU012" in available_skus:
    default_sku_idx = available_skus.index("SKU012")

selected_detail_sku = st.selectbox(
    "Select SKU for Deep-Dive Analysis:",
    options=available_skus,
    index=default_sku_idx,
    help="Select any commercial SKU to inspect demand history, forward ML forecast, and inventory buffer dynamics."
)

sku_risk_record = risk_df[risk_df["SKU"] == selected_detail_sku].iloc[0]
sku_fc_records = fc_df[fc_df["SKU"] == selected_detail_sku].sort_values("Forecast_Week")
sku_hist_records = hist_df[hist_df["SKU"] == selected_detail_sku].sort_values("Week_Start")

# SKU Highlights Cards
st.markdown(f"#### Overview: **{sku_risk_record['Product_Name']}** (`{selected_detail_sku}`) — *{sku_risk_record['Category']}*")

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("Current Warehouse Stock", f"{sku_risk_record['Current_Stock']} units")
    st.metric("In-Transit Pipeline", f"{sku_risk_record['On_Order']} units")
with c2:
    st.metric("Supplier Lead Time", f"{sku_risk_record['Lead_Time_Days']} days")
    st.metric("Safety Stock Buffer", f"{sku_risk_record['Safety_Stock']} units")
with c3:
    st.metric("Lead-Time Demand", f"{sku_risk_record['lead_time_demand']} units")
    st.metric("Required Total Buffer", f"{sku_risk_record['required_inventory_buffer']} units")
with c4:
    st.metric("Stockout Risk Score", f"{sku_risk_record['stockout_score']:.3f}", delta="BREACH" if sku_risk_record['stockout_flag'] == 1 else "OK", delta_color="inverse")
    st.metric("Overstock Risk Score", f"{sku_risk_record['overstock_score']:.3f}", delta="EXCESS" if sku_risk_record['overstock_flag'] == 1 else "OK", delta_color="inverse")
with c5:
    st.metric("Decision Quadrant", sku_risk_record['decision_quadrant'])
    st.metric("Total Value at Stake", f"₹{sku_risk_record['value_at_stake']:,.2f}")

# Business Plain-Language Explanation
st.markdown("##### 💡 Executive Business Rationale & Operational Next Steps")
if sku_risk_record["stockout_flag"] == 1:
    st.error(
        f"🚨 **Stockout Buffer Breach Detected:** Total available inventory "
        f"(**{sku_risk_record['available_units']} units** = {sku_risk_record['Current_Stock']} on-hand + {sku_risk_record['On_Order']} on-order) "
        f"is insufficient to cover projected customer demand during the **{sku_risk_record['Lead_Time_Days']}-day** vendor lead time "
        f"(**{sku_risk_record['lead_time_demand']} units**) plus designated safety stock (**{sku_risk_record['Safety_Stock']} units**). "
        f"This creates an immediate buffer deficit of **{sku_risk_record['buffer_gap_units']} units**, putting "
        f"**₹{sku_risk_record['revenue_at_risk']:,.2f}** in gross sales exposure at risk. "
        f"**Recommended Action:** *{sku_risk_record['recommended_action']}.*"
    )
elif sku_risk_record["overstock_flag"] == 1:
    st.warning(
        f"⚠️ **Excess Inventory Holding:** Physical warehouse inventory (**{sku_risk_record['Current_Stock']} units**) "
        f"exceeds the 4-week forward demand forecast (**{sku_risk_record['forward_demand_units']} units**) by "
        f"**{sku_risk_record['excess_units']} excess units**, locking up **₹{sku_risk_record['inventory_value_tied_up']:,.2f}** "
        f"in idle working capital. "
        f"**Recommended Action:** *{sku_risk_record['recommended_action']}.*"
    )
else:
    st.success(
        f"✅ **Balanced Inventory Position:** Available inventory safely covers forward lead-time requirements "
        f"({sku_risk_record['available_units']} available vs {sku_risk_record['required_inventory_buffer']} required buffer) "
        f"without exceeding 4-week holding targets. "
        f"**Recommended Action:** *{sku_risk_record['recommended_action']}.*"
    )

# SECTION 5: Plotly Forecast vs Actuals Chart
st.markdown("##### 📈 Historical Demand Trend & 8-Week Forward Forecast with Prediction Intervals")

# Prepare historical view (last 52 complete weeks prior to forecast origin)
forecast_origin_date = sku_fc_records["Forecast_Week"].min()
recent_hist = sku_hist_records[sku_hist_records["Week_Start"] < forecast_origin_date].tail(52).copy()

fig_fc = go.Figure()

# 1. Historical Actual Demand
fig_fc.add_trace(go.Scatter(
    x=recent_hist["Week_Start"],
    y=recent_hist["Weekly_Units"],
    mode="lines+markers",
    name="Historical Actual Demand",
    line=dict(color="#38BDF8", width=2.5),
    marker=dict(size=5, color="#38BDF8"),
    hovertemplate="<b>Historical Actual</b><br>Week: %{x|%Y-%m-%d}<br>Units: %{y:.0f}<extra></extra>"
))

# 2. Prediction Interval Band (P10 to P90)
fig_fc.add_trace(go.Scatter(
    x=sku_fc_records["Forecast_Week"].tolist() + sku_fc_records["Forecast_Week"].tolist()[::-1],
    y=sku_fc_records["Interval_P90"].tolist() + sku_fc_records["Interval_P10"].tolist()[::-1],
    fill="toself",
    fillcolor="rgba(168, 85, 247, 0.18)",
    line=dict(color="rgba(255,255,255,0)"),
    name="80% Prediction Interval (P10–P90)",
    hoverinfo="skip"
))

# 3. 8-Week Forward ML Forecast
fig_fc.add_trace(go.Scatter(
    x=sku_fc_records["Forecast_Week"],
    y=sku_fc_records["Forecast_Units"],
    mode="lines+markers",
    name="ML Forecast (HistGradientBoosting)",
    line=dict(color="#A855F7", width=3, dash="solid"),
    marker=dict(size=8, color="#A855F7", symbol="diamond"),
    hovertemplate="<b>Forward ML Forecast</b><br>Week: %{x|%Y-%m-%d}<br>Expected Units: %{y:.1f}<extra></extra>"
))

# 4. Vertical dividing line at exact Forecast Origin boundary (2025-12-29)
fig_fc.add_shape(
    type="line",
    x0=forecast_origin_date,
    y0=0,
    x1=forecast_origin_date,
    y1=max(recent_hist["Weekly_Units"].max(), sku_fc_records["Interval_P90"].max()) * 1.15,
    line=dict(color="#F59E0B", width=2, dash="dash")
)
fig_fc.add_annotation(
    x=forecast_origin_date,
    y=max(recent_hist["Weekly_Units"].max(), sku_fc_records["Interval_P90"].max()) * 1.08,
    text=f"<b>Forecast Origin ({forecast_origin_date.strftime('%Y-%m-%d')})</b>",
    showarrow=True,
    arrowhead=2,
    arrowcolor="#F59E0B",
    font=dict(color="#F59E0B", size=11)
)

fig_fc.update_layout(
    title=f"<b>{selected_detail_sku} Demand History & 8-Week Forward ML Forecast Projection</b>",
    xaxis=dict(title="Timeline (Weekly)", showgrid=True),
    yaxis=dict(title="Weekly Sales Units", showgrid=True),
    height=480,
    margin=dict(l=40, r=40, t=50, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    template="plotly_dark"
)

st.plotly_chart(fig_fc, use_container_width=True)

# SECTION 6: Forecast Data Table
st.markdown("##### 📅 Detailed 8-Week Forward Horizon Schedule")
fc_table_display = sku_fc_records[[
    "Forecast_Week", "Horizon_Step", "Forecast_Units", "Forecast_Revenue", "Interval_P10", "Interval_P90"
]].copy()

fc_table_display["Forecast_Week"] = fc_table_display["Forecast_Week"].dt.strftime("%Y-%m-%d")
fc_table_display.columns = ["Forecast Week", "Horizon Week #", "Expected Demand (Units)", "Projected Revenue (₹)", "P10 Bound", "P90 Bound"]
fc_table_display["Projected Revenue (₹)"] = fc_table_display["Projected Revenue (₹)"].apply(lambda x: f"₹{x:,.2f}")
st.dataframe(fc_table_display, use_container_width=True, hide_index=True)

# Footer
st.markdown("---")
st.caption(
    "Project FORESIGHT v1.0 — Enterprise Demand & Inventory Intelligence Platform | "
    "Designed for NorthBay Living Operations & Merchandising Leadership"
)
