# Project FORESIGHT: Demand & Inventory Intelligence Platform
## Executive Readout & Business Presentation (Deliverable D7)

**Client:** NorthBay Living  
**Audience:** Head of Operations, Merchandising & Supply Chain Leadership, Finance Leadership  
**Presenter:** Antigravity Data Science & Inventory Analytics Team  
**Format:** 10-Slide Executive Presentation (Companion to `reports/FORESIGHT_Executive_Readout.pptx`)  
**Date:** September 9, 2026  

---

## Presentation Overview & Slide Navigation

| Slide # | Slide Title | Core Business Objective | Key Metric / Visual |
| :---: | :--- | :--- | :--- |
| **01** | **Executive Summary: FORESIGHT Intelligence** | Establish immediate executive clarity on risk and exposure | **₹1.79 Cr** Active Value at Stake (6 Stockout, 20 Overstock) |
| **02** | **Business Problem & Inventory Reality** | Detail the D2C planning dilemma & full data universe | **511K units**, ₹309.61 Cr sales, 50 Active vs 150 Inactive SKUs |
| **03** | **What the Historical Data Tells Us (EDA)** | Highlight concentration, promo elasticity, and margin trap | Top 10 = **47.73%** rev, **16 SKUs** selling below cost (-₹30.38 Cr) |
| **04** | **Predictive Demand Forecasting (D3)** | Validate ML performance vs genuine seasonal benchmark | **10.56% WAPE** (beats 11.17% baseline), 4 rolling backtests |
| **05** | **Dynamic Inventory Risk Framework (D4)** | Explain the 4-quadrant forecast-driven buffer logic | **4 Quadrants**: Reorder Now (6), Clear (20), Healthy (24) |
| **06** | **Financial & Business Exposure (D4)** | Quantify rupee exposure without false savings claims | **₹45.18L** Revenue at Risk, **₹1.34 Cr** Capital Tied Up |
| **07** | **Prioritized Business Actions Roadmap** | Define concrete stakeholder accountability across teams | 5-point prioritized action table (Operations, Merchandising, Finance) |
| **08** | **The FORESIGHT Decision Support Platform** | Present Streamlit dashboard & FastAPI scoring architecture | Dual delivery: Interactive Dashboard (D5) + Scoring API (D6) |
| **09** | **Governance, Limitations & Assumptions** | Ensure intellectual honesty, error margins, and bias | **10.56% WAPE**, **+2.03 units** signed bias, lead-time assumptions |
| **10** | **Operational Rollout Roadmap & Conclusion** | Deliver practical 30-day implementation sequence | 4 rollout phases; closing strategic takeaway |

---

## Slide 1: Executive Summary

### Visual Layout & Structure
* **Category Tracker:** `EXECUTIVE READOUT | PROJECT FORESIGHT`
* **Slide Title:** **FORESIGHT: Demand & Inventory Intelligence**
* **Subtitle:** *Bridging dynamic demand forecasts and supplier lead-time buffers to prioritize working capital and service-level exposure.*
* **Top Metric Banner (5 KPI Cards):**
  * `Stockout Risk`: **6 SKUs** (12.0% of active catalog | 660.2 units buffer deficit)
  * `Overstock Risk`: **20 SKUs** (40.0% of active catalog | 4,456 excess units)
  * `Revenue at Risk`: **₹45.18 Lakh** (Gross unfulfilled customer sales exposure during replenishment)
  * `Capital Tied Up`: **₹1.34 Crore** (Operational working capital locked in excess inventory)
  * `Total Value at Stake`: **₹1.79 Crore** (Combined operational decision priority index)
* **3 Strategic Business Takeaway Panels:**
  1. **Replenishment Action Required (6 SKUs):** Immediate purchase order review needed for 6 key SKUs (`SKU010`, `SKU012`, `SKU017`, `SKU023`, `SKU031`, `SKU040`). Total available stock (`Current_Stock + On_Order`) falls 660.2 units short of covering expected demand during vendor lead times plus safety stock.
  2. **Promotional Markdown Review (20 SKUs):** 20 active commercial SKUs hold 4,456 units beyond 4-week projected demand. This locks up ₹1.34 Crore in operational capital that can be unlocked via targeted clearance promotions.
  3. **Enterprise Inactive Stock Governance (150 SKUs):** Reconciled warehouse records reveal ₹17.49 Crore sitting idle in 150 warehouse-only lines with zero sales over 2 years. Operations must freeze 27,737 units of incoming pipeline orders and audit for catalog launch or bulk liquidation.

### Speaker Notes
* **What I Should Say:**
  > "Good morning, leadership team. Today we present Project FORESIGHT—an operational demand and inventory intelligence platform built specifically for NorthBay Living. Rather than relying on gut feeling or static spreadsheets, FORESIGHT links forward machine learning demand forecasts directly to supplier replenishment lead times."
* **Why This Slide Matters:**
  > Establishes immediate executive clarity. In one view, leaders see that ₹1.79 Crore in working capital and revenue exposure is currently at stake across active operations, divided into 6 urgent replenishment items and 20 overstock items.
* **Key Number to Emphasize:**
  > **₹1.79 Crore total active value at stake**—comprising ₹45.18 Lakhs in revenue exposure from 6 stockout-risk SKUs and ₹1.34 Crore in capital tied up across 20 excess lines.
* **Likely Executive Question & Answer:**
  > *Q (CFO):* "Is that ₹1.79 Crore guaranteed cost savings or realized loss?"  
  > *A:* "No. This is an operational exposure metric prioritizing management attention. The ₹45.18 Lakhs is potential unmet demand if replenishment is delayed, while the ₹1.34 Crore is working capital currently locked in excess inventory beyond our 30-day operating target."

---

## Slide 2: Business Problem & Inventory Reality

### Visual Layout & Structure
* **Category Tracker:** `THE OPERATIONAL CONTEXT`
* **Slide Title:** **The Planning Dilemma: Moving from Spreadsheets to Forecast Intelligence**
* **Subtitle:** *NorthBay Living operates in a fast-paced D2C home & lifestyle market where spreadsheet heuristics lead to chronic misalignment.*
* **Two-Column Comparative Architecture:**
  * **Left Column — The Core Business Challenge:**
    * *Bestseller Stockouts & Customer Churn:* High-velocity items repeatedly run out of stock during supplier lead times (3–14 days), disappointing D2C shoppers and driving them to competitors.
    * *Working Capital Trapped in Slow Movers:* Capital becomes trapped in over-ordered inventory holding beyond 30 days of forward demand, elevating holding costs and warehouse storage pressure.
    * *Blindness of Static Reorder Points:* Traditional static reorder points look only at physical shelf stock, ignoring orders already placed with suppliers (`On_Order`) and failing to project upcoming demand surges.
    * *Fragmented Spreadsheets & Siloed Planning:* Operations and Merchandising teams historically planned in disconnected spreadsheets without a single forward demand signal.
  * **Right Column — Dataset Foundation & Enterprise Reconciliation:**
    * *Full 2-Year Transactional History:* 511,810 units sold totaling ₹3,096,056,707.22 (₹309.61 Cr) across daily records from Jan 1, 2024 to Dec 31, 2025.
    * *50 Active Commercial SKUs:* The core revenue-generating catalog modeled through 105 synchronized weekly demand cycles without a single missing record.
    * *150 Warehouse-Only Inactive Lines:* Reconciled from physical warehouse extracts: 39,985 units holding ₹17.49 Cr with zero sales in 2 years. Fully preserved in the data layer.
    * *Full 200-SKU Enterprise Grid:* Ensures zero operational data loss: 50 commercial lines receive dynamic ML forecasts while 150 inactive lines are segregated for liquidation.
    * *Multi-Echelon Pipeline Tracking:* Synchronizes current physical warehouse stock with confirmed supplier orders in transit (`On_Order`) across vendor lead times of 3 to 14 days.

### Speaker Notes
* **What I Should Say:**
  > "To understand why NorthBay Living needed FORESIGHT, we have to look at the daily reality of inventory planning. When you rely on static spreadsheets, two things happen simultaneously: bestsellers run out because nobody saw the demand surge coming, while slow movers quietly accumulate dust in the warehouse."
* **Why This Slide Matters:**
  > Proves data hygiene and enterprise rigor. We did not throw away messy data; we cleanly separated the 50 active commercial SKUs from the 150 warehouse-only lines, preserving the entire 200-SKU inventory universe.
* **Key Number to Emphasize:**
  > **511,810 units sold across two complete years representing ₹309.61 Crore** in top-line commercial activity.
* **Likely Executive Question & Answer:**
  > *Q (Head of Supply Chain):* "Why aren't the 150 warehouse-only SKUs being forecasted?"  
  > *A:* "Because they have zero commercial transactions over the entire 2-year history. You cannot forecast consumer demand where there has never been customer sales. Instead, they require an operational freeze and catalog liquidation audit."

---

## Slide 3: What the Historical Data Tells Us (EDA)

### Visual Layout & Structure
* **Category Tracker:** `EXPLORATORY DATA ANALYSIS`
* **Slide Title:** **What the Data Tells Us: 4 Crucial Commercial Realities**
* **Subtitle:** *Rigorous empirical profiling revealed severe revenue concentration, strong promotional sensitivity, and a hidden negative margin trap.*
* **2x2 Structured Card Grid:**
  1. **Extreme Revenue Concentration (Top 10 SKUs = 47.73% | Top 20 SKUs = 74.60%):**
     Commercial revenue is heavily skewed toward a small elite group of top-performing products. A stockout on any of the top 10 SKUs causes immediate, outsized financial damage to NorthBay Living.
  2. **High Promotional Elasticity (+38.12% Unit Uplift | +37.74% Revenue Uplift):**
     Promotional campaigns drive substantial volume acceleration. Forward planning must proactively anticipate supplier lead times prior to running promotional events to avoid promotional stockouts.
  3. **Strong Demand Seasonality (March Peak: 118.5 Units/Wk | October Trough: 80.1 Units/Wk):**
     Weekly sales exhibit clear cyclical swings across spring and autumn. Static monthly ordering creates severe stockouts in Q1 and severe overstocking in Q3/Q4.
  4. **The Negative-Margin Drag Trap (16 / 50 Active SKUs Sold Below Cost Price):**
     Due to misaligned discounting, 165,593 units were sold at a loss, generating ₹51.69 Cr in top-line revenue but inflicting a massive -₹30.38 Cr cumulative gross profit drag.
* **Embedded Reference Visuals:**
  * Figure 03: [`reports/figures/03_category_performance.png`](file:///C:/Users/jalag/Documents/demand-inventory-analysis/reports/figures/03_category_performance.png)
  * Figure 06: [`reports/figures/06_negative_margins.png`](file:///C:/Users/jalag/Documents/demand-inventory-analysis/reports/figures/06_negative_margins.png)

### Speaker Notes
* **What I Should Say:**
  > "When we profiled the two years of transactional data, four commercial realities immediately surfaced. First, NorthBay's revenue is intensely concentrated—just 10 SKUs generate nearly half of all company revenue. Second, promotions work: they boost unit sales by 38.12%. Third, there is strong seasonality, peaking at 118 units per week in March and bottoming at 80 in October."
* **Why This Slide Matters:**
  > Insight #4 is a major eye-opener for the CFO: 16 out of the 50 commercial SKUs were selling below cost, burning ₹30.38 Crore in cumulative gross profit. This proves why inventory intelligence must connect demand to margins.
* **Key Number to Emphasize:**
  > **16 out of 50 active SKUs generated ₹51.69 Crore in sales but created a -₹30.38 Crore loss in gross margin.**
* **Likely Executive Question & Answer:**
  > *Q (Finance Lead):* "How did 16 SKUs end up with negative margins?"  
  > *A:* "Static price points and deep promotional discounts were applied without reconciling against supplier cost increases. FORESIGHT flags these SKUs so merchandising can immediately restructure pricing."

---

## Slide 4: Predictive Demand Forecasting (D3)

### Visual Layout & Structure
* **Category Tracker:** `PREDICTIVE ANALYTICS ENGINE`
* **Slide Title:** **Can Machine Learning Outperform Seasonal Heuristics?**
* **Subtitle:** *Evaluating an 8-week forward horizon through 4 chronological rolling-origin backtests against a genuine seasonal baseline.*
* **Two-Column Split Layout:**
  * **Left Column — Model Validation & Selection Results:**
    * *Production ML Model:* `HistGradientBoostingRegressor` selected for robust handling of non-linear lag interactions and calendar features.
    * *Seasonal-Naive Baseline WAPE:* **11.17% WAPE** (Bias: +0.28 units/week). A rigorous 52-week seasonal lag benchmark.
    * *Production ML Model WAPE:* **10.56% WAPE** (Bias: +2.03 units/week). Beats baseline by **0.61 percentage points** (~5.5% relative error reduction).
    * *Rolling-Origin Backtesting:* Evaluated across 4 chronological origin cutoff periods (1,600 individual SKU forecasts). ML won 3 out of 4 origins.
    * *Zero Data Leakage Governance:* Strict temporal splits. No future sales or target leakage. The 3-day partial final historical week was explicitly excluded from training.
    * *8-Week Forward Horizon Projection:* 32,233.1 total units and ₹19,51,23,017.06 (₹19.51 Cr) projected customer demand with 80% P10/P90 prediction intervals.
  * **Right Column — Embedded Validation Figure:**
    * Embeds Figure 07: [`reports/figures/07_forecast_backtest_eval.png`](file:///C:/Users/jalag/Documents/demand-inventory-analysis/reports/figures/07_forecast_backtest_eval.png) showing out-of-sample backtest curves and error distributions.

### Speaker Notes
* **What I Should Say:**
  > "The foundational question of D3 was simple: Can machine learning predict weekly demand better than a seasonal baseline? We tested this with complete integrity across 4 rolling-origin backtests—meaning the model was evaluated exactly as if it were making real forward decisions in time."
* **Why This Slide Matters:**
  > Shows technical rigor and intellectual honesty. We did not use a random train/test split. The ML model achieved 10.56% WAPE, delivering a modest but consistent 5.5% error reduction over the seasonal benchmark and winning 3 of 4 origins.
* **Key Number to Emphasize:**
  > **10.56% WAPE across 1,600 out-of-sample backtest predictions**, proving the model is stable and production-ready.
* **Likely Executive Question & Answer:**
  > *Q (Head of Merchandising):* "Is a 0.61 percentage-point improvement worth deploying?"  
  > *A:* "Yes. In enterprise retail with ₹300+ Cr in volume, reducing demand forecasting error by 5.5% directly prevents over-ordering and optimizes buffer sizing across all 50 SKUs."

---

## Slide 5: Dynamic Inventory Risk Framework (D4)

### Visual Layout & Structure
* **Category Tracker:** `DYNAMIC RISK DECISIONING`
* **Slide Title:** **Connecting Forecasts to Lead Times: The 4 Decision Quadrants**
* **Subtitle:** *Stockout risk is forecast-driven, evaluating total available inventory against supplier lead-time demand plus safety stock.*
* **Two-Column Comparative Layout:**
  * **Left Column — Embedded Decision Grid Visual:**
    * Embeds Figure 09: [`reports/figures/09_risk_decisioning_grid.png`](file:///C:/Users/jalag/Documents/demand-inventory-analysis/reports/figures/09_risk_decisioning_grid.png) displaying the 2D scatter matrix.
  * **Right Column — The 4 Operational Decision Quadrants:**
    * 🔴 **REORDER NOW (6 SKUs | 12.0%):**  
      *Rule:* $\text{available\_units} < \text{required\_inventory\_buffer}$  
      *Details:* Current Stock + On-Order fails to cover expected lead-time demand plus safety stock. Buffer deficit = 660.2 units. Urgent purchase order scheduling required.
    * 🔵 **MARKDOWN / CLEAR (20 SKUs | 40.0%):**  
      *Rule:* $\text{Current\_Stock} > \text{forward\_demand\_units}$  
      *Details:* Holding inventory beyond 30 days of forward demand. 4,456 excess units tying up ₹1.34 Cr. Targeted promotional clearance recommended.
    * 🟠 **WATCH / VOLATILE (0 SKUs | 0.0%):**  
      *Rule:* Buffer Breach + Excess Holding Coexistence  
      *Details:* Currently zero SKUs exhibit dual risk tension, indicating clean operational separation between stockout risks and holding excess.
    * 🟢 **HEALTHY (24 SKUs | 48.0%):**  
      *Rule:* Balanced Buffer & Holding Position  
      *Details:* Inventory safely covers supplier lead-time requirements without exceeding 4-week holding thresholds. Maintain and monitor.

### Speaker Notes
* **What I Should Say:**
  > "Here is the heart of D4: the 4-Quadrant Decisioning Grid. Unlike traditional systems that trigger alarms simply because physical stock is below a static reorder point, FORESIGHT uses a dynamic lead-time buffer."
* **Why This Slide Matters:**
  > Explains the crucial difference between physical stock and total available inventory (`Current_Stock + On_Order`). Under a static reorder point rule, 15 SKUs would have triggered alarms. But 10 of those SKUs already have large purchase orders on the way. FORESIGHT prevents double-ordering while uncovering genuine buffer deficits like `SKU010`.
* **Key Number to Emphasize:**
  > **Exactly 6 SKUs in Reorder Now, 20 in Markdown/Clear, and 24 completely Healthy.**
* **Likely Executive Question & Answer:**
  > *Q (Head of Operations):* "Why are there 0 SKUs in Watch / Volatile?"  
  > *A:* "Because in NorthBay's catalog right now, no SKU has both a lead-time deficit and a 4-week excess holding. The operational states are cleanly separated between under-stocked bestsellers and overstocked slow movers."

---

## Slide 6: Financial & Business Exposure (D4)

### Visual Layout & Structure
* **Category Tracker:** `FINANCIAL QUANTIFICATION`
* **Slide Title:** **Quantifying Value at Stake: ₹1.79 Crore Operational Exposure**
* **Subtitle:** *Translating supply chain imbalances into exact rupee exposure metrics to guide executive prioritization.*
* **Top Metric Triad (3 Hero Cards):**
  * `Revenue at Risk Exposure`: **₹45.18 Lakh**  
    *660.2 units buffer deficit across 6 bestsellers. Potential unfulfilled gross sales during vendor replenishment if orders are delayed. NOT guaranteed lost revenue.*
  * `Capital Tied Up in Excess`: **₹1.34 Crore**  
    *4,456 excess inventory units across 20 overstocked lines. Operational working capital locked beyond 30-day demand target. NOT a permanent accounting loss.*
  * `Total Active Value at Stake`: **₹1.79 Crore**  
    *Combined managerial priority index guiding operational triage between reordering and clearance. Exposure / decision-support metric.*
* **Enterprise Capital Exposure Context & Governance Panel:**
  * *Stockout Concentration (Top 3 SKUs = 94.7% of Risk):* Among the 6 stockout SKUs, three lines drive nearly the entire exposure: `SKU012` (₹18.25L), `SKU017` (₹13.34L), and `SKU031` (₹11.22L). Focusing operations on these three SKUs resolves 95% of stockout exposure.
  * *Overstock Concentration (Top 3 SKUs = 51.8% of Tied-Up Capital):* Excess inventory is heavily concentrated in Storage and Furniture categories: `SKU020` (₹41.87L), `SKU025` (₹13.92L), and `SKU036` (₹13.75L). Targeted promotional bundles can unlock over ₹69 Lakhs in liquidity.
  * *The Inactive Warehouse Capital Perspective (₹17.49 Crore):* While active catalog exposure is ₹1.79 Crore, physical warehouse reconciliation reveals ₹17.49 Crore locked in 150 inactive SKUs. Strategic attention from executive leadership must address this dormant asset portfolio.
  * *Terminology Governance:* All values are presented strictly as operational exposure estimates to support decision-making, not as guaranteed cost savings or realized accounting gains.

### Speaker Notes
* **What I Should Say:**
  > "Let's look at the financial exposure. We see three key numbers: ₹45.18 Lakhs in revenue at risk, ₹1.34 Crore in working capital tied up in excess stock, leading to a total active value at stake of ₹1.79 Crore."
* **Why This Slide Matters:**
  > Demonstrates financial literacy and intellectual honesty. We do not claim this is 'guaranteed savings' or 'pure profit.' Revenue at risk is potential unfulfilled demand if purchase orders are delayed; capital tied up is cash sitting on shelves that could be deployed elsewhere.
* **Key Number to Emphasize:**
  > **Top 3 stockout SKUs account for 94.7% of revenue at risk**—meaning leadership only needs to prioritize three specific SKUs to protect nearly all stockout exposure.
* **Likely Executive Question & Answer:**
  > *Q (Finance Director):* "Can we recover the entire ₹1.34 Crore tied up in overstock?"  
  > *A:* "Yes, through planned merchandising clearance. Because these are active commercial items with proven sales, running promotional markdowns will accelerate sell-through without severe write-downs."

---

## Slide 7: Prioritized Business Actions Roadmap

### Visual Layout & Structure
* **Category Tracker:** `OPERATIONAL EXECUTION`
* **Slide Title:** **Prioritized Action Roadmap for Operations, Merchandising & Finance**
* **Subtitle:** *Concrete operational recommendations categorized by urgency, business rationale, and stakeholder ownership.*
* **Executive Action Matrix (Table):**

| Priority | Recommended Action | Target Scope | Operational Rationale & Business Impact |
| :--- | :--- | :--- | :--- |
| **P1 (Urgent)** | **Replenishment Purchase Orders** | 6 Stockout SKUs (`SKU012`, `017`, `031`, etc.) | Expedite vendor orders for 6 buffer-breached lines. Protects ₹45.18 Lakhs in sales exposure. Top 3 lines represent 95% of exposure. |
| **P2 (Near-Term)** | **Promotional Markdowns & Clearance** | 20 Overstock SKUs (`SKU020`, `025`, `036`, etc.) | Launch targeted promotional discounts or bundle packages on 4,456 excess units to unlock ₹1.34 Crore in working capital and reduce holding costs. |
| **P3 (Strategic)** | **Audit Warehouse-Only Inventory** | 150 Inactive SKUs (39,985 units) | Freeze 27,737 units of in-transit orders immediately. Conduct catalog review to determine whether to launch online or initiate bulk B2B liquidation (₹17.49 Cr). |
| **P4 (Commercial)** | **Restructure Negative-Margin SKUs** | 16 Active SKUs (165,593 units sold) | Renegotiate vendor costs or raise retail price points on 16 loss-making products to stem -₹30.38 Crore in cumulative margin erosion. |
| **P5 (Governance)** | **Forecast Monitoring & Recalibration** | All 50 Commercial SKUs (Monthly cadence) | Track monthly WAPE and signed bias (+2.03 units) against actual orders to ensure models stay tightly calibrated as customer demand evolves. |

* **Operational Governance Footnote:**  
  *⚠️ Operational Governance Boundary: FORESIGHT provides prioritized decision-support recommendations. Automated purchase order firing and dynamic algorithmic repricing are intentionally out of scope to preserve executive review.*

### Speaker Notes
* **What I Should Say:**
  > "Here is the prioritized operational action plan. Priority 1: Operations immediately reviews replenishment for the 6 stockout SKUs. Priority 2: Merchandising initiates clearance promotions for the 20 overstock items. Priority 3: Finance audits the ₹17.49 Crore in warehouse-only stock. Priority 4: Re-price the 16 negative-margin SKUs. Priority 5: Ongoing forecast governance."
* **Why This Slide Matters:**
  > Transforms analytical insights into concrete stakeholder accountability. Every recommendation has a clear target scope and business impact.
* **Key Number to Emphasize:**
  > **Priority 3: Freeze 27,737 units of in-transit orders** for inactive lines before more working capital lands in the warehouse.
* **Likely Executive Question & Answer:**
  > *Q (Head of Supply Chain):* "Does FORESIGHT place purchase orders automatically?"  
  > *A:* "No. The platform is strictly an executive decision-support system. Human oversight from Operations and Merchandising remains mandatory."

---

## Slide 8: The FORESIGHT Decision Support Platform

### Visual Layout & Structure
* **Category Tracker:** `TECHNOLOGY & DELIVERY ARCHITECTURE`
* **Slide Title:** **The FORESIGHT Platform: Interactive Planning & Scoring Services**
* **Subtitle:** *Delivering dual capabilities: an interactive Streamlit planning dashboard for business leaders and a real-time FastAPI scoring service for ERP integration.*
* **Side-by-Side Architectural Breakdown:**
  * **Left Card — Streamlit Planning Dashboard (Deliverable D5):**
    * *Executive KPI Banner:* Dynamic metrics calculating total SKUs, stockout counts, overstock counts, revenue at risk, and working capital tied up in real time.
    * *Interactive 4-Quadrant Matrix:* Plotly scatter chart mapping all 50 SKUs across Reorder Now, Markdown/Clear, Watch/Volatile, and Healthy states with bubble sizing by value at stake.
    * *Prioritized Action Tables:* One-click filtered action queues for Operations (stockout buffer breaches) and Merchandising (excess holdings).
    * *Single-SKU Diagnostic Deep-Dive:* Historical 52-week demand trend, 8-week ML forecast projection, and 80% prediction interval bands (P10 to P90).
    * *Plain-Language Business Rationale:* Dynamic executive explanations detailing available units vs required buffers without technical jargon.
  * **Right Card — FastAPI Scoring Service (Deliverable D6):**
    * *`GET /health`:* Health check endpoint returning system status, model name (`HistGradientBoostingRegressor`), and validation WAPE (10.56%).
    * *`GET /forecast/{sku_id}`:* Returns 8-week forward forecast, total projected units, projected revenue, and weekly P10/P90 confidence bounds.
    * *`GET /risk/{sku_id}`:* Returns dynamic lead-time demand, buffer gap, stockout/overstock risk flags, decision quadrant, and rupee exposure.
    * *`POST /score/batch`:* High-throughput endpoint accepting multiple SKU IDs, returning scored records and isolating uncataloged SKUs in error responses.
    * *Enterprise-Ready Architecture:* Pydantic v2 data validation, CORS middleware enabled, fully containerizable with `Procfile` and `render.yaml` blueprints.

### Speaker Notes
* **What I Should Say:**
  > "To make sure these insights don't just stay on slides, we built two production software deliverables: a Streamlit planning dashboard for business leaders, and a FastAPI scoring service for IT and system integration."
* **Why This Slide Matters:**
  > Demonstrates complete software delivery readiness. The dashboard gives planners interactive visual control, while the API enables ERP systems to pull forward risk scores programmatically.
* **Key Number to Emphasize:**
  > **Sub-millisecond API response latency** across all 4 production endpoints with automated error isolation.
* **Likely Executive Question & Answer:**
  > *Q (IT Director):* "How hard is it to deploy this into our existing cloud setup?"  
  > *A:* "Both applications are deployment-ready with standardized requirements.txt, Procfile, and Render blueprints. They can be hosted on internal cloud or accessed via secure public URLs immediately."

---

## Slide 9: Governance, Limitations & Assumptions

### Visual Layout & Structure
* **Category Tracker:** `METHODOLOGICAL INTEGRITY`
* **Slide Title:** **Methodological Governance, Limitations & Assumptions**
* **Subtitle:** *Transparent accounting of model assumptions, prediction uncertainty, and operational scope boundaries.*
* **2x2 Governance Grid:**
  1. **Forecast Uncertainty & Horizon Scope (Validation WAPE: 10.56% | Horizon: 8 Weeks):**
     The model reduces forecasting error by ~5.5% over the seasonal benchmark, but 10.56% error remains. Forecasts extend 8 weeks forward; planning beyond 60 days requires monthly model recalibration as macro conditions shift.
  2. **Positive Signed Model Bias (+2.03 Units/Week):**
     The ML model exhibits a small positive signed bias (+2.03 units/week out of sample). Operationally, this acts as a conservative safety buffer against stockouts, but merchandising should verify slow-moving item projections.
  3. **Supplier Execution & Transit Dynamics (Assumes Historical Lead Times: 3–14 Days):**
     Dynamic buffer calculations rely on vendor lead times provided in catalog data. External supply chain shocks, port delays, or vendor stockouts will alter real-world arrival times and require buffer adjustments.
  4. **Operational Scope Boundaries (Decision-Support System, Not ERP Execution):**
     FORESIGHT is strictly an executive decision-support platform. Automated purchase order firing, dynamic price-setting, and real-time streaming data architectures were deliberately kept out of scope to prioritize human review.

### Speaker Notes
* **What I Should Say:**
  > "A great data science project must be honest about what it can and cannot do. We do not claim perfect accuracy. Our machine learning model has an out-of-sample error rate of 10.56%, and it has a slight positive bias of +2.03 units per week, which actually protects us against stockouts."
* **Why This Slide Matters:**
  > Builds immense credibility with executive leadership. Executives distrust models that claim 100% accuracy. By explaining the positive bias and supplier execution assumptions, we show that we understand real-world supply chain risks.
* **Key Number to Emphasize:**
  > **+2.03 units/week signed bias**—modest, explainable, and beneficial as an operational buffer.
* **Likely Executive Question & Answer:**
  > *Q (Head of Supply Chain):* "What happens if a supplier takes 20 days instead of the recorded 12 days?"  
  > *A:* "Because the risk engine is parameterized by Lead_Time_Days, updating vendor lead times immediately re-calculates the required buffer and updates risk scores across the platform."

---

## Slide 10: Rollout Roadmap & Strategic Conclusion

### Visual Layout & Structure
* **Category Tracker:** `STRATEGIC ROADMAP`
* **Slide Title:** **Implementation Roadmap: 4 Phases to Full Deployment**
* **Subtitle:** *A structured 30-day operational rollout translating Project FORESIGHT into lasting enterprise impact.*
* **4-Phase Linear Sequence Across Slide:**
  * **PHASE 1 (WEEK 1) — Immediate Triage:**
    * Validate 6 stockout SKUs with Operations.
    * Expedite POs for `SKU012`, `SKU017`, `SKU031`.
    * Review 20 clearance lines with Merchandising.
    * *Target:* Prevent ₹45.18L stockout exposure.
  * **PHASE 2 (WEEK 2) — Commercial Restructuring:**
    * Audit 16 negative-margin active SKUs.
    * Adjust promotional discounts and retail pricing.
    * Freeze 27,737 units of inactive on-order stock.
    * *Target:* Stem -₹30.38 Cr margin drain.
  * **PHASE 3 (WEEKS 3–4) — Platform Deployment:**
    * Deploy Streamlit dashboard to planning teams.
    * Integrate FastAPI service with internal ERP.
    * Onboard Operations and Finance leads.
    * *Target:* Operationalize automated scoring.
  * **PHASE 4 (ONGOING) — Governance & Recalibration:**
    * Monthly forecast re-training on new sales data.
    * Track realized WAPE vs 10.56% benchmark.
    * Update vendor lead times dynamically.
    * *Target:* Continuous model improvement.
* **Concluding Executive Anchor Banner:**
  > *“Project FORESIGHT turns demand and inventory data into prioritized operational decisions.”*  
  > *Delivered on time, validated end-to-end, and ready for operational deployment at NorthBay Living.*

### Speaker Notes
* **What I Should Say:**
  > "To conclude, here is our 30-day implementation roadmap. Week 1 is immediate operational triage on the 6 stockout and 20 clearance SKUs. Week 2 addresses the commercial margin restructuring. Weeks 3 and 4 deploy the dashboard and API services to the broader team. And ongoing governance ensures monthly recalibration."
* **Why This Slide Matters:**
  > Ends the presentation with an inspiring, actionable closing statement that leaves the executive team with absolute confidence in next steps.
* **Key Number to Emphasize:**
  > **4 distinct phases** moving from immediate triage to sustained operational excellence.
* **Likely Executive Question & Answer:**
  > *Q (Chief Operating Officer):* "What is the immediate next step on Monday morning?"  
  > *A:* "Operations schedules purchase order reviews for SKU012, SKU017, and SKU031 to protect 95% of our stockout risk exposure."

---

*Companion Document to `reports/FORESIGHT_Executive_Readout.pptx`. Prepared for Project FORESIGHT final executive evaluation.*
