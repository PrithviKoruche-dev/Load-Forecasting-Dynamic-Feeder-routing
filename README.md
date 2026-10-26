# ⚡ AI-Driven Load Forecasting & Dynamic Feeder Rerouting

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![OpenDSS](https://img.shields.io/badge/Simulator-OpenDSS-brightgreen.svg)](https://www.epri.com/pages/sa/opendss)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red.svg)](https://streamlit.io/)
[![Plotly](https://img.shields.io/badge/Plots-Plotly-purple.svg)](https://plotly.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Final Year Engineering Capstone Project**  
> An autonomous cyber-physical energy management platform that combines **24-hour-ahead machine learning load forecasting** with **real-time physics-constrained power flow optimization** to proactively detect and mitigate transformer overloads through **dynamic low-voltage (LV) load transfers**, with optional **medium-voltage (MV) feeder transfers** over the normally open tie `TIE1`. Every switching proposal is checked by AC power flow, explained, applied according to an operating mode (monitor, recommend-and-approve, or automatic), and recorded in a database that survives restarts.

---

## 📌 Table of Contents
- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Distribution Network & Switching Topology](#-distribution-network--switching-topology)
- [Machine Learning Load Forecasting](#-machine-learning-load-forecasting)
- [Physics-Constrained Rerouting Engine](#-physics-constrained-rerouting-engine)
- [Stress Scenarios & Experimental Results](#-stress-scenarios--experimental-results)
- [Operator Dashboard (Streamlit & Plotly)](#-operator-dashboard-streamlit--plotly)
- [Closed-Loop Simulation Service](#-closed-loop-simulation-service)
- [Configuration, Asset IDs & Logging](#-configuration-asset-ids--logging)
- [Repository Structure](#-repository-structure)
- [Getting Started & Installation](#-getting-started--installation)
- [Usage Guide & CLI Commands](#-usage-guide--cli-commands)
- [API & Interface Contract](#-api--interface-contract)
- [Future Enhancements](#-future-enhancements)
- [License](#-license)

---

## 📖 Project Overview

Modern electrical distribution grids are facing unprecedented stress caused by:
1. High-power **Electric Vehicle (EV)** charging sessions during evening peak hours.
2. Extreme weather events (such as **heatwaves**) triggering concentrated air-conditioning loads.
3. Rapid residential and commercial load growth that exceeds local substation transformer ratings.

Conventional utility solutions rely on expensive capital infrastructure upgrades (replacing transformers and underground cables) or crude load shedding (rolling blackouts). 

**Our Solution:**  
This project presents an intelligent, automated alternative: **Dynamic Feeder Rerouting**. By utilizing smart-meter telemetry and weather data, an **XGBoost Regressor** predicts distribution transformer loads 24 hours in advance. When an impending overload is detected (exceeding 85% of rated capacity), a **greedy constraint-satisfaction optimization engine** interfaced with **EPRI OpenDSS** calculates optimal tie-switch operations to reroute movable load blocks to adjacent, under-utilized transformers. The reconfiguration executes in **sub-10 milliseconds**, strictly enforcing ANSI voltage tolerances and thermal ampacity limits.

---

## ✨ Key Features

- **🧠 Day-Ahead Load Forecasting:**
  - 24-hour direct multi-step hourly demand forecasting using real smart-meter telemetry.
  - Single unified XGBoost model serving all transformers via train-mean scaling.
  - Engineered lag series ($\ge 24\text{ h}$ to prevent data leakage), rolling statistics, and cooling degree days (CDD).
  - Outperforms persistence baselines by over **21% in nRMSE** (achieving **25.3% nRMSE** at transformer level and **10.8%** for the sum of all six transformers).
  - P10–P90 forecast intervals by rolling conformal calibration (78.3% coverage), with peak-hour and overload-detection metrics showing why forecast risk uses the P90 bound.
  - Versioned model metadata with a file hash, a cold-start builder, step-by-step `forecast_at(t)` and live scoring of forecasts as their hour passes.

- **⚡ Physics-Based Power Flow Emulation (OpenDSS):**
  - Full AC distribution power flow simulation using `opendssdirect.py`.
  - Realistic 11 kV dual-feeder topology with 6 distribution transformers (500 kVA each) and ANSI C84.1 voltage regulation ($0.95 \le V \le 1.05\text{ pu}$).
  - Split-load architecture: 80% non-transferable baseline load (Block A) and 20% transferable load (Block B).

- **🔄 Fast Greedy Rerouting Algorithm:**
  - Multi-round heuristic search for single-switch load transfers with automatic pairwise fallback.
  - Sub-second deterministic decision-making: average execution time of **5–9 ms** (maximum $<25\text{ ms}$).
  - Thermal and voltage safety checks on every candidate move, on every bus including the 0.415 kV customer side, and a NetworkX radiality check (no loops, every load supplied) before each power flow.
  - Every candidate is recorded with its outcome and the reason it was rejected; lower electrical losses break ties between equally good moves.
  - Failsafe handling: returns `no_safe_option` when grid capacity is exhausted, with a plain-language **binding constraint** (e.g. "T4: even with its block b moved to T5 it would be at 114.2%").
  - Optional MV feeder transfers (open sectionaliser `A2`/`B2`, close `TIE1`) for feeder-cable and voltage problems; voltage and line violations trigger a search even when no transformer is overloaded.

- **📊 Interactive Streamlit Operator Dashboard:**
  - Live interactive single-line network diagram powered by Plotly, transformers coloured by risk class, open and closed switches drawn as they are.
  - Time-scrubbing replay slider with automatic "Jump to Peak Hour" capability.
  - Automated 24-hour day-ahead overload scanning matrix with severity tagging (`CRITICAL` vs `WARNING`).
  - Five risk classes (normal, elevated, approaching, predicted overload, actual overload), with forecast risk shown separately from measured risk.
  - Three operating modes: monitor-only, recommend (operator approves), automatic; emergency stop, rollback and an action log.
  - Structured recommendations: ID, numbered switching plan with switch IDs, before/after, losses, constraint checks and a confidence level.
  - Thread-isolated worker thread architecture preventing C-extension race conditions.

- **🗄 Persistent Storage (SQLite, PostgreSQL optional):**
  - Forecasts, measured loads, every recommendation with all its candidates, every switch operation, alerts and the full event log are stored in a database.
  - The controller's mode, switch configuration, emergency stop, undo stack and pending recommendation survive a restart; a recommendation left pending can still be approved afterwards.
  - Filterable audit log (by day, transformer, event type and severity).

- **🔁 Closed-Loop Simulation Service:**
  - An orchestrator runs every module once per simulated hour (measure, power flow, forecast, detect, validate, apply or propose, record) as a background service, independent of any dashboard.
  - Simulation clock with pause, resume, single step, jump, restart and variable speed, including an accelerated mode of one simulated year per 24 hours; the achieved speed is measured.
  - All state in the database: the live dashboard and a command-line tool only send commands; a killed service restarts where it stopped and gives the same result.

- **🔬 Comprehensive Scenario Verification:**
  - Validated across **1,073 continuous test hours** across 5 distinct operational scenarios: Normal Day, Feeder A Heatwave, EV Evening Surge, Severe Multi-Feeder Stress, and Extreme Single-Transformer Overload.

---

## 🏗 System Architecture

```mermaid
flowchart TD
    subgraph Data_Pipeline [Data Ingestion & Preprocessing]
        A[eMARC Smart Meter Data\n15-min interval kW] --> C[Data Cleaning & Gap Interpolation]
        B[Open-Meteo Weather API\nHourly Temp & CDD] --> C
        C --> D[Processed Cluster Dataset\nclusters.parquet]
    end

    subgraph ML_Forecasting [24-Hour Load Forecasting]
        D --> E[Feature Engineering\nLags >=24h, Rolling Stats, Calendar]
        E --> F[XGBoost Model Training\nEarly Stopping & Refit]
        F --> G[Grid-Calibrated Forecast\ndata/forecast.csv]
    end

    subgraph Optimization_Engine [Physics Engine & Reconfiguration]
        G --> H[Grid Engine\ngrid/engine.py]
        I[Network Model\ngrid/network.dss] --> H
        H --> J{Overload Detected?\nLoading > 85%}
        J -- No --> K[Status: no_action]
        J -- Yes --> L[Greedy Switch Search\nEvaluate SW1A-SW6A]
        L --> M{AC Power Flow Feasible?\n0.95 <= V <= 1.05 pu\nLoading <= 100%}
        M -- Yes --> N[Status: rerouted\nApply Optimal Tie Switches]
        M -- No --> O[Status: no_safe_option\nRetain Safe Topology]
    end

    subgraph Operator_UI [Decision Support Dashboard]
        K --> P[Streamlit Operator Console\napp/app.py]
        N --> P
        O --> P
        P --> Q[Plotly Topology Map]
        P --> R[Day-Ahead Alert Scanner]
        P --> S[Manual Override & Action Log]
    end
```

---

## 🔌 Distribution Network & Switching Topology

The electrical network represents an urban 11 kV / 0.415 kV distribution system with two radial feeders originating from a central 33/11 kV substation (`SUB`). In `network.dss`, `Line.A2`, `Line.B2` and `TIE1` are declared with `switch=yes`, which OpenDSS turns into 0.001 km switches: each feeder is 2 km of cable (`A1`+`A3`, `B1`+`B3`), and `A2`/`B2` act as sectionalising switches (asset IDs `SW-014`, `SW-015`) used by the MV transfers.

```
       +------------------- 11 kV Substation (SUB) -------------------+
       |                                                               |
  [Feeder A]                                                      [Feeder B]
       |                                                               |
    (Line A1)                                                       (Line B1)
       |                                                               |
      [A1] -------- Transformer T1 (500 kVA)                          [B1] -------- Transformer T4 (500 kVA)
       |            L1 (Block 1a: 80%)                              |            L4 (Block 4a: 80%)
       |            L1b (Block 1b: 20%) --[SW1A]--> (to L2)          |            L4b (Block 4b: 20%) --[SW4A]--> (to L5)
    (Line A2)                                                       (Line B2)
       |                                                               |
      [A2] -------- Transformer T2 (500 kVA)                          [B2] -------- Transformer T5 (500 kVA)
       |            L2 (Block 2a: 80%)                              |            L5 (Block 5a: 80%)
       |            L2b (Block 2b: 20%) --[SW2A]--> (to L3)          |            L5b (Block 5b: 20%) --[SW5A]--> (to L6)
    (Line A3)                                                       (Line B3)
       |                                                               |
      [A3] -------- Transformer T3 (500 kVA)                          [B3] -------- Transformer T6 (500 kVA)
       |            L3 (Block 3a: 80%)                              |            L6 (Block 6a: 80%)
       |            L3b (Block 3b: 20%) --[SW3A]--> (to L6) *        |            L6b (Block 6b: 20%) --[SW6A]--> (to L3) *
       |                                                               |
       +---------------- [TIE1: Normally Open MV Tie] -----------------+
       * Note: SW3A and SW6A provide inter-feeder cross-routing between Feeder A and Feeder B!
```

### Switching Matrix & Movable Blocks
Each transformer serves two distinct load components:
- **Block $a$ (80%):** Static, non-switchable critical base load connected to LV bus `Lx`.
- **Block $b$ (20%):** Transferable load block on auxiliary bus `Lxb`.

| Transformer | Rated Capacity | Base Bus | Movable Bus | Home Switch (Closed) | Alt Switch (Open) | Target Neighbor | Route Type |
|:-----------:|:--------------:|:--------:|:-----------:|:--------------------:|:-----------------:|:---------------:|:----------:|
| **T1** | 500 kVA | `L1` | `L1b` | `SW1H` | `SW1A` | **T2** | Intra-Feeder |
| **T2** | 500 kVA | `L2` | `L2b` | `SW2H` | `SW2A` | **T3** | Intra-Feeder |
| **T3** | 500 kVA | `L3` | `L3b` | `SW3H` | `SW3A` | **T6** | **Inter-Feeder** |
| **T4** | 500 kVA | `L4` | `L4b` | `SW4H` | `SW4A` | **T5** | Intra-Feeder |
| **T5** | 500 kVA | `L5` | `L5b` | `SW5H` | `SW5A` | **T6** | Intra-Feeder |
| **T6** | 500 kVA | `L6` | `L6b` | `SW6H` | `SW6A` | **T3** | **Inter-Feeder** |

---

## 📈 Machine Learning Load Forecasting

### 1. Dataset & Weather Calibration
- **Smart Meter Telemetry:** Prayas Energy Group **eMARC** (Electricity Monitoring Analysis and Reporting for Consumers) dataset comprising 15-minute resolution load blocks from June 1, 2019 to March 24, 2020 (filtered before COVID lockdowns).
- **Geographic Representation:** Consumer clusters mapped to regional distribution nodes in **Pune City** (T1, T2), **Pune** (T3), **Aurangabad** (T4), **Kanpur Rural** (T5), and **Gonda** (T6).
- **Weather Telemetry:** Real-time synchronized hourly 2-meter air temperature and calculated Cooling Degree Days ($\text{CDD} = \max(0, T - 22^\circ\text{C})$) retrieved from the Open-Meteo Historical Archive API.

### 2. Feature Matrix (Strict 24-Hour Horizon)
To ensure zero lookahead bias and emulate real-world Day-Ahead scheduling, all features for target hour $t$ strictly use information available at $t - 24\text{ h}$:
- **Calendar:** `hour` ($0\dots23$), `dow` ($0\dots6$), `is_weekend` (binary flag).
- **Lags:** `lag_24` (same hour yesterday), `lag_48` (two days prior), `lag_168` (same hour last week).
- **Rolling Aggregations:** `roll_mean_3`, `roll_mean_24`, `roll_mean_168`, `roll_max_24`.
- **Weather Covariates:** Forecasted temperature `temp`, `temp_lag_24`, and cooling demand `cdd`.
- **Transformer Embedding:** Categorical identifier mapped with mean-load normalization.

### 3. Model Benchmark & Test Accuracy
Evaluated over 1,073 continuous held-out test hours:

| Transformer | Model / Baseline | MAE (kW) | RMSE (kW) | nRMSE (%) | MAPE (%) |
|:-----------:|:-----------------|:--------:|:---------:|:---------:|:--------:|
| **T1** | **XGBoost (Refit)** | **0.42** | **0.61** | **19.99%** | **13.33%** |
| T1 | Same hour yesterday | 0.54 | 0.82 | 27.04% | 16.66% |
| T1 | Same hour last week | 0.53 | 0.76 | 24.88% | 16.84% |
| **T2** | **XGBoost (Refit)** | **0.43** | **0.60** | **20.98%** | **15.03%** |
| T2 | Same hour yesterday | 0.53 | 0.79 | 27.51% | 18.01% |
| **T3** | **XGBoost (Refit)** | **0.47** | **0.64** | **18.53%** | **14.18%** |
| T3 | Same hour yesterday | 0.56 | 0.79 | 22.99% | 16.63% |
| **T4** | **XGBoost (Refit)** | **0.69** | **0.90** | **28.66%** | **27.49%** |
| T4 | Same hour yesterday | 0.88 | 1.18 | 37.52% | 33.69% |
| **T5** | **XGBoost (Refit)** | **0.25** | **0.32** | **31.26%** | **35.52%** |
| T5 | Same hour yesterday | 0.27 | 0.36 | 34.85% | 33.43% |
| **T6** | **XGBoost (Refit)** | **0.63** | **0.85** | **32.95%** | **37.71%** |
| T6 | Same hour yesterday | 0.75 | 1.01 | 39.44% | 40.71% |
| **ALL (Pooled)** | **XGBoost (Refit)** | **0.48** | **0.68** | **25.33%** | **22.59%** |
| ALL (Pooled) | Same hour yesterday | 0.59 | 0.86 | 32.18% | 25.83% |
| ALL (Pooled) | Same hour last week | 0.64 | 0.90 | 33.43% | 28.54% |

MAE and RMSE are in raw cluster kW (the households' measured demand, before scaling to the 500 kVA grid model), which is why they are small. nRMSE and MAPE are scale-free.

> **Aggregation Effect:** When forecasting the total demand of all six transformers, individual consumer volatility smooths out significantly: XGBoost's **nRMSE drops from 25.4% (average per transformer) to 10.8%** for the sum. The persistence baseline shows the same effect (31.6% to 14.1%), so the aggregate is easier to forecast for any method.

### 4. Peak Hours (`docs/metrics_peak.csv`)
Overloads happen at the peaks, so `evaluate.py` also scores the top 10% of hours of each transformer's actual load in the test period (648 transformer-hours):

| Model | MAE (kW) | Bias (kW) | MAPE (%) |
|:---|:---:|:---:|:---:|
| **XGBoost (refit)** | **0.96** | **−0.91** | **20.4%** |
| Same hour yesterday | 1.01 | −0.78 | 21.4% |
| Same hour last week | 1.11 | −0.88 | 25.2% |

At the peaks almost all of the error is **under-forecasting** (bias −0.91 kW of a 0.96 kW MAE), and the model's lead over yesterday's value shrinks to one point of MAPE. A point forecast therefore tends to arrive below the real peak, which is what the intervals below are for.

### 5. Forecast Intervals (`ml/intervals.py`, `data/forecast_intervals.csv`)
Each forecast gets a P10–P90 band from **conformal calibration**: the ratio *actual / forecast* is collected per transformer and per 6-hour block of the day, and its 10th and 90th percentiles scale the forecast. For a target hour *t* the ratios come from the last 28 days of hours whose actual value was already known when the forecast was issued (*t* − 24 h); a test checks that changing any later actual value leaves the band unchanged.

| Method (test period) | Coverage of P10–P90 (target 80%) | Below P10 | Above P90 |
|:---|:---:|:---:|:---:|
| Static (validation-period ratios only) | 75.8% | 8.6% | 15.6% |
| **Rolling 28 days (used)** | **78.3%** | 9.4% | 12.3% |

Coverage per transformer, the pinball loss of each bound and the band width are in `docs/metrics_intervals.csv`; the static ratios are in `docs/interval_calibration.csv`. **Disclosure:** the window length was chosen by comparing 7, 14, 28 and 56 days on the test period (coverage 74.4%, 76.7%, 78.3%, 78.2%), so the 78.3% is slightly optimistic; 28 days is also the natural choice (a whole month, every weekday four times).

### 6. Does the Forecast Catch Overloads? (`grid/overload_skill.py`, `docs/metrics_overload.csv`)
For every test hour and scenario, OpenDSS is solved three times (point forecast, P90 bound, actual load) and each transformer-hour is classed as overloaded or not:

| Scenario | Flag raised on | Precision | Recall |
|:---|:---|:---:|:---:|
| S2 Heatwave | point forecast | 100% | **12.2%** |
| S2 Heatwave | P90 bound | 22.4% | **80.5%** |
| S3 EV surge | point forecast | 50.0% | **18.6%** |
| S3 EV surge | P90 bound | 35.6% | **96.6%** |
| S4 Severe stress | point forecast | 69.1% | 57.2% |
| S4 Severe stress | P90 bound | 43.0% | 89.7% |

With the point forecast most real overloads in S2 and S3 are **missed**, a direct effect of the peak under-forecasting. Flagging on the P90 bound catches 80–97% of them at the cost of more false alarms. The dashboard therefore classes **forecast risk on the P90 loading** (`intervals.use_upper_for_risk` in `config/forecasting.yaml`). **Design decision:** the rerouting decision itself still uses the point forecast, so switching only happens for overloads the model actually predicts; the P90 bound raises warnings but never switches by itself. The closed-loop results below show what this choice costs.

### 7. Model Metadata, Registry and Cold Start
- `ml/model_xgb.meta.json` records the model version (`xgb-24h-v1`), feature list, parameters, train/validation/test windows, test metrics and the SHA-256 of `model_xgb.json`. `ml/metadata.py` refuses to load a model whose file no longer matches its metadata; the dashboard registers the model in the database's `model_registry`.
- `python ml/cold_start.py` checks that a model, its metadata and the forecasts exist and builds whatever is missing in order (`prepare_data` → `train_xgb` → `evaluate` → `intervals`); with everything present it does nothing (`--force` retrains).

### 8. Forecasting Step by Step and Scoring as Time Passes
- `ml/forecaster.py`: `Forecaster().forecast_at(t, history)` produces the 24 h-ahead forecast for one hour from a history table, with every load after *t* − 24 h blanked before the features are built. It gives exactly the batch result (tested) and takes about 30 ms; Phase 5's orchestrator will call it every simulated hour.
- The database stores each forecast as *pending*; `Store.score_forecasts(now)` scores the ones whose hour has passed (error = forecast − actual) and `Store.forecast_performance()` gives MAE, RMSE, nRMSE and bias per transformer so far. The dashboard does this as the replay clock moves (expander **Forecast accuracy so far**); scored to the end of the test period it reproduces `docs/metrics.csv` per transformer.

### 9. Forecasting v2 Experiments (`ml/train_v2.py`)
`train_v2.py` compares, on the same split, against the unchanged v1 model:
- **Extra features** (`FEATURES_V2`): `ramp_1h` and `ramp_24h` (how fast the load was changing at issue time) and two asset features, `rating_kva` and `share_residential`.
- **Random Forest** on the v2 features, as a second model family.
- **XGBoost quantile regression** (P10/P50/P90 in one model) as an alternative to the conformal band.
- **Weekly retraining backtest**: the test period is forecast in 7-day blocks, each by a model retrained on everything known at the block's first issue time.

Results on the eMARC test period (1,073 hours × 6 transformers, grid kW; `docs/metrics_v2.csv`, `docs/metrics_v2_peak.csv`):

| Model | MAE (kW) | nRMSE (%) | Bias (kW) | Peak-hour MAE (kW) | Peak-hour bias (kW) |
|:---|:---:|:---:|:---:|:---:|:---:|
| XGBoost, v1 features (control, retrained in the same run) | 21.38 | 25.27 | +0.07 | 42.70 | −40.76 |
| XGBoost, v2 features | 21.38 | 25.26 | +0.04 | 42.65 | −40.75 |
| Random Forest, v2 features | 21.64 | 25.63 | −0.03 | 43.34 | −41.66 |
| XGBoost quantile model, median | 21.20 | 25.28 | −1.59 | 44.30 | −42.69 |
| XGBoost v2, retrained every 7 days | 21.35 | 25.28 | −0.34 | 42.62 | −40.59 |
| Same hour yesterday | 25.91 | 31.84 | −0.75 | 44.07 | −33.95 |

The quantile model's P10–P90 band covers **75.9%** of actual values (`docs/metrics_v2_intervals.csv`), below the rolling conformal band's 78.3%, and only 70–72% for T4 and T5.

**Decision: v1 stays the production model and the conformal band stays the interval method.** None of the challengers improves on the v1 feature set by more than 0.02 points of nRMSE, and Random Forest is worse. The ramp features carry little weight (`ramp_24h` 2.6%, `ramp_1h` 1.1% of the importance), and the two asset features carry none, because all six transformers here have the same rating and are all residential; they are kept for the synthetic city of Phase 7, where assets differ. Weekly retraining gains nothing over a 45-day test period. Every model, including the challengers, under-forecasts the peaks by about 41 kW: the shortfall comes from the data (peaks that yesterday's and last week's values do not announce), not from the model family, which is why the risk assessment relies on the P90 bound. A test (`test_v2_is_not_adopted_without_a_real_gain`) flags this decision for review if a rerun ever changes the picture.

The script writes `data/forecast_v2.csv`, the tables above, `docs/backtest_v2.csv`, `docs/feature_importance_v2.csv`, and the models with their metadata in `ml/models/` (not committed, as they are reproducible). It needs `data/processed/clusters.parquet`; the committed tables are re-derived from `data/forecast_v2.csv` by a test, so they can be checked without the eMARC data.

> **Reproducibility note:** retraining v1 from the same `clusters.parquet` on another machine gives a slightly different model (here early stopping chose 129 trees instead of the committed 121, with the same library versions; the same machine always gives the same result at any thread count). The committed model applied to the same data reproduces `data/forecast_all.csv` exactly, and the retrained one scores within 0.3 nRMSE points per transformer (pooled 25.29% vs 25.33%). The published numbers come from the committed model.

---

## ⚙️ Physics-Constrained Rerouting Engine

The rerouting engine (`grid/engine.py`) optimizes network switch configurations under strict physical power flow constraints solved via EPRI OpenDSS.

### Mathematical Formulation
$$\min_{\mathbf{s} \in \mathcal{S}} \max_{i \in \{1,\dots,6\}} \left( \frac{S_{T_i}(\mathbf{s})}{S_{T_i}^{\text{rated}}} \right)$$

Subject to:
1. **Power Flow Solvability:** $\mathbf{f}(\mathbf{V}, \boldsymbol{\theta}, \mathbf{P}, \mathbf{Q}) = 0$ (AC power flow converges).
2. **Voltage Tolerances:** $0.95\text{ pu} \le |V_b| \le 1.05\text{ pu} \quad \forall b \in \mathcal{B}_{\text{MV}} \cup \mathcal{B}_{\text{LV}}$.
3. **Transformer Thermal Capacity:** $S_{T_i} \le 100\% \times S_{T_i}^{\text{rated}} \quad (500\text{ kVA})$.
4. **Feeder Line Ampacity:** $I_l \le I_l^{\text{rated}} \quad (220\text{ A})$.
5. **Radiality & Safety:** Break-before-make switching without closed loops or backfeeding.

### Algorithm Flow
1. **Evaluate Baseline:** Set forecasted/active loads. Solve AC power flow. If all transformers are at or below the 85% threshold and no voltage or line limit is violated, return `status: no_action`.
2. **Identify Bottlenecks:** Isolate overloaded transformers and any voltage/line violations.
3. **Greedy Round (single actions):** Test each admissible action (LV block moves; MV transfers when enabled). Each candidate is first checked for radiality, then solved in OpenDSS and checked against every limit. Select the safe candidate with the lowest peak loading; within 0.001 points, lower losses wins.
4. **Pairwise Fallback:** If no single action improves the peak by at least 0.05 points (or removes the voltage/line violation), test all pairs of actions.
5. **Convergence or Reversion:** If the peak is at or below the threshold and no limit is violated, return `status: rerouted` with the target configuration and an ordered switching plan. Otherwise restore the original switches and return `status: no_safe_option` with the best partial configuration and the binding constraint.

Every evaluated candidate is returned with its status (`selected`, `feasible`, `rejected`) and reason (e.g. "T6 would be at 103.5% (> 100% limit)"; "would still be at" when the problem existed before switching).

---

## 🧪 Stress Scenarios & Experimental Results

The platform was subjected to extensive validation over **1,073 continuous test hours** across 5 grid operating regimes:

| Scenario | Description | Overload Hours | Total Tx Hours Over | Successfully Rerouted | No Safe Option | Resolution Rate (%) | Mean Peak Before (%) | Mean Peak After (%) | Mean Decision Latency |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **S1 Normal Day** | Unperturbed test period demand | 0 | 0 | 0 | 0 | n/a (no overloads) | — ¹ | — ¹ | 7.0 ms |
| **S2 Heatwave** | Feeder A (T1–T3) scaled by $\times 1.6$ | 5 | 5 | 5 | 0 | **100.0%** | 87.4% | 74.5% | 6.0 ms |
| **S3 EV Evening Surge** | T4 $+200\text{ kW}$ between 18:00–22:00 | 23 | 22 | 23 | 0 | **100.0%** | 87.3% | 69.5% | 5.0 ms |
| **S4 Severe Stress** | System-wide surge (all transformers $\times 2.5$) | 431 | 974 | 151 | 280 | **35.0%** | 105.5% | 79.7% | 7.0 ms |
| **S5 Edge Case** | Catastrophic localized spike (T3 $\times 4.0$) | 882 | 879 | 388 | 494 | **44.0%** | 118.0% | 75.4% | 9.0 ms |

¹ The peak columns average over overload hours only, and S1 has none. Across all S1 hours the highest transformer loading averages 32.7% (maximum 56.3%). Decision latency is machine-dependent; the figures above are from the original development laptop.

### Mitigation When No Complete Fix Exists

When no configuration brings every transformer under the threshold, the engine's best partial configuration is still offered, as a **mitigation**, if it respects every hard limit and is safer than the present one (it removes a hard-limit violation or lowers the peak). It is always labelled "Mitigation, not a full fix" and names what it removes and what stays above the threshold; it is never reported as a successful reroute. From `docs/results_mitigation.csv`:

| Scenario | No Safe Option (hours) | Mitigation Available | Removes a Transformer Overload > 100% | Mean Peak Before → After (mitigated hours) |
|:---|:---:|:---:|:---:|:---:|
| **S4 Severe Stress** | 280 | 128 | 74 | 104.2% → 91.0% |
| **S5 Edge Case** | 494 | 168 | 168 | 115.5% → 91.9% |

S1–S3 have no such hours. In recommend mode the operator approves a mitigation like any other recommendation; in automatic mode it is applied by itself, since it only moves the network to a state that respects every hard limit. Its confidence is the probability that every transformer stays under the 100% hard limit.

### Worst-Hour Stress Analysis

| Scenario | Worst Timestamp | Engine Status | Peak Before (%) | Peak After (%) | Min Voltage Before (pu) | Min Voltage After (pu) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **S2 Heatwave** | 12 Mar 08:00 | `rerouted` | 88.6% | **79.6%** | 0.9739 pu | 0.9760 pu |
| **S3 EV Surge** | 29 Feb 21:00 | `rerouted` | 91.5% | **72.9%** | 0.9748 pu | 0.9798 pu |
| **S4 Severe Stress** | 21 Feb 06:00 | `no_safe_option` | 143.8% | 143.8% | 0.9581 pu | 0.9581 pu |
| **S5 Edge Case** | 17 Feb 07:00 | `no_safe_option` | 200.6% | 200.6% | 0.9405 pu | 0.9405 pu |

Minimum voltage is the lowest phase voltage on any bus, including the 0.415 kV customer side. An earlier version checked only the 11 kV buses; correcting this changed no rerouting decision, only these voltage columns (see `project/docs/archive/README.md`). In the S5 worst hour T3's customer side is below the 0.95 pu limit.

> **Key Finding:** In realistic high-stress scenarios (S2 heatwave and S3 EV charging surges), the engine achieved a **100% resolution rate**, eliminating all transformer overloads and bringing peak utilization well below the 85% safety boundary, while the lowest customer-side voltage rose slightly (0.974 → 0.976 pu and 0.975 → 0.980 pu in the worst hours).

---

## 🖥 Operator Dashboard (Streamlit & Plotly)

The project includes an operator-grade web console built with **Streamlit** and **Plotly**:

1. **Top KPI Ribbon:** overloaded transformer count, peak loading, lowest bus voltage (all buses, including the customer side), network losses, and the forecast nRMSE.
2. **Interactive Topology Visualizer:** transformers coloured by risk class (green normal, amber elevated, orange approaching, red predicted overload, dark red measured overload); dotted grey for open switches; blue for a closed tie, a moved block or a transferred feeder section; dashed MV segments when a sectionaliser is open.
3. **Load Forecasting Horizon Plot:** 72-hour window with actual demand, forecast, the scenario-adjusted forecast, the P10–P90 band (with the scenario applied to it too) and the overload line.
4. **Risk Now:** for every transformer, the forecast, P90 and measured loading with their risk classes side by side (forecast risk uses the P90 loading), plus voltage and line alerts. Below it, **Forecast accuracy so far** scores the forecasts whose hour has passed.
5. **Day-Ahead Overload Scanning Matrix:** which transformer overloads first, for how many hours, and how badly (`CRITICAL` above 100%, `WARNING` above the threshold).
6. **Decision Support** (depends on the operating mode chosen in the sidebar):
   - *Monitor only:* risks and alerts, no switching proposed.
   - Recommendation kinds: *reroute* (fixes every overload), *restore* (back to normal once loads fall), *mitigate* (no full fix exists; a safer configuration that respects every hard limit), *no safe option* (nothing safer exists; the binding constraint is explained).
   - *Recommend:* **Get recommendation** produces a recommendation (ID, summary, numbered switching plan with switch IDs, before/after loading, losses, lowest voltage, confidence, constraint checks, every candidate and why it was rejected). **Approve** applies it; **Reject** discards it. A recommendation computed for another hour or scenario cannot be approved.
   - *Automatic:* every change of time or settings is assessed and safe reconfigurations (or a return to normal) are applied by themselves.
   - **EMERGENCY STOP** halts automation at once (mode drops to monitor-only, pending proposals are cancelled) and leaves the switches as they are; **Roll back last change** and **Reset to normal configuration** remain available to the operator.
7. **Manual switching** of any LV move (and MV transfer when enabled); unsafe configurations (loops, unsupplied loads) are refused.
8. **Audit log** (from the database, so it survives restarts), in four tabs: *Events* (filter by simulated day, transformer, event type and severity), *Recommendations* (with the candidates of any one), *Switching operations* and *Alerts*.

This replay dashboard has its own controller and time slider. The live view of the simulation service (`app/live.py`) is described in the next section; Phase 6 merges both into one multi-page dashboard.

---

## 🔁 Closed-Loop Simulation Service

The replay dashboard answers "what if?" for one hour at a time. The simulation service runs the whole system **hour after hour on a clock**, the way a deployed digital twin would, with every result written to the database.

### One Step of the Loop (`sim/orchestrator.py`)
Step *t* is the boundary at the end of hour *t*:

| # | Stage | What happens |
|:-:|:---|:---|
| 1 | Measure | The meters' loads for hour *t* (scenario applied), which ran in the configuration then in force |
| 2 | Power flow | OpenDSS on those loads in that configuration, and in the normal configuration (what would have happened without control) |
| 3 | Forecast | The 24 h-ahead forecast for hour *t* + 1 (issued 23 h earlier) and its P90 bound; forecasts whose hour has passed are scored |
| 4 | Detect | Risk classes for hour *t* (forecast against measurement) and voltage/line alerts from the measured state, stored as alerts |
| 5 | Validate | If the next hour's forecast breaks a limit, or the network is not in its normal configuration, the controller evaluates the candidates |
| 6 | Apply or propose | *monitor*: nothing; *recommend*: a proposal waits for the operator and the clock pauses so it cannot go stale; *auto*: applied before hour *t* + 1 begins |
| 7 | Record | One `sim_steps` row: what was measured, forecast and decided, and how long each stage took |

Deciding at the boundary means a switching change, automatic or approved, is in force for the whole hour it was computed for, and the next step measures its effect. Forecasts come from the committed file by default (`forecast_source: precomputed`, reproducible anywhere); with `live` the loop calls `forecast_at(t)` from `data/processed/clusters.parquet` at run time and issues each forecast 24 h ahead (a test checks that both give the same values).

### Simulation Clock (`core/clock.py`, `config/simulation.yaml`)
Simulated time and wall-clock time are separate. Controls: resume, pause, single step, jump to any hour, restart (a new run from the beginning) and speed: any factor (3600 = one simulated hour per second), `accelerated` (one simulated **year in 24 hours**, i.e. a 60-minute step every 9.863 s, or a 15-minute step every 2.466 s once the synthetic city of Phase 7 runs at 15 minutes) or `max`. The clock measures the speed it actually achieves. If a step overruns its slot, it carries on from the present at the same pace rather than bursting through the missed steps (`catch_up: true` changes that) and logs a "falling behind" event.

### Background Service (`sim/service.py`)
- Runs independently of any browser; **the database is the one authoritative state**: clock, run number, scenario and heartbeat (`sim_state`), every step (`sim_steps`), and the controller's mode, configuration, pending proposal and emergency stop.
- It is steered only through **commands** written to the database (`commands` table): by the live dashboard (`app/live.py`) or from a terminal (`sim/control.py`). Commands are executed in order and each records its result or the reason it was refused (for example, approving a proposal after the scenario changed).
- **Restart recovery:** after a crash or Ctrl+C, the next start continues with the step after the last one recorded. A test kills the service process part way through S3 and restarts it: every hour is recorded exactly once and the outcome equals the uninterrupted run.
- Only one service per name runs at a time: a second one refuses to start while the first one's heartbeat is fresh.

### Closed-Loop Results (`sim/closed_loop.py`, `docs/closed_loop.csv`)
Each benchmark scenario run hour by hour in automatic mode over the 1,073 test hours. Unlike the per-hour benchmark above, a reroute stays in force until the controller restores the normal configuration, decisions are taken an hour ahead on the forecast, and the effect is judged on what the meters then measure:

| Scenario | Reroutes / restores / mitigations | Switch operations | Measured transformer-hours > 85%: no control → with control | Reduction | Hours with a transformer > 100%: no control → with control |
|:---|:---:|:---:|:---:|:---:|:---:|
| S1 Normal day | 0 / 0 / 0 | 0 | 0 → 0 | – | 0 → 0 |
| S2 Heatwave | 5 / 5 / 0 | 20 | 41 → 37 | 9.8% | 10 → 8 |
| S3 EV surge | 18 / 18 / 0 | 72 | 59 → 48 | 18.6% | 5 → 3 |
| S4 Severe stress | 115 / 90 / 87 | 500 | 1,177 → 1,150 | 2.3% | 364 → 307 |
| S5 Edge case | 51 / 62 / 12 | 250 | 803 → 554 | 31.0% | 566 → 349 |

In monitor mode the loop sees exactly the forecast overloads of the published benchmark (22 transformer-hours in S3), which a test checks. The reductions are modest for S2 and S3 because the control can only act on overloads the point forecast predicts, and it predicts few of the real ones (recall 12–19%, see "Does the Forecast Catch Overloads?"). This is the measured cost of the design decision to switch on the point forecast only, and the motivation for the short-horizon forecasting of Phase 9.

### Performance (`sim/benchmark.py`, `docs/performance.csv`)
Measured on the busiest scenario (S4, automatic mode, 1,073 steps; a 2-CPU Linux container):

| Measure | Value |
|:---|:---|
| Step time, mean / 95th percentile | 37 ms / 58 ms (power flow 16 ms, forecast and scoring 11 ms, decision 9 ms) |
| Throughput | 27 steps/s: a simulated year of hourly steps in about 5.4 minutes |
| Accelerated mode (1 year per 24 h) | uses 0.4% of the 9.86 s budget per hourly step (2.47 s per 15-minute step later) |
| Clock accuracy at 36,000× | achieved 36,023× (0.06% error) |
| Memory | 176 MB resident; growth about 1.4 MB per 1,000 steps |
| Database growth | about 1.7 KB per step, about 14 MB per simulated year |

Run `python sim/benchmark.py` to measure your own computer.

---

## 🧩 Configuration, Asset IDs & Logging

### Configuration (`project/config/`)
Every setting that used to be a constant in the code now lives in one of four YAML files, and `core/config.py` checks each file when it is loaded: an invalid value (a threshold above the hard limit, an unknown transformer, a malformed ID, a split that leaves no test data) stops the program with a message naming the file and key.

| File | Contents |
|:---|:---|
| `grid.yaml` | Substation, feeders, transformers (ratings, buses, loads), switches with their normal state, the 20% movable block and the move table |
| `operations.yaml` | 85% overload threshold, 100% hard limit, 0.95–1.05 pu voltage band, line limit, greedy stop rule, dashboard slider range and colour bands |
| `scenarios.yaml` | Benchmark S1–S5, the quick flat-load scenarios and the dashboard presets — defined once |
| `forecasting.yaml` | Data window, cleaning rules, household-to-transformer clusters, weather coordinates, split, XGBoost settings, 340 kW scaling target, interval method and window, v2 experiment settings |
| `storage.yaml` | Database URL (SQLite file by default), batch size, retention, labels for the imported forecasts |
| `simulation.yaml` | Simulation clock (range, step, speed, accelerated factor, catch-up), orchestrator (scenario, first mode, forecast source, pause on recommendation) and service settings (polling, heartbeat) |

`network.dss` still holds the electrical model that OpenDSS solves; tests check that `grid.yaml` agrees with it (transformers, ratings, buses, loads, switch states and the buses each move connects). Set `LFDR_CONFIG_DIR` to point the whole project at another config folder.

**Scenarios are defined once.** The benchmark (`run_results.py`), the quick script (`grid/scenarios.py`) and the dashboard all use `core/scenarios.py`, and a test checks that the dashboard presets at their default slider values reproduce benchmark S1–S4 exactly.

### Asset IDs
Assets carry IDs in the research specification's scheme, while the short names stay in OpenDSS and on the dashboard:

| Asset | IDs | Example |
|:---|:---|:---|
| Substation | `SUB-001` | `SUB` |
| Feeders | `FD-001`, `FD-002` | A, B |
| Transformers | `DT-001` … `DT-006` | `T1` → `DT-001` |
| Switches | `SW-001` … `SW-015` | `SW1H` → `SW-001`, `TIE1` → `SW-013`, `A2` → `SW-014` |
| Meters | `MTR-00001` … | one per household kept by `prepare_data.py` |

`core/ids.py` provides the registry and conversions. The forecast files carry an `asset_id` column, and `prepare_data.py` writes `data/processed/meter_registry.csv`, linking every household's meter ID to its transformer.

### Database
On first start the dashboard creates `project/data/twin.db` (SQLite), imports `data/forecast.csv` (forecasts and measured loads) and, if present, `data/processed/meter_registry.csv`, then restores the controller from it. To start from a clean history, stop the dashboard and delete `data/twin.db`. The file is local and gitignored.

| Table | Holds |
|:---|:---|
| `assets`, `meters` | Network assets and the meter registry, with their IDs |
| `forecasts`, `measurements` | Forecasts per model version and horizon (pending until scored, then with their error); measured loads per source, with a quality flag |
| `forecast_quantiles` | Interval bounds (P10, P90) per forecast hour, model version and method |
| `recommendations`, `candidates` | Every recommendation (plan, before/after, checks, losses, confidence, binding constraint) and every option evaluated for it |
| `actions` | Every switch operation executed: who/what triggered it (operator, auto, manual, rollback, reset) |
| `events`, `event_assets` | The audit log with severity, and which transformers and switches each event concerns |
| `alerts` | Risk and voltage/line alerts as they were observed |
| `controller_state` | Mode, configuration, emergency stop and undo stack (one row per controller: `default` for the replay dashboard, `service` for the simulation service) |
| `sim_state`, `sim_steps` | The simulation service's clock, run, scenario and heartbeat; one row per completed step |
| `commands` | Commands to the service and their results |
| `model_registry` | Forecasting model versions with features, windows, metrics and file hash |

To use PostgreSQL instead, set `url` in `config/storage.yaml` (or the `LFDR_DB_URL` environment variable) to e.g. `postgresql+psycopg://user:password@localhost:5432/feeder` and `pip install "psycopg[binary]"`. The code uses SQLAlchemy Core and dialect-aware inserts, so nothing else changes.

### Logging
`core/logging_setup.py` gives every module the same log format. Set `LFDR_LOG_LEVEL=DEBUG` to see each rerouting decision, or `LFDR_LOG_FILE=path` to also write a file. The dashboard always writes `logs/dashboard.log`, the simulation service `logs/service.log`.

---

## 📂 Repository Structure

```
Load-Forecasting-Dynamic-Feeder-routing/
├── README.md                      # Primary project documentation (this file)
├── LICENSE                        # MIT licence
├── requirements.txt               # Dependency ranges the code works with
├── requirements-lock.txt          # Exact versions that reproduced docs/results.csv and docs/metrics.csv
└── project/
    ├── config/                    # All settings, in YAML (see "Configuration, Asset IDs & Logging")
    │   ├── grid.yaml              # Assets, IDs, ratings, switches, movable-load scheme
    │   ├── operations.yaml        # Overload threshold, voltage/line limits, dashboard bands
    │   ├── scenarios.yaml         # Benchmark, quick and dashboard scenarios (defined once)
    │   ├── forecasting.yaml       # Data window, cleaning rules, clusters, split, model settings
    │   ├── storage.yaml           # Database URL, batch size, retention
    │   └── simulation.yaml        # Simulation clock, orchestrator and service settings
    ├── storage/                   # Database layer
    │   ├── schema.py              # Tables (SQLAlchemy Core; SQLite or PostgreSQL)
    │   └── store.py               # Store: every read and write of the database
    ├── core/                      # Shared building blocks used by every module
    │   ├── config.py              # Loads and validates config/*.yaml
    │   ├── ids.py                 # Asset ID registry (SUB-001, FD-001, DT-001, SW-001, MTR-00001)
    │   ├── scenarios.py           # The one scenario implementation (benchmark, script, dashboard)
    │   ├── clock.py               # Simulation clock: pause, resume, step, speed, accelerated mode
    │   ├── logging_setup.py       # Project-wide logging
    │   └── paths.py               # Folder locations
    ├── app/                       # Streamlit web application & visualization
    │   ├── app.py                 # Main operator dashboard entry point
    │   ├── plots.py               # Plotly network topology & load curve renderers
    │   └── live.py                # Live view of the simulation service (reads the database, sends commands)
    ├── sim/                       # Closed-loop simulation (Phase 5)
    │   ├── orchestrator.py        # One step: measure, power flow, forecast, detect, validate, apply, record
    │   ├── sources.py             # Demand and forecast sources (classic eMARC data; precomputed or live forecasts)
    │   ├── service.py             # Background service: clock, commands, heartbeat, restart recovery
    │   ├── control.py             # Command-line control of the service (status, pause, speed, approve, ...)
    │   ├── closed_loop.py         # Closed-loop results for S1-S5 -> docs/closed_loop.csv
    │   └── benchmark.py           # Step time, throughput, memory, clock accuracy -> docs/performance.csv
    ├── grid/                      # Power systems simulation & reconfiguration
    │   ├── engine.py              # GridEngine: OpenDSS power flow, switching actions, rerouting search
    │   ├── risk.py                # Five risk classes, voltage/line alerts, recommendation confidence
    │   ├── overload_skill.py      # Overload precision/recall: point forecast vs P90 vs actual
    │   ├── decisions.py           # Recommendations, operating modes, approval, rollback, emergency stop
    │   ├── network.dss            # OpenDSS circuit definition (11 kV dual-feeder)
    │   ├── scenarios.py           # Quick check of the engine on a flat load
    │   ├── test_day1.py           # Basic power flow & manual switch test script
    │   └── run_results.py         # Full 1,073-hour simulation benchmark script
    ├── ml/                        # Machine learning forecasting pipeline
    │   ├── prepare_data.py        # Smart meter extraction, weather sync, meter registry
    │   ├── features.py            # Feature engineering (lags, rolling stats, CDD; v2 ramps and asset features)
    │   ├── train_xgb.py           # XGBoost training with early stopping & refit
    │   ├── evaluate.py            # Accuracy benchmarking vs persistence models, peak-hour metrics
    │   ├── intervals.py           # Conformal P10-P90 intervals (rolling 28 days), coverage, pinball loss
    │   ├── forecaster.py          # forecast_at(t): one 24 h-ahead forecast from a history table
    │   ├── metadata.py            # Model metadata (version, features, windows, metrics, SHA-256), registry
    │   ├── cold_start.py          # Build whatever is missing: data -> model -> forecasts -> intervals
    │   ├── train_v2.py            # v2 experiments: new features, Random Forest, quantile XGBoost, weekly retraining
    │   ├── model_xgb.json         # Serialized production XGBoost model (v1)
    │   ├── model_xgb.meta.json    # Its metadata
    │   └── models/                # v2 experiment models (created by train_v2.py, not in the repository)
    ├── data/                      # Data assets & outputs
    │   ├── raw/                   # eMARC Excel/CSV files & weather cache (not in the repository)
    │   ├── processed/             # clusters.parquet, meter_registry.csv (not in the repository)
    │   ├── forecast.csv           # Final held-out test predictions (interface contract)
    │   ├── forecast_all.csv       # Complete train/val/test predictions
    │   ├── forecast_intervals.csv # P10/P90 bounds for every test hour
    │   ├── forecast_v2.csv        # v2 experiment forecasts for the test period
    │   ├── scale_factors.json     # Normalization & grid scaling parameters
    │   └── twin.db                # Local database, created on first run (not in the repository)
    ├── docs/                      # Documentation artifacts & figures
    │   ├── metrics.csv            # Comprehensive model accuracy metrics
    │   ├── metrics_peak.csv       # Accuracy on the top 10% of hours
    │   ├── metrics_intervals.csv  # Interval coverage, pinball loss, width (static and rolling)
    │   ├── interval_calibration.csv # Static conformal ratios per transformer and block of the day
    │   ├── metrics_overload.csv   # Overload precision/recall per scenario (point vs P90)
    │   ├── metrics_v2*.csv, backtest_v2.csv, feature_importance_v2.csv  # v2 experiment results
    │   ├── results.csv            # S1-S5 scenario summary over 1,073 hours
    │   ├── results_worst_hour.csv # Worst-hour peak reduction analysis
    │   ├── results_mitigation.csv # Hours with no full fix where a safe mitigation exists
    │   ├── closed_loop.csv        # S1-S5 run hour by hour in automatic mode: measured overloads without/with control
    │   ├── performance.csv        # Step time per stage, throughput, memory, database growth, clock accuracy
    │   ├── archive/               # Superseded result tables, with an explanation
    │   └── fig_actual_vs_pred.png # Actual vs predicted multi-transformer plot
    ├── tests/                     # Automated pytest suite (see "Running the Tests")
    │   ├── test_golden_results.py # Reproduces docs/results.csv and docs/metrics.csv exactly
    │   ├── test_engine.py         # Rerouting outcomes, safety limits, switch restoration
    │   ├── test_network.py        # Network structure, default switch state, operating point
    │   ├── test_config.py         # Config validation; grid.yaml agrees with network.dss
    │   ├── test_ids.py            # Asset ID scheme and registry
    │   ├── test_scenarios.py      # Scenario rules; dashboard defaults equal the benchmark
    │   ├── test_engine_v2.py      # Candidates, reasons, binding constraints, losses, P/Q/S, feeders, MV transfers
    │   ├── test_risk.py           # Risk classes, alerts, confidence
    │   ├── test_decisions.py      # Modes, approval, stale proposals, rollback, emergency stop
    │   ├── test_dashboard.py      # The dashboard run headless: recommend, approve, roll back, e-stop, auto, restart
    │   ├── test_storage.py        # Schema, imports, filters, batching, retention, restart restores everything
    │   ├── test_simulation.py     # Clock, orchestrator, service commands, kill-and-restart recovery, live view, benchmark
    │   ├── test_features.py       # No-leakage checks for the 24 h-ahead features
    │   ├── test_forecasting_v2.py # Metadata, peak/overload metrics, intervals, scoring, forecast_at, v2, cold start
    │   ├── synthetic.py           # Synthetic data in the clusters.parquet format (tests only)
    │   └── test_artifacts.py      # Forecast files, scale factors, model and config agree
    ├── scripts/                   # Helper scripts, not part of the pipeline
    │   ├── check_imports.py       # Environment sanity verification
    │   ├── smoke_test.py          # OpenDSS direct solver smoke test
    │   └── exploration/           # One-off scripts used to study the eMARC raw files
    ├── logs/                      # Runtime logs (created on first run, not in the repository)
    └── pytest.ini                 # Test configuration
```

> `data/raw/` and `data/processed/` are not in the repository (they are gitignored). The eMARC smart-meter files must be obtained from Prayas Energy Group and placed in `data/raw/` before `ml/prepare_data.py` and `ml/train_xgb.py` can be re-run. Everything else (dashboard, grid benchmark, evaluation, tests) runs from the committed forecast files.
>
> The dashboard writes `logs/dashboard.log` (operator actions) and, only if OpenDSS crashes the process, `logs/dss_crash.log`. Both are machine-specific and gitignored.

---

## 🚀 Getting Started & Installation

### Prerequisites
- **Python:** Version `3.10` or higher recommended.
- **Operating System:** Windows, Linux, or macOS.

### 1. Clone the Repository
```bash
git clone https://github.com/PrithviKoruche-dev/Load-Forecasting-Dynamic-Feeder-routing.git
cd "Load-Forecasting-Dynamic-Feeder-routing"
```

### 2. Create and Activate Virtual Environment
```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
To reproduce the published numbers with exactly the versions used to verify them, install `requirements-lock.txt` instead.

### 4. Verify System Installation
Run the sanity checks to verify OpenDSS and ML library bindings:
```bash
python project/scripts/check_imports.py
python project/scripts/smoke_test.py
```
*Expected output: All packages print `OK`, and OpenDSS reports `converged: True`.*

---

## 💻 Usage Guide & CLI Commands

### 1. Launch the Operator Dashboard
Launch the interactive web application:
```bash
streamlit run project/app/app.py
```
Open your browser at `http://localhost:8501`.

### 2. Run the Switching Engine & Stress Scenarios
Quick check of the rerouting algorithm on a flat 300 kW load (the "quick" scenarios in `config/scenarios.yaml`):
```bash
cd project/grid
python scenarios.py
```

### 3. Run the Full 1,073-Hour Benchmark
Execute the complete test-period evaluation across all scenarios and generate report CSVs:
```bash
cd project/grid
python run_results.py
```

### 4. Re-train the Machine Learning Model
Requires the eMARC raw files in `project/data/raw/` (see the note under Repository Structure). Re-run data preparation, feature engineering, training, and model export:
```bash
cd project/ml
python prepare_data.py
python train_xgb.py
python evaluate.py
python intervals.py
```
Or let `python cold_start.py` run whichever of these steps are needed. A model retrained on another machine differs slightly from the committed one (see the reproducibility note under Forecasting v2), so the golden tests, which check the published numbers, will then report differences; `git checkout -- ml data docs` (from the `project` folder) restores the committed model, forecasts and tables. To rerun the v2 experiments (needs `data/processed/clusters.parquet`, about 30 s):
```bash
python train_v2.py
```

### 5. Run the Closed-Loop Simulation Service
Two terminals, both in `project/`:
```bash
python sim/service.py                     # terminal 1: the service (Ctrl+C stops it; starting it again resumes)
streamlit run app/live.py                 # terminal 2: the live view, at http://localhost:8501
```
Useful options: `--speed accelerated` (one simulated year per 24 h), `--speed max`, `--mode auto`, `--scenario "S3 EV surge (T4 +200 kW, 18-22 h)"`, `--forecast live` (needs `data/processed/clusters.parquet`), `--paused`, `--new-run`. While it runs, it can also be steered from any terminal:
```bash
python sim/control.py status
python sim/control.py pause                # resume | step | restart | stop
python sim/control.py speed 86400          # one simulated day per second
python sim/control.py jump "2020-02-29 18:00"
python sim/control.py mode auto            # monitor | recommend | auto
python sim/control.py approve REC-0001     # reject REC-0001
python sim/control.py steps 10             # the last 10 steps
```
In recommend mode the clock pauses itself when a switching proposal is waiting; approving or rejecting it (on the live page or with `control.py`) lets the clock run on.

The report tables are produced by:
```bash
python sim/closed_loop.py                  # docs/closed_loop.csv (about 3 min)
python sim/benchmark.py                    # docs/performance.csv (about 1 min, numbers depend on the computer)
```

### 6. Running the Tests
```bash
cd project
python -m pytest                  # full suite, about 5 min (benchmark, closed loop, kill-and-restart, v2 training, cold start)
python -m pytest -m "not slow"    # everything except those, about 45 s
```
`python -m pytest` makes sure the tests run with the Python of the active environment (a bare `pytest` can pick up another environment's copy). The golden tests re-run the scenario benchmark and the forecast evaluation and compare them with `docs/results.csv`, `docs/results_worst_hour.csv` and `docs/metrics.csv` (timing columns excluded); if a change alters any published number, they fail. The dashboard tests start the app headless and drive it through the S3 and S4 worst hours.

> **Windows note:** while loading, the OpenDSS library raises and handles an internal exception (code `0xE0465043`, "FPC" for Free Pascal). Python's crash reporter prints such events as "Windows fatal exception" even though nothing failed. The test setup and the dashboard load OpenDSS before switching the crash reporter on, so this no longer appears.

---

## 📑 API & Interface Contract

### 1. Forecasting Data Contract (`data/forecast.csv`)
The ML forecasting pipeline exposes a simple CSV interface utilized by the grid simulator:
| Column Name | Data Type | Description |
|:---|:---:|:---|
| `timestamp` | `YYYY-MM-DD HH:MM:SS` | Hourly timestamp of target forecast |
| `transformer_id` | `string` (`T1` to `T6`) | Short name of the distribution transformer (as in `network.dss`) |
| `asset_id` | `string` (`DT-001` to `DT-006`) | Asset ID of the transformer (see `config/grid.yaml`) |
| `predicted_kw` | `float` | 24-hour-ahead forecast in kW |
| `actual_kw` | `float` | Real ground-truth demand in kW |

### 2. Grid Engine API (`grid/engine.py`)
```python
def evaluate_and_reroute(loads_kw: dict, threshold: float = 85.0, use_mv: bool = None) -> dict:
    """
    Evaluates transformer loadings and computes the best safe switching configuration.

    Parameters:
        loads_kw (dict): Active power per transformer, e.g. {"T1": 320.0, ..., "T6": 290.0}
        threshold (float): Loading % trigger limit (default from config/operations.yaml)
        use_mv (bool): also consider MV feeder transfers (default from config)

    Returns:
        dict: {
            "status": "rerouted" | "no_action" | "no_safe_option",
            "action": list[str],          # e.g. ["move T4 block b -> T5"]
            "target": list[str],          # action keys of the target configuration, e.g. ["T4>T5"]
            "plan": list[dict],           # ordered switch operations: step, switch, switch_id, operation, action
            "before": dict,               # metrics in the normal configuration (see below)
            "after": dict,                # metrics in the target (or best partial) configuration
            "candidates": list[dict],     # every option evaluated: id, round, actions, status, reason,
                                          #   max_loading, min_voltage_pu, max_line_loading, losses_kw
            "binding": list[str],         # why no safe option exists (empty otherwise)
            "violations_before": list,    # hard-limit violations in the starting state
            "best_partial": list[str],    # no_safe_option only
            "log": list[str],             # step-by-step solver audit trail
            "overloaded": list[str],      # transformers above the threshold
            "compute_s": float            # algorithm execution time in seconds
        }
    Metrics: max_loading, loading_by_tx, min_voltage_pu, min_voltage_bus, max_voltage_pu, max_line_loading,
    losses_kw, and (for before/after) transformers {kw, kvar, kva, loading_pct}, feeders {kw, kvar, kva,
    current_a, loading_pct, min_voltage_pu, transformers}, bus_voltages_pu, line_loading_pct.
    """
```
`GridEngine(grid_cfg, ops_cfg)` builds an engine for any configuration (the module-level functions use one built from `config/`). `grid/decisions.py` wraps it in a `Controller` (modes, approve/reject, rollback, emergency stop, history).

---

## 🔮 Future Enhancements

- [ ] **Multi-Period Lookahead Optimization:** Incorporate a receding-horizon Model Predictive Control (MPC) formulation to penalize excessive switch wear over 24 hours.
- [ ] **Distributed Energy Resources (DERs):** Integrate rooftop Solar PV arrays and Battery Energy Storage Systems (BESS) for localized peak shaving.
- [ ] **Reinforcement Learning (RL):** Benchmark deep Q-networks (DQN) or PPO agents against the greedy heuristic for complex meshed distribution grids.
- [ ] **IEC 61850 / DNP3 SCADA Integration:** Provide standardized protocol gateways for hardware-in-the-loop (HIL) testing with physical microgrid controllers.

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <b>Final Year Engineering Project</b> • Department of Electrical / Computer Engineering<br>
  Developed with ❤️ using Python, OpenDSS, and Streamlit.
</p>