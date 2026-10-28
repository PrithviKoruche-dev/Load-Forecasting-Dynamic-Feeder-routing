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
- [Synthetic City (Phase 7)](#-synthetic-city-phase-7)
- [Meters and Data Quality (Phase 8)](#-meters-and-data-quality-phase-8)
- [Short-Horizon Forecast (Phase 9)](#-short-horizon-forecast-phase-9)
- [Hardening (Phase 10)](#-hardening-phase-10)
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
  - One multi-page app: overview, network with a clickable inspector, sortable transformer table with history, forecast accuracy over time, alerts and decisions, controls, a filterable audit log and data quality, all reading one state from the database; plus a what-if replay sandbox.
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

- **🏙 Synthetic City:**
  - A generated city of 3 substations, 8 feeders and 44 transformers serving 5,083 households, shops, offices, mixed-use buildings and factories, each with a meter, built from one YAML file and a seed.
  - 15-minute demand for a whole year from consumer profiles (households calibrated on the eMARC data), weekdays, holidays, seasons and weather; heatwaves, EV charging and festivals are events in the city's demand.
  - Runs through the same engine, service and dashboard (`--city demo`), within 7% of the accelerated mode's time budget.
  - A 15-minute-ahead forecast trained on the city's own recorded history (nRMSE about 4% against 22% for 24 h persistence) doubles the overloads the control removes under EV charging and catches the heatwave.
  - Separate measurement streams: transformer meters that fail, smart meters whose readings arrive late, get lost or come from stuck meters, and a head-end that estimates, flags and revises. The operator decides on what was measured; outcomes are judged on the true loads.

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

One multi-page web app (`streamlit run app/app.py`, from `project/`). The **live network** pages show what the simulation service is doing (see the next section): they only read the database, and their buttons only send commands to the service, so every page and every browser shows the same, single state. If the service is not running, any page offers a **Start the service** button (it starts as a separate process and keeps running when the dashboard is closed).

| Page | Gap item | What it shows |
|:---|:---:|:---|
| **A Overview** | G3 | Substations, feeders, transformers, switches, meters and capacity; total load, peak loading (with the change made by control), active alerts, configuration; simulated and wall-clock time, season, target and achieved speed, time of the last power flow; the last 48 hours with and without control |
| **B Network** | G4 | Single-line diagram for the hour just measured or the next hour's forecast; click a transformer (or pick it, or a feeder) to inspect rating, load, kVA, measured and forecast loading, P90, risk class, limits and its switching options; feeder current, voltage and the transformers it supplies; every switch's state |
| **C Transformers** | G5 | Sortable table: rating, load, loading %, forecast %, P90 %, next hour %, risk class, time; select a row for that transformer's history (48 h, 7 days or the whole run) |
| **D Forecasting** | G6 | Scored and pending forecasts, MAE, nRMSE, bias; MAE or RMSE per day or week by transformer; any model version in the database (e.g. the live one); forecast against actual with the P10–P90 band |
| **E Alerts and decisions** | G7 | The proposal waiting for the operator with **Approve** / **Reject**; alerts of the hour and their history; every recommendation with its switching plan, before/after bars, checks, confidence and every candidate evaluated with its outcome and reason |
| **F Controls** | G8 | Start/stop the service; resume, pause, single step; speed (incl. accelerated and custom); jump to a date and hour, or a new run from it; scenario; season (shown; demand profiles by season arrive with the synthetic city, Phase 7); operating mode; approval policy; threshold; MV transfers; roll back, reset, emergency stop. Every control starts from the value in force in the service |
| **G Audit log** | G9 | Events (filter by simulated day, transformer, event type and severity), recommendations with candidates, switch operations, alerts and commands with their results; for the live network or the sandbox |
| **What-if replay** | – | The sandbox described below |

**What-if replay (sandbox).** The single-page dashboard of earlier phases is kept as a sandbox page: pick any hour with a slider and study it with its own controller and switch configuration, separate from the live network:

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

Recommendation numbers (REC-nnnn) are unique across the sandbox and the live network, which share the database.

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
- It is steered only through **commands** written to the database (`commands` table): by the dashboard's pages or from a terminal (`sim/control.py`). Commands are executed in order and each records its result or the reason it was refused (for example, approving a proposal after the scenario changed).
- **Restart recovery:** after a crash or Ctrl+C, the next start continues with the step after the last one recorded. A test kills the service process part way through S3 and restarts it: every hour is recorded exactly once and the outcome equals the uninterrupted run.
- Only one service per name runs at a time: a second one refuses to start while the first one's heartbeat is fresh.

### Closed-Loop Results (`sim/closed_loop.py`, `docs/closed_loop.csv`)
Each benchmark scenario run hour by hour in automatic mode over the 1,073 test hours. Unlike the per-hour benchmark above, a reroute stays in force until the controller restores the normal configuration, decisions are taken an hour ahead on the forecast, and the effect is judged on what the meters then measure:

| Scenario | Reroutes / restores / mitigations | Switch operations | Measured transformer-hours > 85%: no control → with control | Reduction | Hours with a transformer > 100%: no control → with control |
|:---|:---:|:---:|:---:|:---:|:---:|
| S1 Normal day | 0 / 0 / 0 | 0 | 0 → 0 | – | 0 → 0 |
| S2 Heatwave | 5 / 5 / 0 | 20 | 41 → 37 | 9.8% | 10 → 8 |
| S3 EV surge | 18 / 18 / 0 | 72 | 59 → 48 | 18.6% | 5 → 3 |
| S4 Severe stress | 115 / 90 / 87 | 1,058 | 1,177 → 1,150 | 2.3% | 364 → 307 |
| S5 Edge case | 51 / 62 / 12 | 250 | 803 → 554 | 31.0% | 566 → 349 |

In monitor mode the loop sees exactly the forecast overloads of the published benchmark (22 transformer-hours in S3), which a test checks. (The first version of this table gave S4 500 switch operations: the count stopped at 500 rows. Corrected in Phase 7; nothing else changed.) The reductions are modest for S2 and S3 because the control can only act on overloads the point forecast predicts, and it predicts few of the real ones (recall 12–19%, see "Does the Forecast Catch Overloads?"). This is the measured cost of the design decision to switch on the point forecast only, and the motivation for the short-horizon forecasting of Phase 9.

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

## 🏙 Synthetic City (Phase 7)

The six-transformer network on real eMARC data stays the validated baseline ("classic mode", every published number above). Phase 7 adds a **generated city** that runs through the same engine, service and dashboard, as a different configuration folder.

### What is Generated (`city/generate.py`, `cities/demo/`)
`cities/demo/city.yaml` describes the city; `python city/generate.py demo` builds it (everything random comes from its `seed`, and a test checks that generating again gives the committed files byte for byte):

| | Demo city |
|:---|:---|
| Substations | 3, each an 11 kV source (`SUB-001` … `SUB-003`; the 33/11 kV substation transformers are not modelled) |
| Feeders | 8 (3 + 3 + 2), cable segments of 0.3–0.8 km (240 mm² Al XLPE), a sectionaliser half-way along each |
| Transformers | 44, rated 250–630 kVA (20,050 kVA in all), each with a fixed block a and a movable block b (as in the classic network) |
| Switches | 100: 44 LV home, 44 LV alt, 8 sectionalisers, 4 normally open ties between feeder ends (2 of them across substations); 8 MV transfer options |
| Consumers | 5,083, each with a meter (`MTR-00001` …) in `consumers.csv` and the database: 4,218 households, 654 shops and offices, 197 mixed-use buildings, 14 factories |

Each transformer gets a dominant consumer type and a mix of consumers, sized so that its normal-year peak (before noise) is 50–80% of its rating. The generator writes `network.dss` (OpenDSS), `consumers.csv` and `config/` (`grid.yaml` with every asset, ID, switch, move, MV transfer and the diagram layout, plus the operations, scenarios, forecasting, storage and simulation settings of the city).

To make the engine work on any such network, it now reads several substations from `grid.yaml` (an island fed by two substations through a closed tie is refused, like a loop), and `rerouting.candidate_scope: relevant` (used by the city) tries only moves that take load off an overloaded transformer or make room for it. On the classic benchmark this gives **the same decision in all 5,365 runs** with 65% fewer candidates evaluated; a test checks the same on busy city steps.

### Demand Model (`city/demand.py`)
Every transformer's load at **15-minute** resolution for a whole year (35,040 steps), from its consumers:

- **Households** follow the real eMARC households: a profile per season, weekday/weekend and hour of day plus a cooling term per °C above 24 °C, fitted by least squares on all 42,768 hours of the eMARC data (`city/calibrate.py` → `city/profiles/residential_emarc.csv`; strongest cooling at night, when air conditioners run). An urban household is taken to use 4× the eMARC sample's average (`demand.residential.scale`, an assumption: the eMARC homes are small consumers). The eMARC data covers June to March, so summer is fitted from March only, and the hotter April and May act through the cooling term.
- **Shops and offices** and **factories** follow hourly profiles from `city.yaml` (opening hours, two shifts), lower on Saturdays and on Sundays and **public holidays**, with cooling while open; **mixed-use** buildings are part household, part shop.
- **Seasons** come from the calendar in `config/simulation.yaml` (winter Dec–Feb, summer Mar–May, monsoon Jun–Sep, transition Oct–Nov); **weather** from monthly mean temperatures, a daily cycle by season and seeded day-to-day variation.
- **Events** change the demand itself rather than scaling a forecast: `diwali_lights` (always on: households use 25% more in the evening for five days), `may_heatwave` (+6 °C for nine days, acting through every cooling term) and `ev_adoption` (15% of households charge a 3.3 kW car in the evening, about 7 kWh a day). Scenarios switch events on: **C1** normal year, **C2** May heatwave, **C3** EV adoption, **C4** both. The weather and the noise do not depend on the events, so scenarios differ only where their events act (tested).

### Forecast in the City
The XGBoost model was trained on the six eMARC clusters, so the city starts with a transparent baseline: the **24 h-ahead persistence forecast** (the load at the same time the day before), with a P10–P90 band from the same rolling conformal calibration as Phase 4, using only loads known at issue time (tested by changing every later load). The service writes each step's meter readings and issues the forecast for 24 h later, so the Forecasting page scores them as time passes. Since Phase 8 the forecast is built from what the transformer meters recorded, not from the true loads. Phase 9 replaced it for the switching decisions with a 15-minute-ahead model trained on the city's own recorded history (see Short-Horizon Forecast); persistence stays available (`forecast_source: persistence`).

### City Results with the Persistence Forecast
May (the peak month, 2,976 steps of 15 minutes), automatic mode, closed loop as in Phase 5, with the operator seeing the loads through the city's meters (Phase 8; outcomes are judged on the true loads) and the 24 h persistence forecast deciding the switching (`docs/closed_loop_city_demo_persistence.csv`; the short-horizon forecast of Phase 9 is the default now, results in its section):

| Scenario | Reroutes / restores / mitigations | Switch operations | Measured transformer-steps > 85%: no control → with control | Reduction | Steps with a transformer > 100% | Forecast nRMSE | P10–P90 coverage |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| C1 Normal year | 19 / 20 / 4 | 160 | 38 → 36 | 5.3% | 0 → 0 | 21.3% | 80.2% |
| C2 May heatwave | 32 / 32 / 31 | 348 | 317 → 320 | −0.9% | 1 → 1 | 22.4% | 79.6% |
| C3 EV adoption (15% of households) | 119 / 33 / 18 | 552 | 485 → 323 | 33.4% | 1 → 1 | 20.9% | 80.3% |
| C4 Heatwave and EV adoption | 122 / 40 / 59 | 800 | 1,031 → 838 | 18.7% | 17 → 11 | 22.0% | 79.6% |

The P10–P90 band is well calibrated (about 80% coverage in every scenario), and the control removes a third of the overloads where they come back every day (EV charging), but **almost none in the heatwave**: persistence cannot foresee the first hot day (on 8 May none of the 122 overloaded steps was forecast, Phase 7) and then expects yesterday's overloads at the wrong times. The point forecast decides the switching (the design decision recorded under Forecasting), so a forecast that follows the weather is what the city needs most. Phase 9 provides it.

### Running the City
```bash
cd project
python city/generate.py demo                                  # only after editing cities/demo/city.yaml
python sim/service.py --city demo                             # or start it from the dashboard
streamlit run app/app.py -- --city demo                       # the dashboard for the city (note the --)
python sim/control.py status --city demo                      # every control.py command takes --city
python ml/train_short.py --city demo                          # the short-horizon forecast (about 1 min)
python sim/closed_loop.py --city demo                         # docs/closed_loop_city_demo.csv (about 25 min)
python sim/closed_loop.py --city demo --restore-margin 0      # without the deadband (docs/deadband_city_demo.csv)
python sim/closed_loop.py --city demo --forecast persistence --restore-margin 0   # docs/closed_loop_city_demo_persistence.csv
python sim/closed_loop.py --city demo --data-quality --restore-margin 0          # docs/data_quality_city_demo.csv (about 50 min)
python sim/benchmark.py --city demo                           # docs/performance_city_demo.csv
```
The city has its own database (`data/city_demo.db`), so its runs never mix with the classic network's. The dashboard pages work unchanged (the Network page draws the city's 44 transformers, 8 feeders and ties from the layout in `grid.yaml`); the what-if replay sandbox, which replays the eMARC forecasts, is not shown for a city.

### City Performance (`docs/performance_city_demo.csv`)
One busy week (heatwave and EV adoption, 672 steps of 15 minutes, automatic mode, with the meters of Phase 8 and the short-horizon forecast of Phase 9) on the same 2-CPU container (Phase 7, without meters: 104 ms per step):

| Measure | Value |
|:---|:---|
| Step time, mean / 95th percentile | 152 ms / 328 ms (meters 30 ms, including building the year's short-horizon forecast once per scenario, about 7 s; power flow 53 ms; forecast and scoring 31 ms; decision 37 ms) |
| Throughput | 6.6 steps/s: a simulated year of 15-minute steps in about 89 minutes |
| Accelerated mode (1 year per 24 h) | uses 6.2% of the 2.47 s budget per 15-minute step |
| Memory | peak about 560 MB while the forecast table is built, then flat (about 2 MB per 1,000 steps after warm-up, 1,500 steps checked in Phase 8) |
| Database | about 110 KB per step before retention; the city keeps 60 days of measurements, alerts and forecasts and 2 days of smart-meter readings (`storage.yaml retention`, applied by the service once per simulated day), and only switch positions away from normal are stored with each step |

Most of the power-flow time is OpenDSS recompiling the circuit before every evaluation (6.5 of 11.4 ms on the city); keeping the circuit loaded between evaluations is the obvious next speed-up if a faster-than-hourly city is needed.


---

## 📡 Meters and Data Quality (Phase 8)

Until Phase 7 the operator saw every transformer's true load at the end of each step. A real operator sees measurements: a transformer meter that sometimes stops reporting, and thousands of smart meters whose readings arrive late, get lost or come from a stuck meter. Phase 8 models these two streams separately (`city/meters.py`, settings in the `meters` section of `city.yaml`). The simulator keeps two views of every step. The **true loads** are what the network carried; the power flow and the outcome (`measured_*` columns, every result table) use them. The **observed loads** are what the operator knew, and the forecast, risk classes and alerts use them.

### The Two Streams
| | Transformer meters (SCADA) | Smart meters (AMI) |
|:---|:---|:---|
| How many | one per transformer (44) | one per consumer (5,083, `MTR-00001` …) |
| Reading | true load × (1 + 1% error) | the consumer's share of its transformer's load, varied ±20% per consumer, less 4% LV losses (together the meters see 96% of the transformer) |
| Arrives | at once | after a lognormal delay (median 2 min; 5% later than 15 min); 1% never arrive |
| Faults | sensor out of service 2 times a year per transformer, about 12 h each (0.3% of transformer-steps) | data concentrator down 4 times a year per transformer, about 6 h each (its readings wait and arrive when it is back); 0.5% of meters get stuck on one value |

Everything is drawn from the city's seed and the interval, so every run sees the same readings, faults and delays.

### The Head-End
The head-end collects smart-meter readings as they arrive. For every transformer and interval it estimates the load from the readings received so far, as (sum of readings ÷ expected share of those consumers) ÷ (1 − LV losses). Each estimate gets a quality flag: **good** (at least 98% of the expected load has reported), **estimated** (at least 70%), **poor** or **missing**. An estimate is revised whenever late readings for its interval arrive. A meter that sends 8 identical non-zero readings in a row is flagged **suspect** and left out of every estimate in the 48-hour window, including earlier ones, which are revised. Readings more than 48 hours late are counted as lost. Its state is rebuilt by replaying that window after a restart or a jump in time; a test checks that this gives the same state as running through it.

### What the Operator Uses
For each transformer and step, in this order:
1. The transformer meter.
2. If its sensor is out, the smart meters' estimate of the interval before, when it is good or estimated. The current interval's readings have not arrived yet.
3. Otherwise the forecast, or the last value seen.

The 24 h-ahead persistence forecast is built from what the transformer meters recorded; a gap is filled from the same time a day earlier. Forecasts are scored against the transformer meters (`dt-scada`). The database keeps:
- the transformer meters,
- the smart-meter estimates (`ami-estimate`, one row per transformer and interval, revised in place),
- the exception readings: those later than one interval, and those from suspect meters (`meter_readings`). Storing every reading costs about 95 MB per simulated day, so `meters.store_readings` chooses `all`, `exceptions` or `none`.

The schema is now version 4, and an older database is upgraded when opened. A city without a `meters` section, or with `enabled: false`, and the classic network still give the operator the true loads.

The new **Data quality** page shows:
- where each transformer's value came from,
- seen against true loading,
- the last two hours of smart-meter estimate quality,
- late readings, readings in transit, suspect meters and concentrators down, over time,
- the exception readings by transformer.

The **Transformers** page now shows the loading the operator saw, with its source and the true loading beside it.

### Perfect Measurement against the Meters (`docs/data_quality_city_demo_persistence.csv`)
May, automatic mode, the persistence forecast of Phase 8, each scenario run twice: once with the operator seeing the true loads, once through the meters. (With the Phase 9 forecast: `docs/data_quality_city_demo.csv`, below.)

| Scenario | Switch operations: perfect → meters | Transformer-steps > 85%, with control: perfect → meters (no control) | Seen error, mean / 99th pct (points) | Overloads missed / false |
|:---|:---:|:---:|:---:|:---:|
| C1 Normal year | 92 → 160 | 38 → 36 (38) | 0.33 / 1.5 | 8 / 15 |
| C2 May heatwave | 236 → 348 | 316 → 320 (317) | 0.35 / 1.5 | 31 / 36 |
| C3 EV adoption | 536 → 552 | 324 → 323 (485) | 0.34 / 1.5 | 28 / 33 |
| C4 Heatwave and EV adoption | 732 → 800 | 840 → 838 (1,031) | 0.35 / 1.6 | 66 / 64 |

In every scenario, 686 of the 130,944 transformer-steps (0.5%) had no transformer-meter reading.

**What this shows:**
- **The outcome barely changes.** With a meter on every transformer the overloads removed are within ±1% of perfect measurement.
- **The cost is switching.** A 1% meter error moves loads that sit near the 85% threshold back and forth across it. The persistence forecast copies that noise into the next day, so the controller reroutes and restores more often: 74% more switch operations in the normal year, 9% more in C4.
- **Missed and false overloads balance.** "Overloads missed" are transformer-steps above the threshold that the operator saw at or below it. In C4, 97% of the missed ones were within 1.5 points of the threshold (median 0.5), and as many false ones went the other way. Where the smart meters stood in for a transformer meter, the error was larger: 2.3 points on average, against 0.34 for the transformer meters.

A deadband on rerouting and restoring (act at 85%, restore only below, say, 80%) would remove much of this switching wear; it was added in Phase 10 (see Hardening).

**With the short-horizon forecast of Phase 9** (`docs/data_quality_city_demo.csv`), the meters cost a little more in outcome and less in switching. Transformer-steps above 85% with control, perfect → meters:
- C1: 33 → 34
- C2: 261 → 263
- C3: 178 → 186
- C4: 619 → 628

Switch operations, perfect → meters:
- C1: 140 → 140
- C2: 304 → 360
- C3: 516 → 552
- C4: 708 → 800

A forecast that starts from the last reading reacts to the meter error at once instead of a day later.

**Limits of the model:**
- Every transformer has a meter, so the smart meters only fill gaps. A network metered by smart meters alone would depend much more on the head-end's estimates.
- Delays are independent from meter to meter, apart from concentrator outages.
- Consumer readings are the consumer's share of the transformer's load rather than separately simulated appliances.
- Clock errors, tampering and theft are not modelled.


---

## 🎯 Short-Horizon Forecast (Phase 9)

The switching decision taken at the end of a step is for the next step only. The persistence forecast answers that question with yesterday's load, made 24 hours earlier, so it cannot see a heatwave coming and expects yesterday's peaks at yesterday's times. Phase 9 forecasts the next 15 minutes instead, at the end of each step (`ml/short_horizon.py`), from:
- **what the operator has recorded** (the transformer meters of Phase 8, up to now): the last readings, the last 4 hours, the same time yesterday and a week ago;
- **the weather forecast** for the target time (the true temperature plus a forecast error of 0.5 °C, `city.yaml forecast.temperature_error_c`);
- **the calendar** (time of day, weekday / Saturday / Sunday or holiday);
- **the transformer's consumer mix** (shares of residential, commercial, industrial and mixed-use).

One XGBoost model serves all 44 transformers. It works on loads divided by each transformer's mean and predicts the change from the last reading. It never sees a true load, only the recorded ones.
- **Training data** (`python ml/train_short.py --city demo`): 8 January to 31 March, recorded histories of C1 and C3. C2 and C4 differ from them only from May, so they add nothing before it.
- **Calibration:** April stops the training and sets the P10–P90 band, per 4-hour block of the day.
- **Test:** May, with its heatwave, never seen in training.

The model is committed (`cities/demo/forecast/`); the service uses it when `forecast_source: short` (the city's default now). Forecasts are stored with a 0.25 h horizon as `short-15m-v1`, and the Forecasting page scores them as time passes.

### Accuracy in May (`docs/forecast_short_city_demo.csv`)
Scored against the true loads:

| Scenario | nRMSE: model / last value / persistence 24 h | Heatwave days | 10% highest loads | Model P10–P90 coverage |
|:---|:---:|:---:|:---:|:---:|
| C1 Normal year | **3.7%** / 5.3% / 21.4% | 3.7% / 5.3% / 23.2% | 3.1% / 3.2% / 20.7% | 78.5% |
| C2 May heatwave | **4.1%** / 5.6% / 22.4% | 4.7% / 6.0% / 25.2% | 3.2% / 3.3% / 22.6% | 72.7% |
| C3 EV adoption | **3.6%** / 5.5% / 20.9% | 3.6% / 5.5% / 22.7% | 3.1% / 3.5% / 18.9% | 78.1% |
| C4 Heatwave and EV adoption | **4.0%** / 5.7% / 22.0% | 4.6% / 6.1% / 24.6% | 3.2% / 3.5% / 21.1% | 72.5% |

"Last value" is the simplest short-horizon forecast: the reading just taken. Most of the gain over persistence comes from forecasting 15 minutes ahead instead of 24 hours. The model takes another 25–35% off the error of the last value by anticipating the ramps: the time of day carries the most weight (`docs/feature_importance_short_city_demo.csv`). The band, calibrated on April, covers 78% of true loads in normal weather but only 72–73% in the heatwave, when errors are larger than anything in its calibration month.

### Closed Loop with the Short-Horizon Forecast
May, automatic mode, through the meters, without the deadband of Phase 10 (`docs/deadband_city_demo.csv`, rows with `restore_margin_pct` 0; persistence: `docs/closed_loop_city_demo_persistence.csv`):

| Scenario | Transformer-steps > 85%, no control | With control: persistence → short | Reduction: persistence → short | Steps > 100%: persistence → short | Switch operations: persistence → short |
|:---|:---:|:---:|:---:|:---:|:---:|
| C1 Normal year | 38 | 36 → 34 | 5.3% → 10.5% | 0 → 0 | 160 → 140 |
| C2 May heatwave | 317 | 320 → **263** | −0.9% → **17.0%** | 1 → 1 | 348 → 360 |
| C3 EV adoption | 485 | 323 → **186** | 33.4% → **61.6%** | 1 → 0 | 552 → 552 |
| C4 Heatwave and EV adoption | 1,031 | 838 → **628** | 18.7% → **39.1%** | 11 → **1** | 800 → 800 |

With the same number of switch operations, the control now removes twice as many overloads under EV charging and in the combined scenario. The heatwave is no longer missed (17% removed instead of none), and the steps with a transformer above its rating fall from 11 to 1 in C4. What remains is mostly `no_safe_option` (130 steps in C4): transformers whose neighbours are full too, which no switching can relieve. The decision still uses the point forecast; the P90 bound only raises the risk class (the design decision of Phase 4).

**Limits:**
- The model learned from the same simulator it is tested on, so the accuracy is that of a model fitted to this city's demand model. On a real network the gap to the last-value forecast would likely be smaller.
- The weather forecast is the true temperature plus random noise; real forecast errors are larger and correlated.
- Forecasting 15 minutes ahead is enough for switching decided one step ahead. Planning further ahead (a day ahead) still needs the 24 h forecast.


---

## 🛡 Hardening (Phase 10)

### Switching Deadband (`operations.yaml rerouting.restore_margin_pct`)
Phase 8 showed meter noise near the threshold making the controller reroute and restore back and forth. The deadband rule: while the configuration in force keeps every transformer under the threshold and respects every limit, the controller **holds it** unless the normal configuration would be at least `restore_margin_pct` points under the threshold. Otherwise it returns to normal or reroutes as before. A hold is recorded as "Holding the current configuration … returns to normal below 80%". The classic benchmark keeps 0, so every published classic number is unchanged; the city uses 5.

May, automatic mode, short-horizon forecast, through the meters (`docs/deadband_city_demo.csv`):

| Scenario | Switch operations: no deadband → 5 points | Reroutes | Transformer-steps > 85% with control | Reduction from no control |
|:---|:---:|:---:|:---:|:---:|
| C1 Normal year | 140 → **108** | 20 → 15 | 34 → 31 | 10.5% → 18.4% |
| C2 May heatwave | 360 → **320** | 32 → 21 | 263 → 260 | 17.0% → 18.0% |
| C3 EV adoption | 552 → **484** | 124 → 64 | 186 → **160** | 61.6% → **67.0%** |
| C4 Heatwave and EV adoption | 800 → **748** | 129 → 64 | 628 → **592** | 39.1% → **42.6%** |

The deadband removes 7–23% of the switch operations and halves the reroutes, and it also lowers the overloads. A reroute kept in force through the evening no longer drops out between two peaks, so the next ramp starts in a configuration that already has room. The cost is time away from the normal configuration: in C4 the network spent 395 instead of 328 steps rerouted. `docs/closed_loop_city_demo.csv` is now these results.

### A Failed Step Pauses, It Does Not Stop the Service
If a step raises an error (a power flow that cannot be solved, a database error), the service:
- logs it with its traceback,
- records a critical event in the audit log,
- pauses the clock before that step,
- shows the reason in red on every page (status `step_failed`).

It keeps serving commands. Resuming tries the step again; jumping or starting a new run moves past it. Before, such an error ended the service.

### Smaller Fixes
- **Continuous integration** (`.github/workflows/tests.yml`): on every push GitHub installs `requirements.txt`, checks the installation and runs the fast test suite. The two tests that need the eMARC parquet file are skipped there.
- **Service options:** `python sim/service.py --forecast` now accepts the city's `persistence` and `short` sources.
- **One timeline per run (Phase 10.1):** runs on one database can cover the same simulated times with different scenarios or forecasts. A new run, a jump or a scenario change now deletes what earlier runs wrote from its start on: the city's measurements and meter readings, and the run-time forecasts. A new run also uses the configured forecast source instead of the one saved by an older run. The Forecasting page opens on the model the service is using and scores only the current run's run-time forecasts.
- **Dashboard wording:** labels follow the step length and forecast horizon of the configuration in force ("next step" and "15-minute-ahead" for a city), and history windows cover the stated number of hours.
- **Report:** `docs/REPORT.md` gathers every result of the project in one place as a draft results chapter.

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
| `simulation.yaml` | Simulation clock (range, step, speed, accelerated factor, catch-up), orchestrator (scenario, first mode, forecast source, pause on recommendation), service settings (polling, heartbeat) and the season calendar |

A synthetic city has its own set of these files in `cities/<name>/config/`, written by `city/generate.py`; `--city <name>` (service, control, dashboard, closed-loop and benchmark scripts) or the environment variable `LFDR_CONFIG_DIR` selects it. `grid.yaml` may then list several `substations`, and `operations.yaml` sets `rerouting.candidate_scope: relevant`.

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
| `assets`, `meters` | Network assets and the meter registry, with their IDs (for a city: every consumer's meter, with its consumer ID and type) |
| `forecasts`, `measurements` | Forecasts per model version and horizon (pending until scored, then with their error); measured loads per source, with a quality flag (for a city with meters: `dt-scada` transformer meters, and `ami-estimate` smart-meter estimates per transformer, revised as late readings arrive) |
| `meter_readings` | Smart-meter readings as the head-end received them, with their delay and flag (a city keeps the exceptions: late readings and stuck meters) |
| `forecast_quantiles` | Interval bounds (P10, P90) per forecast hour, model version and method |
| `recommendations`, `candidates` | Every recommendation (plan, before/after, checks, losses, confidence, binding constraint) and every option evaluated for it |
| `actions` | Every switch operation executed: who/what triggered it (operator, auto, manual, rollback, reset) |
| `events`, `event_assets` | The audit log with severity, and which transformers and switches each event concerns |
| `alerts` | Risk and voltage/line alerts as they were observed |
| `controller_state` | Mode, configuration, emergency stop and undo stack (one row per controller: `default` for the replay dashboard, `service` for the simulation service) |
| `sim_state`, `sim_steps` | The simulation service's clock, run, scenario, settings and heartbeat; one row per completed step, with each transformer's measured and forecast loading, feeder results and switch states (with meters also the loading the operator saw, where each value came from and the meter statistics) |
| `commands` | Commands to the service and their results |
| `model_registry` | Forecasting model versions with features, windows, metrics and file hash |

**Upgrades:** a database created by an earlier phase is brought up to date when it is opened: missing tables and columns are added and the schema version is recorded (`storage/schema.py`, `MIGRATIONS`); nothing is dropped or rewritten, so an existing `twin.db` keeps its history.

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
    ├── city/                      # Synthetic city (Phase 7)
    │   ├── generate.py            # city.yaml -> network.dss, consumers.csv, config/ (seeded)
    │   ├── demand.py              # 15-minute demand per transformer: consumer types, calendar, seasons, weather, events
    │   ├── meters.py              # Transformer and smart-meter streams, the head-end's estimates and quality flags (Phase 8)
    │   ├── calibrate.py           # Fit the household profile on the eMARC data
    │   └── profiles/residential_emarc.csv  # The fitted household profile (committed)
    ├── cities/demo/               # The demo city
    │   ├── city.yaml              # Its specification: edit this, then generate
    │   ├── network.dss            # Generated OpenDSS model (3 substations, 8 feeders, 44 transformers)
    │   ├── consumers.csv          # Generated consumers and meters
    │   ├── forecast/              # The trained short-horizon model and its metadata (ml/train_short.py)
    │   └── config/                # Generated settings for the engine, service and dashboard
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
    │   ├── app.py                 # Dashboard entry point: page navigation
    │   ├── common.py              # Shared by the pages: database, service state, commands, start the service
    │   ├── plots.py               # Plotly network topology, load curves, before/after, time series
    │   └── views/                 # One file per page
    │       ├── overview.py        # A: live network overview
    │       ├── network.py         # B: single-line diagram with transformer/feeder inspector
    │       ├── transformers.py    # C: sortable transformer table with history
    │       ├── forecasting.py     # D: forecast accuracy over time
    │       ├── decisions.py       # E: alerts, waiting proposal, recommendations with candidates
    │       ├── controls.py        # F: service, clock, scenario, mode, policy, threshold, emergency stop
    │       ├── audit.py           # G: filterable audit log
    │       ├── data_quality.py    # H: meter streams, late and suspect readings, operator view against the truth
    │       └── replay.py          # What-if replay sandbox (the earlier single-page dashboard)
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
    │   ├── short_horizon.py       # A city's 15-minute-ahead forecast: features, model loading, forecast tables (Phase 9)
    │   ├── train_short.py         # Trains it on the city's recorded history; accuracy table
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
    │   ├── closed_loop_city_demo.csv, performance_city_demo.csv  # The same for the demo city (May; one busy week)
    │   ├── data_quality_city_demo.csv  # The city with perfect measurement and with its meters (May)
    │   ├── *_persistence.csv      # The same city tables with the 24 h persistence forecast (Phase 8)
    │   ├── deadband_city_demo.csv # The city with and without the switching deadband (Phase 10)
    │   ├── REPORT.md              # Draft results chapter for the project report: every result in one place
    │   ├── forecast_short_city_demo.csv, feature_importance_short_city_demo.csv  # Short-horizon forecast accuracy (May)
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
    │   ├── test_simulation.py     # Clock, orchestrator, service commands, kill-and-restart recovery, benchmark
    │   ├── test_pages.py          # Every dashboard page headless; service settings; schema migration
    │   ├── test_city.py           # Generator, multi-substation engine, demand model, calibration, persistence forecast, city in the loop
    │   ├── test_meters.py         # Meter streams, head-end estimates and flags, revisions, operator view against the truth
    │   ├── test_short_forecast.py # Short-horizon forecast: no leakage, published accuracy, closed loop
    │   ├── test_hardening.py      # Switching deadband, a failed step pauses the service, settings checks
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
```bash
cd project
streamlit run app/app.py
```
Open your browser at `http://localhost:8501`. The live pages need the simulation service: start it from any page (**Start the service**) or as in step 5.

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
The dashboard's **Start the service** button is enough. To run it in a terminal instead (in `project/`):
```bash
python sim/service.py                     # Ctrl+C stops it; starting it again resumes
```
Useful options: `--speed accelerated` (one simulated year per 24 h), `--speed max`, `--mode auto`, `--scenario "S3 EV surge (T4 +200 kW, 18-22 h)"`, `--forecast live` (needs `data/processed/clusters.parquet`), `--paused`, `--new-run`. While it runs, it can also be steered from any terminal:
```bash
python sim/control.py status
python sim/control.py pause                # resume | step | restart | stop
python sim/control.py speed 86400          # one simulated day per second
python sim/control.py jump "2020-02-29 18:00"
python sim/control.py mode auto            # monitor | recommend | auto
python sim/control.py approve REC-0001     # reject REC-0001
python sim/control.py threshold 80         # use-mv on | off; policy pause | run
python sim/control.py steps 10             # the last 10 steps
```
In recommend mode the clock pauses itself when a switching proposal is waiting; approving or rejecting it (on the Alerts and decisions page or with `control.py`) lets the clock run on. With the approval policy set to *run* (`policy run`), the clock does not wait, and a proposal whose hour begins without a decision expires (it is marked superseded, with the reason in the audit log).

The report tables are produced by:
```bash
python sim/closed_loop.py                  # docs/closed_loop.csv (about 3 min)
python sim/benchmark.py                    # docs/performance.csv (about 1 min, numbers depend on the computer)
```

### 6. Running the Tests
```bash
cd project
python -m pytest                  # full suite, about 7 min (benchmark, closed loop, kill-and-restart, v2 training, cold start)
python -m pytest -m "not slow"    # everything except those, about 2 min
```
The what-if replay page's tests (`tests/test_dashboard.py`) are run in a separate process by `tests/test_dashboard_process.py`, because OpenDSS must be loaded on the page's own worker thread; they show as deselected in the main run. `python -m pytest` makes sure the tests run with the Python of the active environment (a bare `pytest` can pick up another environment's copy). The golden tests re-run the scenario benchmark and the forecast evaluation and compare them with `docs/results.csv`, `docs/results_worst_hour.csv` and `docs/metrics.csv` (timing columns excluded); if a change alters any published number, they fail. The dashboard tests start the pages headless: the sandbox through the S3 and S4 worst hours, and every live page against a database filled by a real service run (approving a proposal, changing the threshold, filtering the log).

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