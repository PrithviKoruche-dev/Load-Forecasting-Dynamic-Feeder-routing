# Results and Discussion (draft chapter)

This chapter gathers every result of the project in one place, with the file it comes from, so each number can be checked or regenerated. All numbers come from the committed result tables in `project/docs/`; the command that regenerates each one is given in the README.

## 1. What was built

The system has two parts that share one engine:

- **The classic network.** Six 500 kVA distribution transformers on two 11 kV feeders, with load from real eMARC smart-meter data. A 24 h-ahead XGBoost forecast and a physics-checked rerouting engine run on it. All validated benchmark numbers come from this network.
- **A synthetic city.** Three substations, 8 feeders, 44 transformers and 5,083 metered consumers, generated from one YAML file and a seed. It runs through the same engine, controller, simulation service and dashboard. Here the operator sees loads through realistic meters, and a 15-minute-ahead forecast trained on the city's own recorded history decides the switching.

Every step of the closed loop does the same things in order:
1. Measure the loads.
2. Solve the power flow (OpenDSS).
3. Forecast the next step.
4. Detect risks.
5. Validate candidate switching actions (radial, every load supplied, thermal and voltage limits).
6. Apply or propose an action.
7. Record everything in the database.

The outcome is always judged on the true loads the network carried.

## 2. Forecasting on real data (classic network)

Source: `docs/metrics.csv`, `docs/metrics_peak.csv`, `docs/metrics_intervals.csv`, `docs/metrics_overload.csv`.

| Forecast (1,073 test hours, 24 h ahead) | nRMSE, per transformer pooled | nRMSE, sum of the six |
|:---|:---:|:---:|
| XGBoost (one model for all transformers) | **25.3%** | **10.8%** |
| Same hour yesterday | 32.2% | 14.1% |
| Same hour last week | 33.4% | – |

- At the peaks the forecast under-predicts: the bias is −0.91 kW of a 0.96 kW MAE.
- The P10–P90 band from rolling conformal calibration covers 78.3% of actual loads (target 80%).
- The point forecast flags only 12–19% of the real overloads in the heatwave and EV scenarios. The P90 bound flags 80–97% of them, at lower precision. The dashboard therefore raises forecast risk on P90, while switching is decided on the point forecast.

## 3. Rerouting engine (classic network, hour by hour)

Source: `docs/results.csv`, `docs/results_mitigation.csv`.

| Scenario | Overload hours | Resolved by switching | Mean peak before → after |
|:---|:---:|:---:|:---:|
| S2 Heatwave | 5 | 100% | 87.4% → 74.5% |
| S3 EV surge | 23 | 100% | 87.3% → 69.5% |
| S4 Severe stress | 431 | 35.0% | 105.5% → 79.7% |
| S5 Edge case | 882 | 44.0% | 118.0% → 75.4% |

- Decisions take 5–9 ms.
- Where no configuration fixes everything, a labelled mitigation is offered if it respects every hard limit. It exists in 128 of 280 such hours in S4 and in 168 of 494 in S5.

## 4. Closed loop on real data

Source: `docs/closed_loop.csv`.

Decisions are taken one hour ahead on the forecast, a reroute stays in force until it is restored, and the outcome is measured. Transformer-hours above 85%, no control → with control:

| Scenario | No control | With control | Reduction |
|:---|:---:|:---:|:---:|
| S2 | 41 | 37 | 9.8% |
| S3 | 59 | 48 | 18.6% |
| S4 | 1,177 | 1,150 | 2.3% |
| S5 | 803 | 554 | 31.0% |

The control can only act on overloads it sees coming. With a 24 h-ahead point forecast that misses most peaks, the gain is modest. That is the central finding that motivated the later phases.

## 5. The synthetic city: what limits the control

Source: `docs/closed_loop_city_demo_persistence.csv`, `docs/forecast_short_city_demo.csv`, `docs/deadband_city_demo.csv`, `docs/data_quality_city_demo*.csv`.

May (2,976 steps of 15 minutes), automatic mode. The table shows transformer-steps above 85%: no control → with control.

| Scenario | No control | 24 h persistence forecast | 15-min model forecast | + deadband (final) | Steps > 100%: no control → final |
|:---|:---:|:---:|:---:|:---:|:---:|
| C1 Normal year | 38 | 36 | 34 | **31** | 0 → 0 |
| C2 May heatwave | 317 | 320 | 263 | **260** | 1 → 1 |
| C3 EV adoption (15% of households) | 485 | 323 | 186 | **160** | 1 → 0 |
| C4 Heatwave and EV adoption | 1,031 | 838 | 628 | **592** | 17 → 1 |

Switch operations, persistence → model → final:

| Scenario | Persistence | Model | Final |
|:---|:---:|:---:|:---:|
| C1 | 160 | 140 | 108 |
| C2 | 348 | 360 | 320 |
| C3 | 552 | 552 | 484 |
| C4 | 800 | 800 | 748 |

Three findings:

1. **The forecast horizon matters more than the model.** The 24 h persistence forecast has an nRMSE of about 22% in May. It cannot see a heatwave coming, and with it the control removed nothing in the heatwave scenario. Forecasting the step the decision is for (15 minutes ahead) from the latest meter readings brings the nRMSE to 3.6–4.1%. The last reading alone already brings it from 20.9–22.4% to 5.3–5.7%; the trained model removes another 25–35% of the error by anticipating the ramps. In closed loop this doubles the overloads removed under EV charging (33% → 62%), catches the heatwave (−1% → 17%), and in C4 brings the steps with a transformer above its rating from 11 to 1, with the same number of switch operations.

2. **Measurement quality costs switching more than safety.** With a meter on every transformer, the overloads left with control are 1–5% more than with perfect measurement, and the smart meters cover sensor outages. The 1% meter error makes loads near the threshold cross it back and forth. With the persistence forecast that meant up to 74% more switch operations (C1); with the short forecast, up to 18% more (C2).

3. **A deadband pays twice.** Holding a safe rerouted configuration until the normal one is 5 points under the threshold:
   - removes 7–23% of the switch operations and halves the reroutes,
   - also lowers the overloads (C3: 186 → 160; C4: 628 → 592), because the network does not return to normal between two evening peaks.

   The cost is more time in a rerouted configuration: 328 → 395 steps in C4.

What remains in C4 is mostly overload that no switching can relieve: 129 decisions found no safe option, because every neighbour of an overloaded transformer was full too. That is a capacity problem, not a control problem.

## 6. Performance

Source: `docs/performance.csv`, `docs/performance_city_demo.csv`, on a 2-CPU container.

| | Classic (hourly) | City (15-minute, meters, model forecast) |
|:---|:---:|:---:|
| Step time, mean | 37 ms | 152 ms |
| A simulated year at full speed | 5.4 min | 89 min |
| Share of the accelerated-mode budget (1 year per 24 h) | 0.4% | 6.2% |
| Memory | 176 MB | about 560 MB peak, then flat |

The service survives a kill and restart with an identical result (tested). A failed step pauses the clock instead of stopping the service.

## 7. Validity and limits

- **Synthetic data:** the city's demand, meters and weather come from a model calibrated on the eMARC households but otherwise invented. Absolute numbers describe this city, not a real one. The trained forecast learned from the same simulator it is tested on, so its edge over the last reading would likely be smaller on a real network.
- **Weather forecast:** modelled as the true temperature plus 0.5 °C of random error. Real errors are larger and correlated.
- **Meters:** every transformer has a meter, so the smart meters only fill gaps. A network relying on smart meters alone would depend much more on the head-end's estimates.
- **The switching model** moves one movable block per transformer, or transfers a feeder section. It does not optimise over several steps, so it trades switch wear against overloads only through the deadband.
- **Design decision:** switching uses the point forecast, while the P90 bound only raises the risk class. In the classic network this choice costs most of the possible reduction, because the 24 h forecast under-predicts peaks. In the city the short-horizon forecast makes it much less costly.
- **Reproducibility:** every random element is seeded. XGBoost retraining is deterministic on one computer and can differ slightly on another; the committed models are used for every published number.

## 8. Further work

- **Multi-step optimisation:** a model-predictive formulation that plans switching over several steps and prices switch wear.
- **Distributed generation and storage:** rooftop solar and batteries for local peak shaving, beyond what switching can do.
- **Planning signal:** use the persistent `no_safe_option` transformers as a planning output (where capacity is short).
- **A real pilot dataset:** transformer-meter and smart-meter streams to retrain the short-horizon forecast and check the gap to the last-reading forecast.
