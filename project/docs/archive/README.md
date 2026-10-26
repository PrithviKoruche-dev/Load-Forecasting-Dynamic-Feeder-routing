# Archived results

## `results_worst_hour_v1_mv_voltage_only.csv`

The worst-hour table as first published. Its two minimum-voltage columns were wrong: the engine read bus
voltages by pairing 19 bus names with the first 19 of 57 per-phase values (`zip(AllBusNames(),
AllBusMagPu())`), so only the 11 kV buses were included and the 0.415 kV customer-side buses were never
checked against the 0.95 pu limit.

Fixed in Phase 2. Re-running the full 1,073-hour benchmark with the corrected check changed **no decision**:
every overload count, rerouting outcome, resolution rate and peak loading in `docs/results.csv` is
identical, because in every accepted state the 100% transformer limit binds before any LV bus falls below
0.95 pu. Only the reported minimum voltages change:

| Scenario | Worst hour | Min V before (old → corrected) | Min V after (old → corrected) |
|:---|:---:|:---:|:---:|
| S2 Heatwave | 12 Mar 08:00 | 0.9936 → 0.9739 | 0.9936 → 0.9760 |
| S3 EV surge | 29 Feb 21:00 | 0.9958 → 0.9748 | 0.9958 → 0.9798 |
| S4 Severe stress | 21 Feb 06:00 | 0.9907 → 0.9581 | 0.9907 → 0.9581 |
| S5 Edge case | 17 Feb 07:00 | 0.9891 → **0.9405** | 0.9891 → **0.9405** |

The S5 worst hour is below the 0.95 pu limit on T3's customer side, which the old check reported as 0.9891.
