# Atlantic Hurricanes (HURDAT2) — Rapid Intensification (RI) Prediction (24h)

A reproducible machine-learning pipeline to predict **Rapid Intensification (RI) within the next 24 hours** using **storm track + wind observations** (HURDAT2-style features).  
This is an **imbalanced classification** problem (RI is rare ~3%), so we evaluate primarily with **PR-AUC (Average Precision)** and then tune a **decision threshold** for operational trade-offs.

---

## What this project does

- Builds a clean, end-to-end notebook workflow:
  - data loading + cleaning
  - feature engineering (lags, wind trends, location, system status, etc.)
  - model training & validation (multiple models)
  - selection using **Validation PR-AUC** (F1 as secondary)
  - test-set evaluation + interpretability (coefficients / importances)
  - **threshold tuning** + **precision–recall (PR) curves**
  - threshold comparison table (VAL → TEST) for operational choice

---

## Dataset

- **Source:** HURDAT2-style best track data (Atlantic).
- **Label:** `RI_24h` (rapid intensification within next 24 hours), derived from wind + track timing.
- **Important:** This dataset does **not** include environmental physics fields (SST, shear, OHC, humidity, satellite imagery, etc.). It’s intentionally a lightweight, track-based screening model.

---

# Modeling Details

## Label (RI_24h)

For each observation at time *t*:

- Compute wind at *t + 24h* (approximated as the next **4 observations** within the same storm).
- Define the binary target:

\[
RI\_{24h} =
\begin{cases}
1, & \text{if } wind(t+24h) - wind(t) \ge 30 \\
0, & \text{otherwise}
\end{cases}
\]

## Features (time-safe)

We only use features available at time *t* (no leakage):

- **Current:** `wind_t`, `lat`, `lon`, `status_of_system`
- **Lag / trend:** `wind_lag_1`, `wind_lag_2`, `wind_lag_4`
- **Deltas:** `dwind_6h`, `dwind_12h`, `dwind_24h_past`
- **Rolling:** `wind_rollmean_4`

## Splits (storm-wise)

Train / Validation / Test split is done **by storm_id** (storm-wise, not row-wise):

- **Train:** 70%
- **Validation:** 15%
- **Test:** 15%

This better estimates generalization to **unseen storms**.

## Why PR-AUC

Rapid intensification is rare (**~3% positives**). With heavy class imbalance, **accuracy is misleading**.

Therefore we prioritize:

- **PR-AUC (Average Precision)** as the primary metric
- **F1 / Recall** as secondary metrics
- Threshold-dependent **precision/recall trade-offs** matter operationally.

---

# Results (Key Numbers)

## Best Model (selected on Validation)

**LinearSVM** was selected primarily by **Validation PR-AUC (AP)** (with F1 as secondary).

### Validation (LinearSVM)

- **PR-AUC (AP) ≈ 0.166**

## Baseline (default decision rule, no threshold tuning) — TEST

Using the model’s default decision rule (no tuning):

- **Precision ≈ 0.08**
- **Recall ≈ 0.70**
- **F1 ≈ 0.14**
- **Confusion matrix (TEST):**  
  - **TN = 4461**, **FP = 1712**, **FN = 61**, **TP = 145**

**Interpretation:**  
Baseline behaves like a strong *screening rule* (high recall), but produces **many false alarms** (low precision), which is operationally costly.

---

# Threshold Tuning (Validation → Test)

Because RI is rare, the operating threshold controls the trade-off:

- **Lower threshold → higher recall**, but **more false alarms**
- **Higher threshold → higher precision**, but **more missed RI**

We compare **three validation-chosen thresholds** and evaluate on held-out test.

## Threshold Comparison Table (VAL → TEST)

| Choice | Threshold | TEST Precision | TEST Recall | TEST F1 | TEST FP | TEST TP |
|---|---:|---:|---:|---:|---:|---:|
| High-Recall (VAL) | ~0.036 | ~0.083 | ~0.694 | ~0.148 | 1581 | 143 |
| Best-F1 (VAL) | ~0.100 | ~0.159 | ~0.277 | ~0.202 | 302 | 57 |
| High-Precision (VAL) | ~0.095 | ~0.160 | ~0.316 | ~0.212 | 341 | 65 |

---

# Operating Threshold (Locked)

## Default recommendation: Best-F1 threshold (~0.10)

We lock the default operating threshold to **Best-F1 (~0.10)** because it provides a **balanced screening rule** for a rare-event detection problem — a practical trade-off between catching RI (recall) and controlling false alarms (precision).

## Operational alternatives

- **High-Recall (~0.036)** → *early-warning mode* (accept high alert volume / many false alarms)
- **High-Precision (~0.095)** → *false-alarm control / triage efficiency* (fewer alerts, more missed RI)

---

# Generalization Note

Validation PR-AUC is higher than Test PR-AUC:

- **VAL AP ≈ 0.166 > TEST AP ≈ 0.114**

This indicates an expected performance drop on unseen storms, but still demonstrates skill above a rare-event baseline.

---

# Limitations

- Label derived only from **track + wind timing**; no environmental physics variables (SST, shear, humidity, OHC, satellite imagery).
- `shift(-4)` assumes ~6-hour cadence (reasonable for HURDAT2, but still approximate).
- Precision is low at baseline; the model is best used as a **screening / triage tool** unless enriched with stronger predictors.

---
