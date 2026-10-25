# Jagawana AI

Explainable machine learning for peatland fire risk prediction in Indonesia.

Jagawana ("forest ranger") predicts which areas of Sumatra and Kalimantan are most likely to see fire hotspots in the next 7 days, and shows which factors drive the risk, so that responders can prioritise where to check first.

**Team Soaring Eagles:** Jin Woori, Rashad Syaryf

**Competition:** GENIUS Olympiad, AI Discipline (Indonesia national selection)

## Problem
TODO (Rashad): 2-3 sentences on peatland fires and haze in Indonesia, with a cited source.

## What it does
- Predicts, for each 0.5° grid cell on each day, whether at least one fire hotspot will appear in the next 7 days
- Compares four approaches: a simple persistence rule, Logistic Regression, Random Forest, and XGBoost (main model)
- Explains the main drivers of the predictions with SHAP
- Tests whether the model transfers between islands (trained on Sumatra, tested on Kalimantan, and the reverse)

## Data
- **NASA FIRMS** VIIRS S-NPP 375 m active fire detections, 2019-2025, Indonesia (hotspots). Kept: vegetation fires (type 0) with nominal or high confidence.
- **NASA POWER** daily meteorology (rainfall, maximum temperature, relative humidity, wind speed).

The raw data files are not stored in this repository because of their size. The scripts below rebuild everything from the public sources.

## Method in brief
- Study area: Sumatra and Kalimantan, 0.5° grid cells. 425 cells with at least 5 hotspots in 2019-2022 are used (cells are selected from training years only).
- Label: 1 if the cell has at least one hotspot during the next 7 days.
- Features (all computed only from information up to the prediction day): hotspot counts over the last 7, 30 and 90 days, days since the last hotspot, hotspot counts in the 8 neighbouring cells, rainfall over 7/14/30/60 days, length of the current dry streak, temperature, humidity, wind, season, and cell location.
- Dataset: 1,045,500 rows (2019-04-01 to 2025-12-24).
- Split by date, with a 7-day gap before each boundary: train 2019-2022, validation 2023, test 2024-2025. Decision thresholds are tuned on validation only.

## Setup
Requires Python 3.10 or newer (developed on Python 3.12).

```
git clone https://github.com/EunwooSec/jagawana-ai.git
cd jagawana-ai
python -m pip install -r requirements.txt
```

## How to run
1. Download the FIRMS file: on the NASA FIRMS Archive Download page, request VIIRS S-NPP, country Indonesia, CSV, 2019-01-01 to 2025-12-31. Save the extracted CSV as `data/raw/fire_archive_SV-C2_830533.csv` (or change `FIRMS_FILE` in `src/build_dataset.py` to your file name).
2. Build the dataset (downloads and caches weather from NASA POWER, then builds features and labels):
   ```
   python src\build_dataset.py
   ```
3. Train and evaluate the models (writes tables to `results/`, figures to `results/figures/`, the model to `models/`):
   ```
   python src\train_models.py
   ```

## Results
Test set (2024-2025, years never used for training or tuning). Share of positive rows: 0.281, so a random guess scores a PR-AUC of about 0.28.

| Model | PR-AUC | ROC-AUC | F1 |
|---|---|---|---|
| Persistence baseline (fire in the last 7 days) | 0.523 | 0.712 | 0.561 |
| Logistic Regression | 0.627 | 0.804 | 0.602 |
| Random Forest | 0.673 | 0.825 | 0.624 |
| **XGBoost** | **0.681** | **0.830** | **0.627** |

XGBoost and Random Forest perform similarly.

**Operational view (XGBoost, test set).** If only the 5% highest-risk cell-days are checked, 88% of them see a hotspot within 7 days (3.15 times better than random). At 10%: 81%. At 20%: 70%.

**Transfer between islands (XGBoost without latitude and longitude, PR-AUC).** Trained on Sumatra and tested on Kalimantan: 0.671 (Kalimantan-trained: 0.687). Trained on Kalimantan and tested on Sumatra: 0.648 (Sumatra-trained: 0.660).

**What drives the predictions (SHAP).** Recent fire history matters most (hotspots in the last 30 and 90 days, days since the last hotspot), followed by the length of the dry streak, hotspots in neighbouring cells, and season. SHAP describes what the model relies on, not what causes fires.

All figures and tables are in `results/` (for example `results/metrics.csv`, `results/figures/pr_curves.png`).

## Limitations
- The model estimates risk, not certainty. It does not predict exact fire locations or times, and about 4 in 10 alerts at the chosen threshold are false alarms.
- Hotspots are satellite detections, not confirmed fires. Clouds and satellite revisit gaps can hide fires. "Fire cell-days" count each day a cell has a hotspot, so one long fire counts many times.
- Evaluated on fire-prone cells only (at least 5 hotspots in 2019-2022).
- Recent fire history is the strongest signal, so performance on cells with no recent fire is likely lower and has not yet been measured separately.
- Weather comes from NASA POWER (MERRA-2 based, about 0.5° resolution), not from ground stations. Peatland extent is not modelled directly.
- 2023 (validation) was a severe fire year, so validation scores are higher than test scores.
- Trained and tested only on Sumatra and Kalimantan. Do not use it for other regions or fire types without retraining.

## License
MIT
