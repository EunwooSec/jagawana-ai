# Jagawana AI

Explainable machine learning for peatland fire risk prediction and resource allocation in Indonesia.

Jagawana ("forest ranger") predicts which areas of Sumatra and Kalimantan are most likely to see fire hotspots in the next 7 days. Going beyond traditional maps, it explains the *causes* of the risk and acts as a **Decision-Support Engine** to recommend exactly where to allocate limited firefighting resources.

**Team Soaring Eagles:** Jin Woori, Rashad Syaryf

**Competition:** GENIUS Olympiad, AI Discipline (Indonesia national selection)

## Problem
Indonesia suffers from severe annual peatland and forest fires (Karhutla), causing transboundary haze, massive carbon emissions, and respiratory illnesses. While satellite hotspot maps exist, they are reactive. They show where fires *are*, but they do not predict future risk, explain the underlying causes (e.g., peat dryness vs. human activity), or help government agencies decide where to deploy their limited firefighting teams.

## What it does (Novelty & Innovation)
- **Predicts Risk:** Forecasts, for each 0.5° grid cell on each day, whether at least one fire hotspot will appear in the next 7 days.
- **Model Comparison:** Compares four approaches: a simple persistence rule, Logistic Regression, Random Forest, and XGBoost (main model).
- **Explains Causes (SHAP):** Uses TreeSHAP to explain *why* a cell is at risk, grouping factors into human-readable causes (e.g., "Peatland coverage & dryness", "Dry spell length", "High wind", "Recent nearby fires").
- **Resource Allocation Command Center:** Translates the cause weights into a priority score. The user inputs how many personnel teams are available, and a greedy optimization algorithm outputs a deployment map and action plan (e.g., "Deploy 1 Rapid Suppression Team + 1 water pump").
- **Live Data Integration:** Connects to the live NASA FIRMS API to overlay real-time 24-hour hotspots alongside the 7-day predictive risk map.
- **Spatial Validation:** Tests whether the model transfers between islands (trained on Sumatra, tested on Kalimantan, and the reverse).

## Data
- **NASA FIRMS** VIIRS S-NPP 375 m active fire detections, 2019-2025, Indonesia (hotspots). Kept: vegetation fires (type 0) with nominal or high confidence.
- **NASA POWER** daily meteorology (rainfall, maximum temperature, relative humidity, wind speed).
- **Indonesia Peatland Extent (GFW / BRGM)** *kabupaten*-level peatland area spatial boundaries (`spatial-metrics-indonesia-peat_area_kabupaten.geojson`) intersected onto each 0.5° grid cell.

The raw data files are not stored in this repository because of their size. The scripts below rebuild everything from the public sources.

## Method in brief
- Study area: Sumatra and Kalimantan, 0.5° grid cells. 425 cells with at least 5 hotspots in 2019-2022 are used (cells are selected from training years only).
- Label: 1 if the cell has at least one hotspot during the next 7 days.
- Features (all computed only from information up to the prediction day): hotspot counts over the last 7, 30 and 90 days, days since the last hotspot, hotspot counts in the 8 neighbouring cells, rainfall over 7/14/30/60 days, length of the current dry streak, temperature, humidity, wind, season, cell location, peatland fraction, and peat-dryness interaction (`peat_fraction * dry_streak`).
- Dataset: 1,045,500 rows (2019-04-01 to 2025-12-24).
- Split by date, with a 7-day gap before each boundary: train 2019-2022, validation 2023, test 2024-2025. Decision thresholds are tuned on validation only.

## Cost Analysis & Scalability
Our solution is designed to be highly cost-effective and scalable for deployment in developing nations:
- **Zero Data Cost:** Relies entirely on free, open-source satellite and spatial data from NASA (FIRMS and POWER) and Global Forest Watch.
- **Low Compute Requirements:** The 0.5° grid resolution and XGBoost architecture are lightweight. The model trains in minutes on a standard laptop CPU, avoiding the need for expensive GPU clusters or supercomputers.
- **Deployment:** The frontend uses Streamlit, which can be hosted on a cloud server for under $10/month (or hosted on-premise by local government agencies like BNPB for free).

## Setup
Requires Python 3.10 or newer (developed on Python 3.12).

    git clone https://github.com/EunwooSec/jagawana-ai.git
    cd jagawana-ai
    python -m pip install -r requirements.txt

## How to run
1. Download the FIRMS file and peatland GeoJSON: on the NASA FIRMS Archive Download page, request VIIRS S-NPP, country Indonesia, CSV, 2019-01-01 to 2025-12-31, and save the extracted CSV as `data/raw/fire_archive_SV-C2_830533.csv`. Place the Indonesia *kabupaten* peatland GeoJSON in `data/raw/spatial-metrics-indonesia-peat_area_kabupaten.geojson`.
2. Build the dataset (downloads and caches weather from NASA POWER, computes spatial peatland fractions, then builds features and labels):

    python src\build_dataset.py

3. Train and evaluate the models (writes tables to `results/`, figures to `results/figures/`, the model to `models/`):

    python src\train_models.py

4. Run the Decision-Support App (Requires a free NASA API key):

    streamlit run app\app.py

## Results
Test set (2024-2025, years never used for training or tuning). Share of positive rows: 0.281, so a random guess scores a PR-AUC of about 0.28.

| Model | PR-AUC | ROC-AUC | F1 |
|---|---|---|---|
| Persistence baseline (fire in the last 7 days) | 0.523 | 0.712 | 0.561 |
| Logistic Regression | 0.627 | 0.804 | 0.601 |
| Random Forest | 0.674 | 0.826 | 0.624 |
| **XGBoost** | **0.683** | **0.832** | **0.629** |

XGBoost and Random Forest perform similarly.

**Operational view (XGBoost, test set).** If only the 5% highest-risk cell-days are checked, 88.6% of them see a hotspot within 7 days (3.16 times better than random). At 10%: 81.1%. At 20%: 70.0%.

**Transfer between islands (XGBoost without latitude and longitude, PR-AUC).** Trained on Sumatra and tested on Kalimantan: 0.671 (Kalimantan-trained: 0.694). Trained on Kalimantan and tested on Sumatra: 0.642 (Sumatra-trained: 0.663).

**What drives the predictions (SHAP) & Ablation.** Recent fire history matters most (hotspots in the last 30 and 90 days, days since the last hotspot), followed by season, hotspots in neighbouring cells, dry streak length, and peatland coverage (`peat_fraction`). In feature-group ablation without fire history (predicting new ignitions in quiet cells with no fire in the last 90 days), `peat_fraction` becomes the **#2 strongest predictor overall** (`0.268` mean |SHAP|), achieving a **3.62x lift** over random. We extract these SHAP values in real-time to generate our cause attributions and action plans.

All figures and tables are in `results/` (for example `results/metrics.csv`, `results/figures/pr_curves.png`).

## Limitations
- The model estimates risk, not certainty. It does not predict exact fire locations or times, and about 4 in 10 alerts at the chosen threshold are false alarms.
- Hotspots are satellite detections, not confirmed fires. Clouds and satellite revisit gaps can hide fires. "Fire cell-days" count each day a cell has a hotspot, so one long fire counts many times.
- Evaluated on fire-prone cells only (at least 5 hotspots in 2019-2022).
- Recent fire history is the strongest signal in active fire zones, whereas environmental and peatland features drive early detection in unburned ("quiet") cells.
- Weather comes from NASA POWER (MERRA-2 based, about 0.5° resolution), not from ground stations. Peatland coverage is aggregated at the *kabupaten* (regency) level rather than subsurface peat depth or water-table telemetry.
- 2023 (validation) was a severe fire year, so validation scores are higher than test scores.
- Trained and tested only on Sumatra and Kalimantan. Do not use it for other regions or fire types without retraining.

## License
MIT