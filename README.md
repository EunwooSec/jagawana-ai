# Jagawana AI

Explainable machine learning for peatland fire risk prediction in Indonesia.

Jagawana ("forest ranger") predicts which areas of Sumatra and Kalimantan are most likely to see fires in the next 7 days, and explains which factors drive the risk, so that responders can prioritise prevention.

**Team Soaring Eagles:** Jin Woori, Rashad Syaryf
**Competition:** GENIUS Olympiad, AI Discipline (Indonesia national selection)

## Problem
TODO: 2-3 sentences on peatland fires and haze in Indonesia, with a cited source.

## What it does
- Predicts 7-day fire risk per 0.5° grid cell
- Explains the main drivers of risk (SHAP)
- TODO: smoke drift view (only if finished)

## Data
- NASA FIRMS VIIRS active fire detections (hotspots)
- Open-Meteo Historical Weather API (rainfall, temperature, humidity, wind)

## Setup
Requires Python 3.10 or newer (developed on Python 3.12).

    git clone https://github.com/EunwooSec/jagawana-ai.git
    cd jagawana-ai
    python -m pip install -r requirements.txt

## How to run
TODO: ordered commands for each script

## Results
TODO: metrics table and figures from results/figures

## Limitations
- The model estimates risk, not certainty. It does not predict exact fire locations or times.
- Hotspots are satellite detections, not confirmed fires, and clouds or satellite revisit gaps can hide fires.
- Weather comes from reanalysis data (Open-Meteo), not ground stations.
- Trained and tested only on Sumatra and Kalimantan. It should not be used for other regions or other fire types without retraining.

## License
MIT
