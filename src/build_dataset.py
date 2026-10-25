"""build_dataset.py

Builds the Jagawana AI training dataset.

Steps:
  1. Load NASA FIRMS VIIRS (S-NPP) hotspots and keep reliable vegetation fires.
  2. Aggregate hotspots into daily counts on a 0.5-degree grid over Sumatra
     and Kalimantan.
  3. Select grid cells with fire history in the TRAINING years only
     (2019-2022), so that no information from the test years is used.
  4. Download daily weather for each selected cell from NASA POWER (cached on
     disk, so the script can be stopped and restarted).
  5. Build features (using only information up to day t) and the label
     "at least one hotspot in the cell during days t+1 ... t+7".
  6. Save the result to data/processed/dataset.parquet.

Run from the project folder:
    python src\\build_dataset.py
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ----------------------------------------------------------------- settings
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
WEATHER_DIR = RAW_DIR / "weather_power"
PROC_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
FIRMS_FILE = RAW_DIR / "fire_archive_SV-C2_830533.csv"

START = "2019-01-01"
END = "2025-12-31"
TRAIN_END = "2022-12-31"       # cells are chosen using years up to here only
WARMUP_DAYS = 90               # first days dropped (rolling windows incomplete)
HORIZON = 7                    # predict fire within the next 7 days

# Study area: Sumatra and Kalimantan (latitude/longitude bounding box)
LAT_MIN, LAT_MAX = -6.0, 6.0
LON_MIN, LON_MAX = 94.5, 118.5
CELL = 0.5                     # grid cell size in degrees
MIN_TRAIN_FIRES = 5            # a cell needs >= 5 hotspots in training years

POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
POWER_PARAMS = "PRECTOTCORR,T2M_MAX,RH2M,WS10M"


# ------------------------------------------------------ step 1 and 2: fires
def load_hotspots():
    """Load FIRMS hotspots and keep reliable vegetation fires in the study area."""
    print("Loading FIRMS file (this can take a minute)...")
    df = pd.read_csv(
        FIRMS_FILE,
        usecols=["latitude", "longitude", "acq_date", "confidence", "type"],
        dtype={"confidence": str},
    )
    print(f"  Rows in file: {len(df):,}")

    # type 0 = presumed vegetation fire (drops volcanoes, industrial sources, offshore)
    df = df[df["type"] == 0]
    # confidence: l = low, n = nominal, h = high. Drop low-confidence detections.
    df = df[df["confidence"].isin(["n", "h"])]
    print(f"  After keeping vegetation fires with nominal/high confidence: {len(df):,}")

    df = df[
        (df["latitude"] >= LAT_MIN) & (df["latitude"] < LAT_MAX)
        & (df["longitude"] >= LON_MIN) & (df["longitude"] < LON_MAX)
    ].copy()
    df["acq_date"] = pd.to_datetime(df["acq_date"])
    df = df[(df["acq_date"] >= START) & (df["acq_date"] <= END)]
    print(f"  Inside study area (Sumatra + Kalimantan box): {len(df):,}")
    return df


def build_counts(df, dates):
    """Daily hotspot counts on the grid: array of shape (days, rows, cols)."""
    n_rows = int(round((LAT_MAX - LAT_MIN) / CELL))
    n_cols = int(round((LON_MAX - LON_MIN) / CELL))
    day_idx = (df["acq_date"] - dates[0]).dt.days.values
    ii = np.floor((df["latitude"].values - LAT_MIN) / CELL).astype(int)
    jj = np.floor((df["longitude"].values - LON_MIN) / CELL).astype(int)
    counts = np.zeros((len(dates), n_rows, n_cols), dtype=np.float32)
    np.add.at(counts, (day_idx, ii, jj), 1)
    return counts


def neighbour_counts(counts):
    """Sum of daily hotspot counts in the 8 surrounding cells."""
    _, n_rows, n_cols = counts.shape
    padded = np.pad(counts, ((0, 0), (1, 1), (1, 1)))
    total = np.zeros_like(counts)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            total += padded[:, 1 + di:1 + di + n_rows, 1 + dj:1 + dj + n_cols]
    return total


# ------------------------------------------------------------ step 4: weather
def fetch_power(lat, lon):
    """Download daily weather for one point from NASA POWER (with retries)."""
    params = {
        "parameters": POWER_PARAMS,
        "community": "AG",
        "longitude": round(lon, 4),
        "latitude": round(lat, 4),
        "start": START.replace("-", ""),
        "end": END.replace("-", ""),
        "format": "JSON",
    }
    last_error = "unknown"
    for attempt in range(5):
        try:
            response = requests.get(POWER_URL, params=params, timeout=120)
            if response.status_code == 200:
                data = response.json()["properties"]["parameter"]
                weather = pd.DataFrame(data)
                weather.index = pd.to_datetime(weather.index, format="%Y%m%d")
                weather.index.name = "date"
                return weather.replace(-999, np.nan).sort_index()
            last_error = f"HTTP {response.status_code}"
        except (requests.RequestException, KeyError, ValueError) as error:
            last_error = repr(error)
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Download failed for ({lat}, {lon}): {last_error}")


def get_weather(cells):
    """Download (or load from cache) weather for every selected cell."""
    WEATHER_DIR.mkdir(parents=True, exist_ok=True)
    weather_by_cell = {}
    started = time.time()
    downloaded = 0
    for n, (i, j) in enumerate(cells, start=1):
        cache_file = WEATHER_DIR / f"cell_{i}_{j}.csv"
        if cache_file.exists():
            weather_by_cell[(i, j)] = pd.read_csv(cache_file, index_col="date", parse_dates=True)
            continue
        lat = LAT_MIN + (i + 0.5) * CELL
        lon = LON_MIN + (j + 0.5) * CELL
        weather = fetch_power(lat, lon)
        weather.to_csv(cache_file)
        weather_by_cell[(i, j)] = weather
        downloaded += 1
        per_cell = (time.time() - started) / downloaded
        remaining = per_cell * (len(cells) - n) / 60
        print(f"  Weather {n}/{len(cells)} done "
              f"({per_cell:.1f} s per cell, about {remaining:.0f} min left)")
        time.sleep(0.5)  # be polite to the free service
    return weather_by_cell


# ----------------------------------------------------------- step 5: features
def make_features(i, j, counts, nb_daily, weather, dates):
    """Features for one cell. Every feature uses only data up to day t."""
    c = pd.Series(counts[:, i, j], index=dates)
    nb = pd.Series(nb_daily[:, i, j], index=dates)
    w = weather.reindex(dates).ffill(limit=3)
    rain = w["PRECTOTCORR"]

    out = pd.DataFrame(index=dates)
    # Fire history in the cell
    out["fires_7d"] = c.rolling(7).sum()
    out["fires_30d"] = c.rolling(30).sum()
    out["fires_90d"] = c.rolling(90).sum()
    pos = np.arange(len(dates), dtype=float)
    last_fire = pd.Series(np.where(c.values > 0, pos, np.nan), index=dates).ffill()
    days_since = pd.Series(pos - last_fire.values, index=dates)
    out["days_since_fire"] = days_since.fillna(365).clip(upper=365)
    # Fire history in the 8 neighbouring cells
    out["nb_fires_7d"] = nb.rolling(7).sum()
    out["nb_fires_30d"] = nb.rolling(30).sum()
    # Rainfall and dryness
    for days in (7, 14, 30, 60):
        out[f"rain_{days}d"] = rain.rolling(days).sum()
    wet = rain >= 1.0
    out["dry_streak"] = (~wet).astype(int).groupby(wet.cumsum()).cumsum()
    # Other weather
    out["tmax_7d"] = w["T2M_MAX"].rolling(7).mean()
    out["rh_7d"] = w["RH2M"].rolling(7).mean()
    out["wind_7d"] = w["WS10M"].rolling(7).mean()
    # Season and location
    month = dates.month.values
    out["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    out["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)
    out["lat"] = LAT_MIN + (i + 0.5) * CELL
    out["lon"] = LON_MIN + (j + 0.5) * CELL
    out["cell_id"] = f"{i}_{j}"

    # Label: any hotspot in days t+1 ... t+7 (looks only forward, never a feature)
    future = c.rolling(HORIZON).sum().shift(-HORIZON)
    out["label"] = np.where(future.isna(), np.nan, (future > 0).astype(float))
    return out


# ----------------------------------------------------------------------- main
def main():
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range(START, END, freq="D")

    hotspots = load_hotspots()
    yearly = hotspots.groupby(hotspots["acq_date"].dt.year).size()
    yearly.to_csv(RESULTS_DIR / "hotspots_per_year.csv", header=["hotspots"])
    print("\nHotspots per year in the study area:")
    print(yearly.to_string())

    counts = build_counts(hotspots, dates)
    nb_daily = neighbour_counts(counts)

    n_train = int((dates <= TRAIN_END).sum())
    train_totals = counts[:n_train].sum(axis=0)
    cells = [tuple(int(v) for v in ij) for ij in np.argwhere(train_totals >= MIN_TRAIN_FIRES)]
    print(f"\nSelected {len(cells)} grid cells "
          f"(>= {MIN_TRAIN_FIRES} hotspots during 2019-2022)")
    pd.DataFrame(
        [(i, j, LAT_MIN + (i + 0.5) * CELL, LON_MIN + (j + 0.5) * CELL) for i, j in cells],
        columns=["row", "col", "lat", "lon"],
    ).to_csv(PROC_DIR / "cells.csv", index=False)

    print("\nDownloading weather from NASA POWER (cached, safe to stop and restart)...")
    weather_by_cell = get_weather(cells)

    print("\nBuilding features and labels...")
    frames = [make_features(i, j, counts, nb_daily, weather_by_cell[(i, j)], dates)
              for i, j in cells]
    data = pd.concat(frames)
    data.index.name = "date"
    data = data.reset_index()
    data = data[data["date"] >= dates[0] + pd.Timedelta(days=WARMUP_DAYS)]
    data = data.dropna().reset_index(drop=True)
    data["label"] = data["label"].astype(int)

    out_file = PROC_DIR / "dataset.parquet"
    data.to_parquet(out_file, index=False)

    print(f"\nSaved {out_file}")
    print(f"Rows: {len(data):,}   Cells: {data['cell_id'].nunique()}   "
          f"Columns: {data.shape[1]}")
    print(f"Date range: {data['date'].min().date()} to {data['date'].max().date()}")
    print(f"Share of rows with a fire in the next {HORIZON} days: {data['label'].mean():.3f}")
    print("\nShare of 'fire next 7 days' rows by year:")
    print(data.groupby(data["date"].dt.year)["label"].mean().round(3).to_string())
    print("\nFeature columns:")
    print([c for c in data.columns if c not in ("date", "cell_id", "label")])


if __name__ == "__main__":
    main()