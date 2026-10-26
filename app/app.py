"""Jagawana AI - demo app (Streamlit).

A prototype that shows, for a chosen date, which monitored 0.5-degree cells in
Sumatra and Kalimantan have the highest 7-day fire danger, and WHY.

It runs entirely on stored data (2023-2025) and the models trained in this
repository. It is NOT a live system.

Run from the project folder:
    python -m streamlit run app\\app.py
"""
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # draw figures to memory, no pop-up windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
import xgboost as xgb

# ----------------------------------------------------------------- file paths
ROOT = Path(__file__).resolve().parents[1]
DATASET_FILE = ROOT / "data" / "processed" / "dataset.parquet"
PRED_FILE = ROOT / "data" / "processed" / "predictions_valid_test.parquet"
MODEL_FILE = ROOT / "models" / "xgboost_model.joblib"
INFO_FILE = ROOT / "models" / "model_info.json"
COND_MODEL_FILE = ROOT / "models" / "xgboost_conditions_only.joblib"
COND_INFO_FILE = ROOT / "models" / "model_info_conditions_only.json"
CLIM_FILE = ROOT / "models" / "climatology.csv"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"

# ------------------------------------------------------------------- settings
TIERS = ["Very high", "High", "Elevated", "Lower"]
TIER_RULE = {
    "Very high": "top 5% of cell-days",
    "High": "next 5% (top 10% together)",
    "Elevated": "next 10% (top 20% together)",
    "Lower": "remaining 80%",
}
TIER_RGB = {
    "Very high": (165, 15, 21),
    "High": (222, 45, 38),
    "Elevated": (252, 146, 114),
    "Lower": (170, 185, 205),
}
DEFAULT_DATE = "2024-09-15"
MAX_DISTANCE_KM = 50.0

PLACES = {
    "Pekanbaru area, Riau": (0.51, 101.45),
    "Jambi city area, Jambi": (-1.61, 103.61),
    "Palembang area, South Sumatra": (-2.99, 104.76),
    "Pontianak area, West Kalimantan": (-0.03, 109.33),
    "Ketapang area, West Kalimantan": (-1.83, 109.97),
    "Palangka Raya area, Central Kalimantan": (-2.21, 113.92),
    "Banjarmasin area, South Kalimantan": (-3.32, 114.59),
    "Samarinda area, East Kalimantan": (-0.50, 117.15),
    "Custom coordinates": None,
}

GUIDANCE_NOTE = ("General guidance only. It is not official advice and must be validated with "
                 "local agencies (BPBD, Manggala Agni, BRGM) before any operational use.")


# ------------------------------------------------- factor groups and wording
def _days_since_text(days):
    if days >= 365:
        return "no hotspot recorded here in the past year"
    if days <= 0:
        return "a hotspot was detected here today"
    return f"the last hotspot here was {int(days)} days ago"


GROUPS = {
    "fire_here": {
        "features": ["fires_7d", "fires_30d", "fires_90d", "days_since_fire"],
        "label": "Fire history in this cell",
        "describe": lambda r: (
            f"hotspot detections in this cell over the last 7 / 30 / 90 days: "
            f"{int(r['fires_7d'])} / {int(r['fires_30d'])} / {int(r['fires_90d'])}; "
            + _days_since_text(r["days_since_fire"])),
        "advice": ("Recent burning here: verify hotspots on the ground, keep patrols and community "
                   "fire brigades on alert, and check that water sources and equipment are ready."),
    },
    "fire_near": {
        "features": ["nb_fires_7d", "nb_fires_30d"],
        "label": "Fire history in neighbouring cells",
        "describe": lambda r: (
            f"hotspot detections in the surrounding cells over the last 7 / 30 days: "
            f"{int(r['nb_fires_7d'])} / {int(r['nb_fires_30d'])}"),
        "advice": ("Fires nearby can reach this area: coordinate with neighbouring villages and "
                   "districts and keep watch on the boundaries."),
    },
    "rain": {
        "features": ["rain_7d", "rain_14d", "rain_30d", "rain_60d"],
        "label": "Recent rainfall",
        "describe": lambda r: (
            f"{r['rain_7d']:.0f} mm in the last 7 days, {r['rain_30d']:.0f} mm in 30 days, "
            f"{r['rain_60d']:.0f} mm in 60 days"),
        "advice": ("Little recent rain: reinforce burn-ban messages and enforcement, and check that "
                   "canals and water points still hold water."),
    },
    "dry_streak": {
        "features": ["dry_streak"],
        "label": "Dry spell length",
        "describe": lambda r: f"{int(r['dry_streak'])} days in a row with under 1 mm of rain",
        "advice": ("Long dry spell: raise patrol frequency, pre-position firefighting equipment, and "
                   "avoid any land clearing by burning."),
    },
    "humidity": {
        "features": ["rh_7d"],
        "label": "Humidity",
        "describe": lambda r: f"average relative humidity of {r['rh_7d']:.0f}% over the last 7 days",
        "advice": "Dry air dries out vegetation: avoid open burning and keep monitoring.",
    },
    "temperature": {
        "features": ["tmax_7d"],
        "label": "Temperature",
        "describe": lambda r: f"average daily maximum of {r['tmax_7d']:.1f} °C over the last 7 days",
        "advice": "Hot conditions dry out fuel: avoid open burning and keep monitoring.",
    },
    "wind": {
        "features": ["wind_7d"],
        "label": "Wind",
        "describe": lambda r: f"average wind speed of {r['wind_7d']:.1f} m/s over the last 7 days",
        "advice": ("Wind helps fires spread and smoke travel: avoid open burning, and prepare "
                   "information and masks for downwind villages in case fires start."),
    },
    "season": {
        "features": ["month_sin", "month_cos"],
        "label": "Time of year",
        "describe": lambda r: f"{pd.Timestamp(r['date']).strftime('%B')} (position in the fire season)",
        "advice": ("Fire-season period: step up awareness, readiness checks, and burn-ban "
                   "enforcement."),
    },
    "location": {
        "features": ["lat", "lon"],
        "label": "Location",
        "describe": lambda r: (f"{fmt_coord(r['lat'], r['lon'])} ({r['island']}); cells here have burned "
                               f"more or less often than others in 2019-2025"),
        "advice": ("Cells in this part of the region have a history of burning: review land use and, "
                   "for drained peatland, discuss canal blocking and rewetting with the local peat "
                   "restoration agency (BRGM) or BPBD."),
    },
}


# ------------------------------------------------------------ small helpers
def fmt_coord(lat, lon):
    return f"{abs(lat):.2f}°{'N' if lat >= 0 else 'S'}, {abs(lon):.2f}°E"


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km (works with numpy arrays)."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlmb = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def assign_tier(risk, cuts):
    risk = np.asarray(risk, dtype=float)
    return np.select(
        [risk >= cuts["c95"], risk >= cuts["c90"], risk >= cuts["c80"]],
        ["Very high", "High", "Elevated"], default="Lower")


def strength_word(value):
    value = abs(value)
    if value >= 0.5:
        return "strong"
    if value >= 0.2:
        return "moderate"
    return "slight"


def nearest_date(dates_index, ts):
    """Latest available date that is on or before ts (or the first date)."""
    pos = dates_index.searchsorted(ts, side="right") - 1
    return dates_index[max(pos, 0)]


def explain_rows(model, rows, features):
    """XGBoost's built-in SHAP values, summed into the readable factor groups."""
    booster = model.get_booster()
    values = booster.predict(xgb.DMatrix(rows[features]), pred_contribs=True)[:, :-1]
    values = pd.DataFrame(values, columns=features, index=rows.index)
    grouped = pd.DataFrame(index=rows.index)
    for key, group in GROUPS.items():
        present = [f for f in group["features"] if f in values.columns]
        if present:
            grouped[key] = values[present].sum(axis=1)
    return grouped


def top_factors(contrib_row, row, n_up=5, n_down=3):
    """Lists of the biggest factors raising and lowering the risk for one row."""
    ordered = contrib_row.sort_values(ascending=False)
    up, down = [], []
    for key, value in ordered.items():
        if value > 0.02 and len(up) < n_up:
            up.append((key, float(value)))
    for key, value in ordered.sort_values().items():
        if value < -0.02 and len(down) < n_down:
            down.append((key, float(value)))

    def pack(items):
        return [{"key": k, "label": GROUPS[k]["label"], "text": GROUPS[k]["describe"](row),
                 "value": v, "advice": GROUPS[k]["advice"]} for k, v in items]

    return pack(up), pack(down)


def contribution_figure(contrib_row):
    values = contrib_row.sort_values()
    labels = [GROUPS[k]["label"] for k in values.index]
    colors = ["#b2182b" if v > 0 else "#2166ac" for v in values.values]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(labels, values.values, color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Push on the risk score (red = raises risk, blue = lowers risk)")
    ax.set_title("How each factor affects this cell's risk")
    fig.tight_layout()
    return fig


def tier_badge(tier, subtitle=""):
    r, g, b = TIER_RGB[tier]
    text_color = "#222" if tier in ("Elevated", "Lower") else "white"
    return (f"<div style='background:rgb({r},{g},{b});color:{text_color};padding:14px 18px;"
            f"border-radius:8px;font-size:1.4rem'><b>{tier} risk</b>"
            f"<div style='font-size:0.9rem'>{subtitle}</div></div>")


def legend_html():
    parts = []
    for tier in TIERS:
        r, g, b = TIER_RGB[tier]
        parts.append(f"<span style='display:inline-block;width:14px;height:14px;border-radius:7px;"
                     f"background:rgb({r},{g},{b});margin-right:6px;vertical-align:middle'></span>"
                     f"<b>{tier}</b> ({TIER_RULE[tier]})&nbsp;&nbsp;&nbsp;")
    return "".join(parts)


def alert_text(row, tier, factors, advice_list):
    reasons = "; ".join(f"{f['label']}: {f['text']}" for f in factors[:3]) or "no strong single factor"
    focus = " ".join(advice_list[:2]) if advice_list else "Keep routine monitoring."
    return (f"[DEMO] Jagawana AI fire-danger notice\n"
            f"Date: {pd.Timestamp(row['date']).date()}\n"
            f"Area: cell centred near {fmt_coord(row['lat'], row['lon'])} ({row['island']})\n"
            f"Level: {tier.upper()} ({TIER_RULE[tier]})\n"
            f"Main factors: {reasons}\n"
            f"Suggested focus: {focus}\n"
            f"This is a demonstration with stored data. No message is actually sent.")


# ------------------------------------------------------------- data loading
def read_features():
    try:
        return pd.read_parquet(DATASET_FILE,
                               filters=[("date", ">=", pd.Timestamp("2023-01-01"))])
    except Exception:  # fall back to reading everything and filtering
        frame = pd.read_parquet(DATASET_FILE)
        return frame[frame["date"] >= "2023-01-01"]


def prepare_context():
    """Load data and models, score everything once. Returns a dict."""
    info = json.loads(INFO_FILE.read_text())
    model = joblib.load(MODEL_FILE)
    features = info["features"]

    preds = pd.read_parquet(PRED_FILE)
    preds["date"] = pd.to_datetime(preds["date"])
    frame = read_features()
    frame["date"] = pd.to_datetime(frame["date"])
    data = frame.merge(preds[["date", "cell_id", "risk", "split", "island"]],
                       on=["date", "cell_id"], how="inner")

    test_risk = data.loc[data["split"] == "test", "risk"]
    cuts = {"c80": float(test_risk.quantile(0.80)),
            "c90": float(test_risk.quantile(0.90)),
            "c95": float(test_risk.quantile(0.95))}
    data["tier"] = assign_tier(data["risk"].values, cuts)

    ctx = {"model": model, "features": features, "cuts": cuts, "has_cond": False,
           "cond_sorted": None}
    if COND_MODEL_FILE.exists() and COND_INFO_FILE.exists() and CLIM_FILE.exists():
        try:
            cond_info = json.loads(COND_INFO_FILE.read_text())
            cond_model = joblib.load(COND_MODEL_FILE)
            sources = cond_info["anomaly_sources"]
            clim = pd.read_csv(CLIM_FILE)
            data["month"] = data["date"].dt.month
            data = data.merge(clim, on=["cell_id", "month"], how="left")
            for col in sources:
                data[f"{col}_anom"] = data[col] - data[f"{col}_clim"]
            data["cond_score"] = cond_model.predict_proba(data[cond_info["features"]])[:, 1]
            ctx["cond_sorted"] = np.sort(data.loc[data["split"] == "test", "cond_score"].values)
            ctx["has_cond"] = True
        except Exception as error:  # the demo still works without this extra panel
            print("Conditions-only model skipped:", repr(error)[:200])
    data = data.sort_values(["date", "cell_id"]).reset_index(drop=True)
    ctx["data"] = data
    return ctx


@st.cache_resource(show_spinner="Loading data and models (first start takes a moment)...")
def load_everything():
    return prepare_context()


# ----------------------------------------------------------------- UI pieces
def render_map(day, ctx, islands):
    counts = day["tier"].value_counts()
    cols = st.columns(4)
    for col, tier in zip(cols, TIERS):
        col.metric(f"{tier}", int(counts.get(tier, 0)))
    st.markdown(legend_html(), unsafe_allow_html=True)

    show_actual = st.checkbox(
        "Demo check: ring the cells where a hotspot really appeared within the next 7 days", value=False)

    view = day[day["island"].isin(islands)].copy()
    if view.empty:
        st.info("No cells to show for this selection.")
        return
    view["color"] = view["tier"].map(lambda t: list(TIER_RGB[t]) + [210])
    view["risk_txt"] = view["risk"].map(lambda v: f"{v:.2f}")
    view["name"] = [fmt_coord(a, b) for a, b in zip(view["lat"], view["lon"])]
    view["actual"] = np.where(view["label"] == 1, "yes", "no")
    view["lat"] = view["lat"].astype(float)
    view["lon"] = view["lon"].astype(float)

    layers = [pdk.Layer(
        "ScatterplotLayer", data=view[["lon", "lat", "color", "tier", "risk_txt", "name", "actual"]],
        get_position="[lon, lat]", get_radius=26000, get_fill_color="color",
        pickable=True, opacity=0.8, stroked=False)]
    html = "<b>{tier} risk</b><br/>Score: {risk_txt}<br/>{name}"
    if show_actual:
        rings = view.loc[view["label"] == 1, ["lon", "lat"]]
        layers.append(pdk.Layer(
            "ScatterplotLayer", data=rings, get_position="[lon, lat]", get_radius=26000,
            filled=False, stroked=True, get_line_color=[0, 0, 0, 255],
            line_width_min_pixels=2, pickable=False))
        html += "<br/>Hotspot in next 7 days: {actual}"
    deck = pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(latitude=0.0, longitude=108.0, zoom=4.2, pitch=0),
        tooltip={"html": html}, map_provider="carto", map_style="light")
    st.pydeck_chart(deck)

    vh = view[view["tier"] == "Very high"]
    if show_actual and len(vh):
        st.caption(f"Demo check: of the {len(vh)} very-high cells on this date, "
                   f"{int(vh['label'].sum())} saw a hotspot within the next 7 days "
                   f"({vh['label'].mean():.0%}). A real deployment would not know this in advance.")
    st.caption("Each circle is a 0.5° cell (about 55 km). Only cells that had fire history in "
               "2019-2022 are monitored. Stored data, not live.")


def render_check(day, ctx):
    st.subheader("Check an area")
    place = st.selectbox("Choose a place", list(PLACES.keys()))
    if PLACES[place] is None:
        c1, c2 = st.columns(2)
        lat = c1.number_input("Latitude (south is negative)", -6.0, 6.0, 0.5, 0.05)
        lon = c2.number_input("Longitude (east)", 94.5, 118.5, 101.5, 0.05)
    else:
        lat, lon = PLACES[place]

    distances = haversine_km(lat, lon, day["lat"].values, day["lon"].values)
    best = int(np.argmin(distances))
    distance = float(distances[best])
    if distance > MAX_DISTANCE_KM:
        st.warning(f"No monitored cell within {MAX_DISTANCE_KM:.0f} km of this point. The demo only covers "
                   "cells that had fire history in 2019-2022, so it cannot say anything about this spot.")
        return
    row = day.iloc[best]
    tier = row["tier"]
    st.caption(f"Using the monitored cell centred at {fmt_coord(row['lat'], row['lon'])} "
               f"({distance:.0f} km from the point you chose).")

    contrib = explain_rows(ctx["model"], day.iloc[[best]], ctx["features"]).iloc[0]
    up, down = top_factors(contrib, row)

    left, right = st.columns([1, 1])
    left.markdown(tier_badge(tier, TIER_RULE[tier]), unsafe_allow_html=True)
    right.metric("Risk score (0 to 1)", f"{row['risk']:.2f}")
    if ctx["has_cond"] and not pd.isna(row.get("cond_score", np.nan)):
        pct = np.searchsorted(ctx["cond_sorted"], row["cond_score"]) / len(ctx["cond_sorted"])
        right.caption(f"Weather, season and location alone (no fire history) rate this day higher than "
                      f"{pct:.0%} of all cell-days in 2024-2025.")
    actual = "a hotspot DID appear" if row["label"] == 1 else "no hotspot appeared"
    st.caption(f"Demo check (not available in real use): in the stored data, {actual} in this cell "
               f"within the next 7 days.")

    st.markdown("#### Why this level?")
    a, b = st.columns(2)
    a.markdown("**Factors raising the risk**")
    if up:
        for f in up:
            a.markdown(f"- **{f['label']}** ({strength_word(f['value'])}): {f['text']}")
    else:
        a.write("No factor stands out.")
    b.markdown("**Factors lowering the risk**")
    if down:
        for f in down:
            b.markdown(f"- **{f['label']}** ({strength_word(f['value'])}): {f['text']}")
    else:
        b.write("No factor stands out.")
    st.caption("These factors show what the model relies on. They describe the model, "
               "not proven causes of fire.")

    figure = contribution_figure(contrib)
    st.pyplot(figure)
    plt.close(figure)

    st.markdown("#### Suggested focus")
    seen = []
    for f in up[:3]:
        if f["advice"] not in seen:
            seen.append(f["advice"])
            st.markdown(f"- {f['advice']}")
    if not seen:
        st.markdown("- Keep routine monitoring.")
    st.caption(GUIDANCE_NOTE)

    history = ctx["data"]
    window = history[(history["cell_id"] == row["cell_id"])
                     & (history["date"] <= row["date"])
                     & (history["date"] > row["date"] - pd.Timedelta(days=60))]
    if len(window) > 3:
        st.markdown("#### Risk score in the 60 days before this date")
        st.line_chart(window.set_index("date")["risk"])


def render_list(day, ctx, islands):
    st.subheader("High-risk list")
    view = day[day["island"].isin(islands)]
    if view.empty:
        st.info("No cells to show for this selection.")
        return
    n = st.slider("How many cells to list", 5, 30, 10)
    top = view.nlargest(n, "risk")
    contrib = explain_rows(ctx["model"], top, ctx["features"])

    records, details = [], []
    for rank, (idx, row) in enumerate(top.iterrows(), start=1):
        up, _ = top_factors(contrib.loc[idx], row)
        advice = []
        for f in up[:3]:
            if f["advice"] not in advice:
                advice.append(f["advice"])
        records.append({
            "Rank": rank, "Level": row["tier"], "Island": row["island"],
            "Location": fmt_coord(row["lat"], row["lon"]), "Risk score": round(float(row["risk"]), 2),
            "Main factors": "; ".join(f["label"] for f in up[:3]) or "-",
            "Suggested focus": advice[0] if advice else "Keep routine monitoring.",
            "Demo check: hotspot within 7 days": "yes" if row["label"] == 1 else "no",
        })
        details.append((row, row["tier"], up, advice))
    table = pd.DataFrame(records)
    st.dataframe(table)
    st.download_button("Download this list as CSV", table.to_csv(index=False).encode("utf-8"),
                       file_name=f"jagawana_high_risk_{pd.Timestamp(top['date'].iloc[0]).date()}.csv",
                       mime="text/csv")
    st.caption(GUIDANCE_NOTE)

    st.markdown("#### Example alert message (mock-up)")
    choice = st.selectbox("Pick a rank to preview", list(range(1, len(details) + 1)))
    row, tier, up, advice = details[choice - 1]
    st.code(alert_text(row, tier, up, advice), language=None)


def render_about(ctx):
    st.subheader("About this demo")
    st.markdown(
        "- Jagawana AI estimates, for each monitored 0.5° cell, how likely a fire hotspot is to "
        "appear within the next 7 days, and shows which factors drive that estimate.\n"
        "- It is a **risk assessment**, not a prediction of when or where a fire will start.\n"
        "- This demo uses **stored data from 2023-2025**. It is not connected to live satellite or "
        "weather feeds.\n"
        "- Data: NASA FIRMS VIIRS hotspots and NASA POWER weather. Model: XGBoost, compared with a "
        "persistence rule, Logistic Regression and Random Forest.")

    data = ctx["data"]
    test = data[data["split"] == "test"]
    st.markdown("#### What the risk levels mean (measured on 2024-2025)")
    rows = []
    for tier in TIERS:
        part = test[test["tier"] == tier]
        rows.append({"Level": tier, "Share of cell-days": TIER_RULE[tier],
                     "Cell-days": len(part),
                     "Share with a hotspot in the next 7 days": f"{part['label'].mean():.0%}"
                     if len(part) else "-"})
    st.dataframe(pd.DataFrame(rows))
    cumulative = []
    for label, tiers in [("Very high only (top 5%)", ["Very high"]),
                         ("Very high + High (top 10%)", ["Very high", "High"]),
                         ("Top 20%", ["Very high", "High", "Elevated"])]:
        part = test[test["tier"].isin(tiers)]
        cumulative.append({"If responders check": label,
                           "Share with a hotspot in the next 7 days": f"{part['label'].mean():.0%}"})
    cumulative.append({"If responders check": "All cell-days (random guess)",
                       "Share with a hotspot in the next 7 days": f"{test['label'].mean():.0%}"})
    st.dataframe(pd.DataFrame(cumulative))
    st.caption("Counts are cell-days, not separate fires: one long fire counts on many days.")

    metrics_file = RESULTS_DIR / "metrics.csv"
    if metrics_file.exists():
        st.markdown("#### Model comparison on the test years (2024-2025)")
        metrics = pd.read_csv(metrics_file)
        metrics = metrics[metrics["split"] == "test"][["model", "pr_auc", "roc_auc", "precision",
                                                       "recall", "f1"]].round(3)
        st.dataframe(metrics)
    for name, caption in [("pr_curves.png", "Precision-recall curves (test set)"),
                          ("shap_importance.png", "What drives the model (SHAP)"),
                          ("ablation_pr_auc.png", "Which information matters (ablation)")]:
        path = FIG_DIR / name
        if path.exists():
            st.image(str(path), caption=caption, width=650)

    st.markdown("#### Limitations")
    st.markdown(
        "- Estimates risk, not certainty. Some alerts are false alarms.\n"
        "- Hotspots are satellite detections, not confirmed fires. Clouds and satellite gaps can "
        "hide fires.\n"
        "- Only cells with fire history in 2019-2022 are monitored, in Sumatra and Kalimantan.\n"
        "- Recent fire history is the strongest signal. Where nothing burned recently, the "
        "estimate is much less certain.\n"
        "- Weather comes from NASA POWER (about 0.5° resolution), not ground stations. Peatland "
        "extent is not modelled directly.\n"
        "- Prevention guidance is general and not validated with agencies.")


# ------------------------------------------------------------------------ main
def main():
    st.set_page_config(page_title="Jagawana AI", page_icon="🔥", layout="wide")
    st.title("Jagawana AI: 7-day peatland fire danger (demo)")
    st.warning("Prototype using stored 2023-2025 data, not live data. It estimates risk, not certainty.")

    needed = {"data/processed/dataset.parquet": DATASET_FILE,
              "data/processed/predictions_valid_test.parquet": PRED_FILE,
              "models/xgboost_model.joblib": MODEL_FILE,
              "models/model_info.json": INFO_FILE}
    missing = [name for name, path in needed.items() if not path.exists()]
    if missing:
        st.error("Missing files: " + ", ".join(missing) +
                 ". Run `python src\\build_dataset.py` and `python src\\train_models.py` first.")
        st.stop()

    ctx = load_everything()
    data = ctx["data"]
    dates = pd.DatetimeIndex(np.sort(data["date"].unique()))

    st.sidebar.header("Choose a date")
    default = min(max(pd.Timestamp(DEFAULT_DATE), dates[0]), dates[-1])
    picked = st.sidebar.date_input("Date", value=default.date(),
                                   min_value=dates[0].date(), max_value=dates[-1].date())
    day_ts = nearest_date(dates, pd.Timestamp(picked))
    if day_ts != pd.Timestamp(picked):
        st.sidebar.info(f"No data for {picked}. Showing {day_ts.date()}.")
    islands = st.sidebar.multiselect("Islands", ["Sumatra", "Kalimantan"],
                                     default=["Sumatra", "Kalimantan"])
    st.sidebar.caption("Dates 2023 are validation years, 2024-2025 are the test years the model "
                       "never saw during training or tuning.")

    day = data[data["date"] == day_ts].reset_index(drop=True)
    tab_map, tab_check, tab_list, tab_about = st.tabs(
        ["Risk map", "Check an area", "High-risk list", "About and results"])
    with tab_map:
        render_map(day, ctx, islands or ["Sumatra", "Kalimantan"])
    with tab_check:
        render_check(day, ctx)
    with tab_list:
        render_list(day, ctx, islands or ["Sumatra", "Kalimantan"])
    with tab_about:
        render_about(ctx)


if __name__ == "__main__":
    main()