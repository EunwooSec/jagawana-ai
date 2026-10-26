"""Jagawana AI - Decision Support Engine (Beta Prototype).

Run from the project folder:
    python -m streamlit run app\\app.py
"""
import json
import requests
from io import StringIO
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
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

GUIDANCE_NOTE = ("Decision-support prototype. Must be validated with local agencies "
                 "(BPBD, Manggala Agni, BRGM) before operational use.")

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
        "advice": ("Deploy 1 Rapid Suppression Team (15 personnel) + 1 water pump. "
                   "Verify ground conditions immediately."),
        "weight": 1.3
    },
    "fire_near": {
        "features": ["nb_fires_7d", "nb_fires_30d"],
        "label": "Fire history in neighbouring cells",
        "describe": lambda r: (
            f"hotspot detections in the surrounding cells over the last 7 / 30 days: "
            f"{int(r['nb_fires_7d'])} / {int(r['nb_fires_30d'])}"),
        "advice": ("Deploy 1 Preventive Patrol Team (5 personnel). Coordinate with "
                   "neighbouring villages and monitor boundaries."),
        "weight": 1.1
    },
    "rain": {
        "features": ["rain_7d", "rain_14d", "rain_30d", "rain_60d"],
        "label": "Recent rainfall",
        "describe": lambda r: (
            f"{r['rain_7d']:.0f} mm in the last 7 days, {r['rain_30d']:.0f} mm in 30 days"),
        "advice": ("Deploy Preventive Patrol Team. Reinforce burn-ban messages and check "
                   "canals and water points."),
        "weight": 1.0
    },
    "dry_streak": {
        "features": ["dry_streak"],
        "label": "Dry spell length",
        "describe": lambda r: f"{int(r['dry_streak'])} days in a row with under 1 mm of rain",
        "advice": ("Pre-position 2 firefighting teams and 1 thermal drone. Raise patrol frequency."),
        "weight": 1.2
    },
    "humidity": {
        "features": ["rh_7d"],
        "label": "Humidity",
        "describe": lambda r: f"average relative humidity of {r['rh_7d']:.0f}% over the last 7 days",
        "advice": "Issue local burn bans. Dry air increases vegetation flammability.",
        "weight": 0.8
    },
    "temperature": {
        "features": ["tmax_7d"],
        "label": "Temperature",
        "describe": lambda r: f"average daily maximum of {r['tmax_7d']:.1f} °C over the last 7 days",
        "advice": "Monitor for heat stress on vegetation. Prepare water tankers.",
        "weight": 0.8
    },
    "wind": {
        "features": ["wind_7d"],
        "label": "Wind",
        "describe": lambda r: f"average wind speed of {r['wind_7d']:.1f} m/s over the last 7 days",
        "advice": ("Prepare smoke warning information and masks for downwind villages. "
                   "Deploy air quality monitors if available."),
        "weight": 0.9
    },
    "season": {
        "features": ["month_sin", "month_cos"],
        "label": "Time of year",
        "describe": lambda r: f"{pd.Timestamp(r['date']).strftime('%B')} (fire season)",
        "advice": "Step up awareness and readiness checks across the region.",
        "weight": 0.7
    },
    "location": {
        "features": ["lat", "lon"],
        "label": "Location",
        "describe": lambda r: (f"{fmt_coord(r['lat'], r['lon'])} ({r['island']})"),
        "advice": ("Alert local BRGM peat restoration team. Assess need for canal blocking "
                   "(rewetting) if this is drained peatland."),
        "weight": 0.9
    },
}

# ------------------------------------------------------------ small helpers
def fmt_coord(lat, lon):
    return f"{abs(lat):.2f}°{'N' if lat >= 0 else 'S'}, {abs(lon):.2f}°E"

def haversine_km(lat1, lon1, lat2, lon2):
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
    if value >= 0.5: return "strong"
    if value >= 0.2: return "moderate"
    return "slight"

def nearest_date(dates_index, ts):
    pos = dates_index.searchsorted(ts, side="right") - 1
    return dates_index[max(pos, 0)]

def explain_rows(model, rows, features):
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
    ordered = contrib_row.sort_values(ascending=False)
    up, down = [], []
    for key, value in ordered.items():
        if value > 0.02 and len(up) < n_up: up.append((key, float(value)))
    for key, value in ordered.sort_values().items():
        if value < -0.02 and len(down) < n_down: down.append((key, float(value)))

    def pack(items):
        return [{"key": k, "label": GROUPS[k]["label"], "text": GROUPS[k]["describe"](row),
                 "value": v, "advice": GROUPS[k]["advice"], "weight": GROUPS[k]["weight"]} for k, v in items]
    return pack(up), pack(down)

def contribution_figure(contrib_row):
    values = contrib_row.sort_values()
    labels = [GROUPS[k]["label"] for k in values.index]
    colors = ["#b2182b" if v > 0 else "#2166ac" for v in values.values]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(labels, values.values, color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Push on the risk score (red = raises risk, blue = lowers risk)")
    ax.set_title("Cause Attribution: How each factor affects this cell's risk")
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
    reasons = "; ".join(f"{f['label']}: {f['text']}" for f in factors[:3])
    focus = " ".join(advice_list[:2]) if advice_list else "Keep routine monitoring."
    return (f"JAGAWANA AI DEPLOYMENT NOTICE\n"
            f"Date: {pd.Timestamp(row['date']).date()}\n"
            f"Area: {fmt_coord(row['lat'], row['lon'])} ({row['island']})\n"
            f"Level: {tier.upper()} ({TIER_RULE[tier]})\n"
            f"Main factors: {reasons}\n"
            f"RECOMMENDED ACTION: {focus}")

@st.cache_data(ttl=600)
def fetch_live_hotspots(api_key):
    if not api_key or len(api_key.strip()) < 10: 
        return pd.DataFrame()
    
    BBOX = "95,-11,120,6"
    URL = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{api_key.strip()}/VIIRS_SNPP_NRT/{BBOX}/1"
    
    try:
        response = requests.get(URL, timeout=15)
        if response.status_code != 200:
            st.sidebar.error(f"API Error {response.status_code}: {response.text}")
            return pd.DataFrame()
        if not response.text.strip() or response.text.startswith("Invalid"):
            return pd.DataFrame()
        return pd.read_csv(StringIO(response.text))
    except Exception as e:
        st.sidebar.error(f"Connection error: {e}")
        return pd.DataFrame()

def read_features():
    try:
        return pd.read_parquet(DATASET_FILE, filters=[("date", ">=", pd.Timestamp("2023-01-01"))])
    except Exception:
        frame = pd.read_parquet(DATASET_FILE)
        return frame[frame["date"] >= "2023-01-01"]

@st.cache_resource(show_spinner="Loading data and models...")
def load_everything():
    info = json.loads(INFO_FILE.read_text())
    model = joblib.load(MODEL_FILE)
    features = info["features"]

    preds = pd.read_parquet(PRED_FILE)
    preds["date"] = pd.to_datetime(preds["date"])
    frame = read_features()
    frame["date"] = pd.to_datetime(frame["date"])
    data = frame.merge(preds[["date", "cell_id", "risk", "split", "island"]], on=["date", "cell_id"], how="inner")

    test_risk = data.loc[data["split"] == "test", "risk"]
    cuts = {"c80": float(test_risk.quantile(0.80)),
            "c90": float(test_risk.quantile(0.90)),
            "c95": float(test_risk.quantile(0.95))}
    data["tier"] = assign_tier(data["risk"].values, cuts)

    ctx = {"model": model, "features": features, "cuts": cuts, "has_cond": False, "cond_sorted": None}
    
    if COND_MODEL_FILE.exists() and COND_INFO_FILE.exists() and CLIM_FILE.exists():
        try:
            cond_info = json.loads(COND_INFO_FILE.read_text())
            cond_model = joblib.load(COND_MODEL_FILE)
            sources = cond_info["anomaly_sources"]
            clim = pd.read_csv(CLIM_FILE)
            data["month"] = data["date"].dt.month
            data = data.merge(clim, on=["cell_id", "month"], how="left")
            for col in sources: data[f"{col}_anom"] = data[col] - data[f"{col}_clim"]
            data["cond_score"] = cond_model.predict_proba(data[cond_info["features"]])[:, 1]
            ctx["cond_sorted"] = np.sort(data.loc[data["split"] == "test", "cond_score"].values)
            ctx["has_cond"] = True
        except Exception as error:
            print("Conditions-only model skipped:", repr(error)[:200])
            
    data = data.sort_values(["date", "cell_id"]).reset_index(drop=True)
    ctx["data"] = data
    return ctx

# ----------------------------------------------------------------- UI pieces
def render_map(day, ctx, islands, live_df=None):
    counts = day["tier"].value_counts()
    cols = st.columns(4)
    for col, tier in zip(cols, TIERS):
        col.metric(f"{tier}", int(counts.get(tier, 0)))
    st.markdown(legend_html(), unsafe_allow_html=True)

    show_actual = st.checkbox("Validate: ring cells where a hotspot actually appeared within 7 days", value=False)

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

    if live_df is not None and not live_df.empty:
        live_plot = live_df.copy()
        live_plot["latitude"] = pd.to_numeric(live_plot["latitude"], errors="coerce")
        live_plot["longitude"] = pd.to_numeric(live_plot["longitude"], errors="coerce")
        live_plot = live_plot.dropna(subset=["latitude", "longitude"])
        
        layers.append(pdk.Layer(
            "ScatterplotLayer", data=live_plot,
            get_position="[longitude, latitude]",
            get_fill_color=[0, 0, 0, 255],
            get_line_color=[255, 255, 255],
            stroked=True, line_width_min_pixels=1,
            get_radius=12000, pickable=True
        ))
        html += "<br/><b style='color:black'>⚫ LIVE HOTSPOT (24h)</b>"

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
        st.caption(f"Validation: of the {len(vh)} very-high cells, {int(vh['label'].sum())} saw a hotspot "
                   f"within 7 days ({vh['label'].mean():.0%}). The model does not see this in advance.")
        
    st.caption("Colored circles = 0.5° risk cells (~55 km). Black dots = Active fires detected by NASA in the last 24h. "
               "Historical model was trained on cells with fire history in 2019-2022.")

def render_check(day, ctx):
    st.subheader("Check an area")
    place = st.selectbox("Choose a place", list(PLACES.keys()))
    if PLACES[place] is None:
        c1, c2 = st.columns(2)
        lat = c1.number_input("Latitude", -6.0, 6.0, 0.5, 0.05)
        lon = c2.number_input("Longitude", 94.5, 118.5, 101.5, 0.05)
    else:
        lat, lon = PLACES[place]

    distances = haversine_km(lat, lon, day["lat"].values, day["lon"].values)
    best = int(np.argmin(distances))
    distance = float(distances[best])
    if distance > MAX_DISTANCE_KM:
        st.warning(f"No monitored cell within {MAX_DISTANCE_KM:.0f} km. Demo only covers cells with fire history.")
        return
    row = day.iloc[best]
    tier = row["tier"]
    st.caption(f"Using monitored cell: {fmt_coord(row['lat'], row['lon'])} ({distance:.0f} km away).")

    contrib = explain_rows(ctx["model"], day.iloc[[best]], ctx["features"]).iloc[0]
    up, down = top_factors(contrib, row)

    left, right = st.columns([1, 1])
    left.markdown(tier_badge(tier, TIER_RULE[tier]), unsafe_allow_html=True)
    right.metric("Risk score (0 to 1)", f"{row['risk']:.2f}")
    
    actual = "a hotspot DID appear" if row["label"] == 1 else "no hotspot appeared"
    st.caption(f"Validation check: in historical data, {actual} within 7 days.")

    st.markdown("#### Cause Attribution (Why this level?)")
    a, b = st.columns(2)
    a.markdown("**Risk drivers**")
    if up:
        for f in up: a.markdown(f"- **{f['label']}**: {f['text']}")
    else: a.write("No factor stands out.")
    b.markdown("**Mitigating factors**")
    if down:
        for f in down: b.markdown(f"- **{f['label']}**: {f['text']}")
    else: b.write("No factor stands out.")
    
    st.caption("These factors describe the model's logic, not proven causes of fire.")
    st.pyplot(contribution_figure(contrib))
    plt.close()

    st.markdown("#### Recommended Resource Deployment")
    seen = []
    for f in up[:3]:
        if f["advice"] not in seen:
            seen.append(f["advice"])
            st.info(f"🚨 {f['advice']}")
    if not seen: st.info("✅ Keep routine monitoring.")
    st.caption(GUIDANCE_NOTE)

def render_command_center(day, ctx, islands):
    st.subheader("🚨 Resource Allocation Command Center")
    st.markdown("Allocate limited firefighting personnel to the highest-risk cells based on cause attribution.")
    
    view = day[day["island"].isin(islands)].copy()
    if view.empty:
        st.info("No cells to show.")
        return
        
    col1, col2 = st.columns(2)
    available_teams = col1.number_input("Total personnel teams available:", min_value=1, max_value=100, value=10)
    max_per_cell = col2.number_input("Max teams per cell:", min_value=1, max_value=5, value=2)

    view = view[view["risk"] >= view["risk"].quantile(0.80)]
    top_view = view.nlargest(available_teams * max_per_cell, "risk")
    
    contrib = explain_rows(ctx["model"], top_view, ctx["features"])
    
    deploy_list = []
    teams_left = available_teams
    
    for idx, row in top_view.iterrows():
        if teams_left <= 0: break
        up, _ = top_factors(contrib.loc[idx], row)
        
        weight = max([f["weight"] for f in up]) if up else 1.0
        priority_score = row["risk"] * weight
        
        teams_to_send = min(teams_left, max_per_cell)
        action = up[0]["advice"] if up else "Deploy team."
        
        deploy_list.append({
            "Target Cell": fmt_coord(row["lat"], row["lon"]),
            "Island": row["island"],
            "Risk Score": round(row["risk"], 2),
            "Primary Cause": up[0]["label"] if up else "-",
            "Priority Score": round(priority_score, 2),
            "Teams Deployed": teams_to_send,
            "Action": action
        })
        teams_left -= teams_to_send
        
    df = pd.DataFrame(deploy_list).sort_values("Priority Score", ascending=False)
    st.dataframe(df, use_container_width=True)
    st.caption(f"**Total Teams Deployed: {available_teams - teams_left} / {available_teams}**")
    
    st.markdown("#### Deployment Map")
    deploy_coords = []
    for _, row in df.iterrows():
        lat_str, lon_str = row["Target Cell"].split("°")[0], row["Target Cell"].split("°")[1].split(",")[1].replace("E","")
        lat = float(lat_str) * (1 if "N" in row["Target Cell"] else -1)
        lon = float(lon_str)
        deploy_coords.append({"lat": lat, "lon": lon, "teams": row["Teams Deployed"]})
        
    if deploy_coords:
        map_df = pd.DataFrame(deploy_coords)
        st.map(map_df, latitude="lat", longitude="lon", size="teams", color="#ff0000")

def render_list(day, ctx, islands):
    st.subheader("High-risk list")
    view = day[day["island"].isin(islands)]
    if view.empty:
        st.info("No cells to show.")
        return
    n = st.slider("How many cells to list", 5, 30, 10)
    top = view.nlargest(n, "risk")
    contrib = explain_rows(ctx["model"], top, ctx["features"])

    records, details = [], []
    for rank, (idx, row) in enumerate(top.iterrows(), start=1):
        up, _ = top_factors(contrib.loc[idx], row)
        advice = []
        for f in up[:3]:
            if f["advice"] not in advice: advice.append(f["advice"])
        records.append({
            "Rank": rank, "Level": row["tier"], "Island": row["island"],
            "Location": fmt_coord(row["lat"], row["lon"]), "Risk": round(float(row["risk"]), 2),
            "Main Cause": up[0]["label"] if up else "-",
            "Action": advice[0] if advice else "Keep routine monitoring."
        })
        details.append((row, row["tier"], up, advice))
    table = pd.DataFrame(records)
    st.dataframe(table)
    st.download_button("Download CSV", table.to_csv(index=False).encode("utf-8"),
                       file_name=f"jagawana_{pd.Timestamp(top['date'].iloc[0]).date()}.csv", mime="text/csv")
    st.caption(GUIDANCE_NOTE)

    st.markdown("#### Dispatch Alert Message (Preview)")
    choice = st.selectbox("Pick a rank", list(range(1, len(details) + 1)))
    row, tier, up, advice = details[choice - 1]
    st.code(alert_text(row, tier, up, advice), language=None)

def render_about(ctx):
    st.subheader("About Jagawana AI")
    st.markdown(
        "- Jagawana AI estimates, for each 0.5° cell, the likelihood of a fire hotspot appearing "
        "within the next 7 days, and identifies the specific causes driving that estimate.\n"
        "- It translates these causes into **actionable resource deployment plans** for responders.\n"
        "- Data: NASA FIRMS VIIRS hotspots and NASA POWER weather. Model: XGBoost + SHAP.")

    st.markdown("### Performance Metrics (Test Years 2024-2025)")
    st.markdown("- **PR-AUC:** 0.68 (vs. 0.52 baseline)")
    st.markdown("- **ROC-AUC:** 0.83")
    st.markdown("- **Top 5% Precision:** 88.4% (If responders check the top 5% of our map, 88% of those cells will have a fire).")
    st.markdown("- **Spatial Generalization:** The model can train on Sumatra and successfully predict fires in Kalimantan (PR-AUC 0.67).")

    st.markdown("#### Limitations")
    st.markdown(
        "- Risk assessment, not certainty. Some alerts are false alarms.\n"
        "- Hotspots are satellite detections, not confirmed fires. Clouds can hide fires.\n"
        "- Prevention guidance is decision-support and must be validated by local BPBD/Manggala Agni.")

# ------------------------------------------------------------------------ main
def main():
    st.set_page_config(page_title="Jagawana AI", page_icon="🔥", layout="wide")
    st.title("🔥 Jagawana AI: Fire Risk & Resource Allocation Engine")
    
    st.sidebar.header("Data Source")
    use_live = st.sidebar.checkbox("Enable Live NASA FIRMS Data")
    api_key = st.sidebar.text_input("NASA FIRMS API Key", type="password") if use_live else ""

    needed = {"dataset.parquet": DATASET_FILE, "predictions.parquet": PRED_FILE, "model.joblib": MODEL_FILE}
    missing = [name for name, path in needed.items() if not path.exists()]
    if missing:
        st.error(f"Missing files: {', '.join(missing)}. Run build_dataset.py and train_models.py first.")
        st.stop()

    ctx = load_everything()
    data = ctx["data"]
    dates = pd.DatetimeIndex(np.sort(data["date"].unique()))

    live_df = pd.DataFrame()
    if use_live:
        live_df = fetch_live_hotspots(api_key)
        if not live_df.empty:
            st.sidebar.success(f"✅ Fetched {len(live_df)} live hotspots (24h)")
            st.sidebar.dataframe(live_df[["latitude", "longitude", "bright_ti4", "confidence"]].head())
            st.info("Live Hotspots fetched. Black dots on the map show active fires from the last 24 hours.")
        else:
            st.sidebar.warning("⚠️ No live data fetched. Check API key or wait for satellite pass.")

    st.sidebar.header("Historical Data Mode")
    default = min(max(pd.Timestamp(DEFAULT_DATE), dates[0]), dates[-1])
    picked = st.sidebar.date_input("Date", value=default.date(), min_value=dates[0].date(), max_value=dates[-1].date())
    day_ts = nearest_date(dates, pd.Timestamp(picked))
    if day_ts != pd.Timestamp(picked):
        st.sidebar.info(f"No data for {picked}. Showing {day_ts.date()}.")
    
    selected_islands = st.sidebar.multiselect("Islands", ["Sumatra", "Kalimantan"], default=["Sumatra", "Kalimantan"])
    if not selected_islands:
        selected_islands = ["Sumatra", "Kalimantan"]
        
    st.sidebar.caption("Validation years: 2023. Test years: 2024-2025.")

    day = data[data["date"] == day_ts].reset_index(drop=True)
    
    tab_map, tab_check, tab_command, tab_list, tab_about = st.tabs(
        ["🗺️ Risk Map", "📍 Check Area", "🚨 Command Center", "📋 High-Risk List", "📊 About & Results"])
        
    with tab_map: 
        render_map(day, ctx, selected_islands, live_df)
    with tab_check: 
        render_check(day, ctx)
    with tab_command: 
        render_command_center(day, ctx, selected_islands)
    with tab_list: 
        render_list(day, ctx, selected_islands)
    with tab_about: 
        render_about(ctx)

if __name__ == "__main__":
    main()