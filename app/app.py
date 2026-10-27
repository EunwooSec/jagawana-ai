"""Jagawana AI - Decision Support Engine (Final Submission Build).

Run from the project folder:
    python -m streamlit run app\\app.py
"""
import json
import requests
from io import StringIO
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydeck as pdk
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

BNPB_TOTAL_PERSONNEL = 71878
BNPB_PERSONNEL_PER_REGU = 15

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

# ------------------------------------------------------------------- translations
TEXT = {
    "EN": {
        "title": "Jagawana AI: Fire Risk & Resource Allocation Engine",
        "sidebar_data": "Live Data Source",
        "sidebar_live": "Enable Live NASA FIRMS Data",
        "sidebar_api": "NASA FIRMS API Key",
        "sidebar_settings": "⚙️ App Settings",
        "sidebar_lang": "Language / Bahasa",
        "sidebar_hist": "📅 Forecast Date (Select Past Date)",
        "sidebar_date": "Select Date",
        "sidebar_help": "Select a date to generate the 7-day forecast for that period. To see live fires happening right now, enable the Live NASA toggle above.",
        "sidebar_islands": "Islands",
        "map_show_grid": "Show Historical Risk Grid",
        "map_show_live": "Show Live Hotspots (24h)",
        "map_validate": "Validate 7-day Rings",
        "map_caption": "Colored circles = 0.5° risk cells (~55 km). Orange dots = Active fires (NASA). Black rings = Actual fires that occurred (Validation).",
        "check_title": "📍 Check Area (Forecast)",
        "check_place": "Choose a place",
        "check_causes": "Cause Attribution (Why this level?)",
        "check_drivers": "Risk drivers",
        "check_mitigating": "Mitigating factors",
        "check_advice": "Recommended Resource Deployment",
        "cc_title": "🚨 Resource Allocation Command Center",
        "cc_desc": "Allocate limited firefighting personnel based on cause attribution and BNPB baseline data.",
        "cc_total_personnel": "National Baseline (BNPB Oct 2026)",
        "cc_teams_avail": "Total Deployment Teams Available (1 Team = 15 personnel)",
        "cc_max_per_cell": "Max Teams per Cell",
        "cc_deploy_table": "Deployment Plan",
        "cc_deploy_caption": "Allocation uses a greedy algorithm to maximize risk reduction.",
        "list_title": "📋 High-Risk List",
        "about_title": "📊 About & Results",
        "tier_vhigh": "Very high",
        "tier_high": "High",
        "tier_elev": "Elevated",
        "tier_low": "Lower",
        "risk_score": "Risk score",
        "primary_cause": "Primary Cause",
        "priority": "Priority",
        "teams_sent": "Teams Deployed",
        "action": "Action",
    },
    "ID": {
        "title": "Jagawana AI: Mesin Risiko & Alokasi Sumber Daya",
        "sidebar_data": "Sumber Data Live",
        "sidebar_live": "Aktifkan Data Live NASA FIRMS",
        "sidebar_api": "API Key NASA FIRMS",
        "sidebar_settings": "⚙️ Pengaturan Aplikasi",
        "sidebar_lang": "Language / Bahasa",
        "sidebar_hist": "📅 Tanggal Prakiraan (Pilih Tanggal Lampau)",
        "sidebar_date": "Pilih Tanggal",
        "sidebar_help": "Pilih tanggal untuk menghasilkan prakiraan 7 hari pada periode tersebut. Untuk melihat kebakaran live saat ini, aktifkan toggle Live NASA di atas.",
        "sidebar_islands": "Pulau",
        "map_show_grid": "Tampilkan Grid Risiko Historis",
        "map_show_live": "Tampilkan Hotspot Live (24 Jam)",
        "map_validate": "Validasi Ring 7-Hari",
        "map_caption": "Lingkaran berwarna = sel risiko 0.5° (~55 km). Titik oranye = Api aktif (NASA). Cincin hitam = Kebakaran aktual (Validasi).",
        "check_title": "📍 Cek Area (Prakiraan)",
        "check_place": "Pilih lokasi",
        "check_causes": "Atribusi Penyebab (Kenapa level ini?)",
        "check_drivers": "Pemicu risiko",
        "check_mitigating": "Faktor pereduksi",
        "check_advice": "Rekomendasi Alokasi Sumber Daya",
        "cc_title": "🚨 Pusat Komando Alokasi Sumber Daya",
        "cc_desc": "Alokasikan personel pemadam terbatas berdasarkan atribusi penyebab dan data baseline BNPB.",
        "cc_total_personnel": "Baseline Nasional (BNPB Okt 2026)",
        "cc_teams_avail": "Total Tim Deploy Tersedia (1 Tim = 15 personel)",
        "cc_max_per_cell": "Maks Tim per Sel",
        "cc_deploy_table": "Rencana Deployment",
        "cc_deploy_caption": "Alokasi menggunakan algoritma greedy untuk memaksimalkan pengurangan risiko.",
        "list_title": "📋 Daftar Risiko Tinggi",
        "about_title": "📊 Tentang & Hasil",
        "tier_vhigh": "Sangat Tinggi",
        "tier_high": "Tinggi",
        "tier_elev": "Meningkat",
        "tier_low": "Rendah",
        "risk_score": "Skor risiko",
        "primary_cause": "Penyebab Utama",
        "priority": "Prioritas",
        "teams_sent": "Tim Dikerahkan",
        "action": "Tindakan",
    }
}

def t(key, lang):
    return TEXT[lang].get(key, key)

# ------------------------------------------------- factor groups and wording
def _days_since_text(days):
    if days >= 365: return "no hotspot recorded here in the past year"
    if days <= 0: return "a hotspot was detected here today"
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
        "advice": ("Deploy 1 Preventive Patrol Team (15 personnel). Coordinate with "
                   "neighbouring villages and monitor boundaries."),
        "weight": 1.1
    },
    "rain": {
        "features": ["rain_7d", "rain_14d", "rain_30d", "rain_60d"],
        "label": "Recent rainfall",
        "describe": lambda r: (
            f"{r['rain_7d']:.0f} mm in the last 7 days, {r['rain_30d']:.0f} mm in 30 days"),
        "advice": ("Deploy Preventive Patrol Team (15 personnel). Reinforce burn-ban messages and check "
                   "canals and water points."),
        "weight": 1.0
    },
    "dry_streak": {
        "features": ["dry_streak"],
        "label": "Dry spell length",
        "describe": lambda r: f"{int(r['dry_streak'])} days in a row with under 1 mm of rain",
        "advice": ("Pre-position 2 firefighting teams (30 personnel) and 1 thermal drone. Raise patrol frequency."),
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
        "advice": ("Alert local BRGM peat restoration team (15 personnel). Assess need for canal blocking "
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

# --- FINAL 5-TIER DYNAMIC ADVICE FUNCTION ---
def get_dynamic_advice(risk, base_advice):
    """
    Doctrine-based allocation strictly following BMKG FDRS Levels and PermenLHK P.32/2016.
    Unit size: 1 Regu = 15 personnel.
    """

    # --- TIER 1: NEGLIGIBLE RISK (< 0.30) ---
    if risk < 0.30:
        return "No immediate deployment required. Notify local community (MPA) for routine monitoring."

    # --- TIER 2: MODERATE RISK (0.30 - 0.49) ---
    elif risk < 0.50:
        return "Deploy 1 Preventive Patrol Team (15 personnel). Conduct 12-hour patrol and public socialization. No water pumps required."

    # --- TIER 3: HIGH RISK (0.50 - 0.74) ---
    elif risk < 0.75:
        return "Deploy 1 Rapid Reaction Team (15 personnel) + 1 water pump. Standby on location. Conduct 24-hour patrol."

    # --- TIER 4: VERY HIGH RISK (0.75 - 0.89) ---
    elif risk < 0.90:
        return "Deploy 2 Rapid Suppression Teams (30 personnel) + 2 water pumps. Initiate immediate suppression and ground verification."

    # --- TIER 5: EXTREME RISK (0.90 - 1.00) ---
    else:
        return "Deploy 3 Rapid Suppression Teams (45 personnel) + 3 water pumps. Initiate full strike force suppression immediately."
# ------------------------------------------------------------------

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

def tier_badge(tier, lang, subtitle=""):
    r, g, b = TIER_RGB[tier]
    text_color = "#222" if tier in ("Elevated", "Lower") else "white"
    tier_name = t(f"tier_{'vhigh' if tier=='Very high' else 'high' if tier=='High' else 'elev' if tier=='Elevated' else 'low'}", lang)
    return (f"<div style='background:rgb({r},{g},{b});color:{text_color};padding:14px 18px;"
            f"border-radius:8px;font-size:1.4rem'><b>{tier_name} risk</b>"
            f"<div style='font-size:0.9rem'>{subtitle}</div></div>")

@st.cache_data(ttl=600)
def fetch_live_hotspots(api_key):
    if not api_key or len(api_key.strip()) < 10: return pd.DataFrame()
    BBOX = "95,-11,120,6"
    URL = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{api_key.strip()}/VIIRS_SNPP_NRT/{BBOX}/1"
    try:
        response = requests.get(URL, timeout=15)
        if response.status_code != 200: return pd.DataFrame()
        if not response.text.strip() or response.text.startswith("Invalid"): return pd.DataFrame()
        return pd.read_csv(StringIO(response.text))
    except Exception:
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
def render_map(day, ctx, islands, live_df, lang):
    counts = day["tier"].value_counts()
    cols = st.columns(4)
    for col, tier in zip(cols, TIERS):
        col.metric(t(f"tier_{'vhigh' if tier=='Very high' else 'high' if tier=='High' else 'elev' if tier=='Elevated' else 'low'}", lang), int(counts.get(tier, 0)))

    c1, c2, c3 = st.columns(3)
    show_grid = c1.checkbox(t("map_show_grid", lang), value=True)
    show_live = c2.checkbox(t("map_show_live", lang), value=(live_df is not None and not live_df.empty))
    show_actual = c3.checkbox(t("map_validate", lang), value=False)

    view = day[day["island"].isin(islands)].copy()
    if view.empty and not show_live:
        st.info("No cells to show for this selection.")
        return
        
    layers = []

    # 1. Historical Grid Layer
    if show_grid and not view.empty:
        view["color"] = view["tier"].map(lambda t: list(TIER_RGB[t]) + [210])
        view["lat"] = view["lat"].astype(float)
        view["lon"] = view["lon"].astype(float)
        
        view["tooltip"] = (
            "Risk Level: " + view["tier"].astype(str) + 
            "\nScore: " + view["risk"].round(2).astype(str) + 
            "\nLat/Lon: " + view["lat"].round(2).astype(str) + ", " + view["lon"].round(2).astype(str)
        )

        layers.append(pdk.Layer(
            "ScatterplotLayer", 
            data=view,
            get_position="[lon, lat]", 
            get_radius=25000, 
            get_fill_color="color",
            pickable=True, 
            opacity=0.8, 
            stroked=False
        ))

    # 2. Live Hotspots Layer
    if show_live and live_df is not None and not live_df.empty:
        live_plot = live_df.copy()
        live_plot["latitude"] = pd.to_numeric(live_plot["latitude"], errors="coerce")
        live_plot["longitude"] = pd.to_numeric(live_plot["longitude"], errors="coerce")
        live_plot = live_plot.dropna(subset=["latitude", "longitude"])
        
        live_plot["tooltip"] = (
            "🔥 LIVE HOTSPOT (24h)\n" +
            "Lat/Lon: " + live_plot["latitude"].round(3).astype(str) + ", " + live_plot["longitude"].round(3).astype(str)
        )
        
        layers.append(pdk.Layer(
            "ScatterplotLayer", 
            data=live_plot,
            get_position="[longitude, latitude]",
            get_fill_color=[255, 69, 0, 220],
            stroked=False,
            get_radius=1000,
            pickable=True
        ))

    # 3. Validation Rings Layer
    if show_actual and not view.empty:
        rings = view.loc[view["label"] == 1, ["lon", "lat"]]
        layers.append(pdk.Layer(
            "ScatterplotLayer", 
            data=rings, 
            get_position="[lon, lat]", 
            get_radius=25000,
            filled=False, 
            stroked=True, 
            get_line_color=[0, 0, 0, 255],
            line_width_min_pixels=2, 
            pickable=False
        ))
        
    deck = pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(latitude=-1.5, longitude=113.0, zoom=4.8, pitch=0),
        map_provider="carto", 
        map_style="light", 
        tooltip={
            "text": "{tooltip}",
            "style": {"backgroundColor": "#1e1e1e", "color": "white", "borderRadius": "5px", "padding": "10px", "fontSize": "13px"}
        }
    )
    
    deck_html = deck.to_html(as_string=True)
    deck_html = deck_html.replace("<body>", "<body oncontextmenu='return false'>")
    components.html(deck_html, height=650)

    st.caption(t("map_caption", lang))

def render_check(day, ctx, lang):
    st.subheader(t("check_title", lang))
    st.caption("This tab visualizes the 7-day forecast for the selected date. To see live fires, enable the Live NASA toggle in the sidebar.")
    
    place = st.selectbox(t("check_place", lang), list(PLACES.keys()))
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
        st.warning(f"No monitored cell within {MAX_DISTANCE_KM:.0f} km.")
        return
    row = day.iloc[best]
    tier = row["tier"]
    st.caption(f"Using monitored cell: {fmt_coord(row['lat'], row['lon'])} ({distance:.0f} km away).")

    contrib = explain_rows(ctx["model"], day.iloc[[best]], ctx["features"]).iloc[0]
    up, down = top_factors(contrib, row)

    left, right = st.columns([1, 1])
    left.markdown(tier_badge(tier, lang), unsafe_allow_html=True)
    right.metric(t("risk_score", lang), f"{row['risk']:.2f}")
    
    st.markdown(f"#### {t('check_causes', lang)}")
    a, b = st.columns(2)
    a.markdown(f"**{t('check_drivers', lang)}**")
    if up:
        for f in up: a.markdown(f"- **{f['label']}**: {f['text']}")
    else: a.write("-")
    b.markdown(f"**{t('check_mitigating', lang)}**")
    if down:
        for f in down: b.markdown(f"- **{f['label']}**: {f['text']}")
    else: b.write("-")
    
    st.pyplot(contribution_figure(contrib))
    plt.close()

    st.markdown(f"#### {t('check_advice', lang)}")
    dynamic_advice = get_dynamic_advice(row["risk"], up[0]["advice"] if up else "")
    st.info(f"🚨 {dynamic_advice}")

def render_command_center(day, ctx, islands, lang):
    st.subheader(t("cc_title", lang))
    st.markdown(t("cc_desc", lang))
    
    view = day[day["island"].isin(islands)].copy()
    if view.empty:
        st.info("No cells to show.")
        return

    st.info(f"🏛️ **{t('cc_total_personnel', lang)}:** {BNPB_TOTAL_PERSONNEL:,} personnel across 6 priority provinces (Baseline).")
        
    available_teams = st.number_input(t("cc_teams_avail", lang), min_value=1, max_value=1000, value=100)

    with st.spinner("Calculating optimal deployment plan..."):
        view = view[view["risk"] >= view["risk"].quantile(0.80)]
        top_view = view.nlargest(available_teams * 3, "risk") 
        
        contrib = explain_rows(ctx["model"], top_view, ctx["features"])
        
        deploy_list = []
        teams_left = available_teams
        
        for idx, row in top_view.iterrows():
            if teams_left <= 0: break
            
            # --- HARD CUTOFF: DO NOT DEPLOY IF RISK IS BELOW 0.30 ---
            if row["risk"] < 0.30:
                continue
                
            up, _ = top_factors(contrib.loc[idx], row)
            
            weight = max([f["weight"] for f in up]) if up else 1.0
            priority_score = row["risk"] * weight
            
            # Determine teams based on the 5-tier logic
            if row["risk"] >= 0.90:
                needed_teams = 3
            elif row["risk"] >= 0.75:
                needed_teams = 2
            else:
                needed_teams = 1
                
            teams_to_send = min(needed_teams, teams_left)
            
            action = get_dynamic_advice(row["risk"], up[0]["advice"] if up else "")

            reason = f"Risk {row['risk']:.2f} + " + (f"{up[0]['label']} (Weight {weight})" if up else "General")
            
            deploy_list.append({
                "Target Cell": fmt_coord(row["lat"], row["lon"]),
                "Island": row["island"],
                "Lat": row["lat"],
                "Lon": row["lon"],
                t("risk_score", lang): round(row["risk"], 2),
                t("primary_cause", lang): up[0]["label"] if up else "-",
                "Reason": reason,
                t("priority", lang): round(priority_score, 2),
                t("teams_sent", lang): teams_to_send,
                t("action", lang): action
            })
            teams_left -= teams_to_send
            
        deployed = available_teams - teams_left
        if deployed > 0:
            st.success(f"✅ **Deployment Successful:** {deployed} Teams ({deployed * BNPB_PERSONNEL_PER_REGU:,} total personnel) dispatched to {len(deploy_list)} priority hotspots.")
        else:
            st.info("ℹ️ **No critical hotspots detected.** All monitored cells are below the 0.30 risk threshold. Keep all teams on standby.")

    if deployed > 0:
        df = pd.DataFrame(deploy_list).sort_values(t("priority", lang), ascending=False)
        df.index = range(1, len(df) + 1) # FIX: Command Center Index
        
        st.download_button(
            label="📥 Download Deployment Plan (CSV)",
            data=df.drop(columns=["Lat", "Lon"]).to_csv(index=False).encode('utf-8'),
            file_name=f"deployment_plan_{day['date'].iloc[0].date()}.csv",
            mime="text/csv",
            help="Download the deployment plan to share with field commanders."
        )

        display_df = df.drop(columns=["Lat", "Lon"])
        st.dataframe(display_df, use_container_width=True)
        st.caption(t("cc_deploy_caption", lang))
        
        st.markdown(f"#### {t('cc_deploy_table', lang)} Map")
        if not df.empty:
            df["tooltip"] = df["Target Cell"] + "\n" + df["Reason"] + "\n" + df[t("teams_sent", lang)].astype(str) + " Teams"
            
            map_layer = pdk.Layer(
                "ScatterplotLayer",
                data=df,
                get_position="[Lon, Lat]",
                get_fill_color=[255, 0, 0, 200],
                get_radius=15000,
                pickable=True
            )
            
            map_deck = pdk.Deck(
                layers=[map_layer],
                initial_view_state=pdk.ViewState(latitude=-1.5, longitude=113.0, zoom=4.5),
                map_provider="carto",
                map_style="dark",
                tooltip={
                    "text": "{tooltip}",
                    "style": {"backgroundColor": "#ff0000", "color": "white"}
                }
            )
            st.pydeck_chart(map_deck)
    else:
        st.dataframe(pd.DataFrame()) 

def render_list(day, ctx, islands, lang):
    st.subheader(t("list_title", lang))
    view = day[day["island"].isin(islands)]
    if view.empty:
        st.info("No cells to show.")
        return
    n = st.slider("How many cells to list", 5, 30, 10)
    top = view.nlargest(n, "risk")
    contrib = explain_rows(ctx["model"], top, ctx["features"])

    records = []
    for rank, (idx, row) in enumerate(top.iterrows(), start=1):
        up, _ = top_factors(contrib.loc[idx], row)
        dynamic_advice = get_dynamic_advice(row["risk"], up[0]["advice"] if up else "Keep routine monitoring.")
        
        records.append({
            "Rank": rank, "Level": row["tier"], "Island": row["island"],
            "Location": fmt_coord(row["lat"], row["lon"]), t("risk_score", lang): round(float(row["risk"]), 2),
            t("primary_cause", lang): up[0]["label"] if up else "-",
            t("action", lang): dynamic_advice
        })
    table = pd.DataFrame(records)
    
    table.index = range(1, len(table) + 1) # FIX: High-Risk List Index
    
    st.dataframe(table)
    st.download_button("Download CSV", table.to_csv(index=False).encode("utf-8"),
                       file_name=f"jagawana_{pd.Timestamp(top['date'].iloc[0]).date()}.csv", mime="text/csv")

def render_about(lang):
    st.subheader(t("about_title", lang))
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

    st.markdown("### Future Work (Phase 2)")
    st.markdown("- **Peatland Integration:** Integrate spatial peatland depth data (from BRGM/GFW) to specifically predict underground peat fires.")
    st.markdown("- **Live BNPB API:** Transition from our static 71,878 personnel baseline to a fully live API feed if granted access by the government.")

# ------------------------------------------------------------------------ main
def main():
    st.set_page_config(page_title="Jagawana AI", page_icon="🔥", layout="wide", initial_sidebar_state="expanded")
    
    st.markdown("""
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .stAppDeployButton {display: none;}
        [data-testid="collapsedControl"] { display: block !important; visibility: visible !important; z-index: 1000; }
        .block-container { padding-top: 3rem !important; padding-bottom: 0rem !important; }
        </style>
    """, unsafe_allow_html=True)
    
    if 'lang' not in st.session_state:
        st.session_state.lang = "EN"

    with st.sidebar.expander(t("sidebar_settings", st.session_state.lang)):
        lang_choice = st.radio(t("sidebar_lang", st.session_state.lang), ["EN", "ID"], index=0 if st.session_state.lang == "EN" else 1)
        st.session_state.lang = lang_choice

    lang = st.session_state.lang

    col1, col2 = st.columns([0.5, 15])
    with col1:
        st.markdown("<h1 style='text-align: center; margin-top: 5px;'>🔥</h1>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<h1 style='margin-top: 5px; line-height: 1.2;'>{t('title', lang)}</h1>", unsafe_allow_html=True)
    
    st.sidebar.header(t("sidebar_data", lang))
    use_live = st.sidebar.checkbox(t("sidebar_live", lang))
    api_key = st.sidebar.text_input(t("sidebar_api", lang), value="") if use_live else ""

    needed = {"dataset.parquet": DATASET_FILE, "predictions.parquet": PRED_FILE, "model.joblib": MODEL_FILE}
    missing = [name for name, path in needed.items() if not path.exists()]
    if missing:
        st.error(f"Missing files: {', '.join(missing)}.")
        st.stop()

    ctx = load_everything()
    data = ctx["data"]
    dates = pd.DatetimeIndex(np.sort(data["date"].unique()))

    live_df = pd.DataFrame()
    if use_live:
        live_df = fetch_live_hotspots(api_key)
        if not live_df.empty:
            st.sidebar.success(f"✅ Fetched {len(live_df)} live hotspots (24h)")
        else:
            st.sidebar.info("ℹ️ No active hotspots detected in the last 24h.")

    st.sidebar.header(t("sidebar_hist", lang))
    default = min(max(pd.Timestamp(DEFAULT_DATE), dates[0]), dates[-1])
    picked = st.sidebar.date_input(
        t("sidebar_date", lang), 
        value=default.date(), 
        min_value=dates[0].date(), 
        max_value=dates[-1].date(),
        help=t("sidebar_help", lang)
    )
    day_ts = nearest_date(dates, pd.Timestamp(picked))
    
    selected_islands = st.sidebar.multiselect(t("sidebar_islands", lang), ["Sumatra", "Kalimantan"], default=["Sumatra", "Kalimantan"])
    if not selected_islands: selected_islands = ["Sumatra", "Kalimantan"]

    day = data[data["date"] == day_ts].reset_index(drop=True)
    
    tab_map, tab_check, tab_command, tab_list, tab_about = st.tabs(
        [t("map_show_grid", lang), t("check_title", lang), t("cc_title", lang), t("list_title", lang), t("about_title", lang)])
        
    with tab_map: render_map(day, ctx, selected_islands, live_df, lang)
    with tab_check: render_check(day, ctx, lang)
    with tab_command: render_command_center(day, ctx, selected_islands, lang)
    with tab_list: render_list(day, ctx, selected_islands, lang)
    with tab_about: render_about(lang)

if __name__ == "__main__":
    main()