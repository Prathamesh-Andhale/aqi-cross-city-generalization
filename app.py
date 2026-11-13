#!/usr/bin/env python3
import sys, os

# Auto re-exec with project .venv python if started with global system Python
VENV_PYTHON = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "bin", "python")
if os.path.exists(VENV_PYTHON) and os.path.abspath(sys.executable) != os.path.abspath(VENV_PYTHON):
    os.execv(VENV_PYTHON, [VENV_PYTHON] + sys.argv)

"""
app.py  —  AQI Cross-City Generalization Dashboard & Live Telemetry Service
==========================================================================
Loads pre-trained XGBoost, LightGBM, and BiLSTM models across 3 geographic cohorts.
Provides interactive exploration, TreeSHAP explainability, and real-time CPCB API integration.

Deployment:
    gunicorn app:app --workers 2 --bind 0.0.0.0:$PORT
    (Or locally: python app.py)
"""

import json, time, threading, warnings
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import joblib
import requests
from flask import Flask, render_template, jsonify, request

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ── Paths & Constants ──────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
PORT       = int(os.environ.get("PORT", 5001))

DATA_GOV_API_KEY = os.environ.get(
    "DATA_GOV_API_KEY",
    "579b464db66ec23bdd000001647a31e9067245075fcd6027f0cb014c"
)

CITY_ORDER = ("Mumbai", "Pune", "Chennai", "Delhi", "Kanpur", "Lucknow", "Bengaluru", "Mysuru", "Hyderabad")
AQI_BANDS  = ("Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe")
PM25_TREND_WINDOW = 14
SCATTER_STRIDE = 7

BASE_FEATURES = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
ENG_FEATURES  = BASE_FEATURES + ["PM_ratio"]
FEATURES      = ENG_FEATURES          # Clean 7 features

GROUP_TO_CITY = {
    "Group1_Pune_Chennai_Mumbai": "Mumbai",
    "Group2_Delhi_Kanpur_Lucknow": "Lucknow",
    "Group3_Bengaluru_Mysuru_Hyderabad": "Hyderabad",
}

CITY_TO_GROUP = {
    "Mumbai": "Group1_Pune_Chennai_Mumbai",
    "Pune": "Group1_Pune_Chennai_Mumbai",
    "Chennai": "Group1_Pune_Chennai_Mumbai",
    "Lucknow": "Group2_Delhi_Kanpur_Lucknow",
    "Delhi": "Group2_Delhi_Kanpur_Lucknow",
    "Kanpur": "Group2_Delhi_Kanpur_Lucknow",
    "Hyderabad": "Group3_Bengaluru_Mysuru_Hyderabad",
    "Bengaluru": "Group3_Bengaluru_Mysuru_Hyderabad",
    "Mysuru": "Group3_Bengaluru_Mysuru_Hyderabad",
}

# Verified historical medians from 841-day training dataset (clean fallback)
CITY_MEDIANS = {
    "Mumbai":    {"PM2.5": 38.26, "PM10": 103.92, "NO2": 23.44, "SO2": 2.72,  "CO": 0.58, "O3": 25.11},
    "Pune":      {"PM2.5": 31.47, "PM10": 79.44,  "NO2": 16.63, "SO2": 3.43,  "CO": 0.57, "O3": 36.55},
    "Chennai":   {"PM2.5": 27.76, "PM10": 58.00,  "NO2": 12.32, "SO2": 7.38,  "CO": 0.97, "O3": 8.74},
    "Delhi":     {"PM2.5": 93.81, "PM10": 274.42, "NO2": 64.47, "SO2": 19.39, "CO": 1.74, "O3": 17.64},
    "Kanpur":    {"PM2.5": 45.71, "PM10": 68.06,  "NO2": 36.51, "SO2": 10.16, "CO": 1.17, "O3": 23.51},
    "Lucknow":   {"PM2.5": 65.40, "PM10": 139.12, "NO2": 31.95, "SO2": 6.64,  "CO": 0.52, "O3": 24.91},
    "Bengaluru": {"PM2.5": 33.57, "PM10": 83.42,  "NO2": 9.36,  "SO2": 5.52,  "CO": 0.72, "O3": 27.38},
    "Mysuru":    {"PM2.5": 18.14, "PM10": 49.11,  "NO2": 16.77, "SO2": 3.44,  "CO": 0.36, "O3": 29.97},
    "Hyderabad": {"PM2.5": 44.62, "PM10": 104.60, "NO2": 16.33, "SO2": 3.92,  "CO": 0.63, "O3": 23.34},
}

app    = Flask(__name__)
_CACHE = {}
_API_CACHE = {}  # 15-minute in-memory cache for API calls


# ── Model file paths ───────────────────────────────────────────────────────────
def _paths_for_group(group_name=None):
    if group_name is None:
        base = MODELS_DIR
    else:
        base = os.path.join(MODELS_DIR, group_name)
    return {
        "xgb_clf":   os.path.join(base, "xgb_classifier.pkl"),
        "xgb_reg":   os.path.join(base, "xgb_regressor.pkl"),
        "lgb_clf":   os.path.join(base, "lgb_classifier.pkl"),
        "lgb_reg":   os.path.join(base, "lgb_regressor.pkl"),
        "lstm_clf":  os.path.join(base, "lstm_classifier.keras"),
        "lstm_reg":  os.path.join(base, "lstm_regressor.keras"),
        "scaler":    os.path.join(base, "scaler.pkl"),
        "le":        os.path.join(base, "label_encoder.pkl"),
        "config":    os.path.join(base, "config.json"),
        "payload":   os.path.join(base, "evaluation_payload.json"),
    }


def _models_exist(group_name=None):
    return all(os.path.exists(p) for p in _paths_for_group(group_name).values())


# ── Load models into memory ────────────────────────────────────────────────────
def get_group_models(group_name, force_reload=False):
    paths = _paths_for_group(group_name)
    payload_path = paths["payload"]
    payload_mtime = os.path.getmtime(payload_path) if os.path.exists(payload_path) else 0
    cached_mtime = _CACHE.get(f"{group_name}_mtime", 0)

    need_reload = (
        force_reload or
        group_name not in _CACHE or
        not isinstance(_CACHE[group_name], dict) or
        "xgb_clf" not in _CACHE[group_name] or
        payload_mtime != cached_mtime
    )

    if need_reload:
        if not _models_exist(group_name):
            raise RuntimeError(
                f"No trained models found for group {group_name}.\n"
                "Please run:  python train.py\n"
            )

        from tensorflow.keras.models import load_model

        print(f"Loading models for {group_name} from disk …")
        g_cache = {}
        g_cache["xgb_clf"]  = joblib.load(paths["xgb_clf"])
        g_cache["xgb_reg"]  = joblib.load(paths["xgb_reg"])
        g_cache["lgb_clf"]  = joblib.load(paths["lgb_clf"])
        g_cache["lgb_reg"]  = joblib.load(paths["lgb_reg"])

        try:
            from tensorflow.keras.initializers import GlorotUniform as GlorotUniformNew
            original_init = GlorotUniformNew.__init__
            def patched_init(self, seed=None, input_axes=None, output_axes=None, **kwargs):
                original_init(self, seed=seed)
            GlorotUniformNew.__init__ = patched_init

            g_cache["lstm_clf"] = load_model(paths["lstm_clf"], custom_objects={"GlorotUniform": GlorotUniformNew})
            g_cache["lstm_reg"] = load_model(paths["lstm_reg"], custom_objects={"GlorotUniform": GlorotUniformNew})
        except Exception as e:
            print(f"  ⚠  Warning: Could not load LSTM models: {e}")
            g_cache["lstm_clf"] = None
            g_cache["lstm_reg"] = None

        g_cache["scaler"]   = joblib.load(paths["scaler"])
        g_cache["le"]       = joblib.load(paths["le"])

        with open(paths["config"])  as f: g_cache["config"]  = json.load(f)
        with open(paths["payload"]) as f: g_cache["payload"] = json.load(f)

        try:
            train_dfs = [pd.read_csv(os.path.join(DATA_DIR, f"{c}.csv")) for c in g_cache["config"]["train_cities"]]
            train_df = pd.concat(train_dfs).dropna()
            y_train_cat = train_df["AQI_Category"].values
            y_train_enc = g_cache["le"].transform(y_train_cat)
            train_present_labels = np.unique(y_train_enc)
            local_to_global = {local: int(g) for local, g in enumerate(train_present_labels)}
            g_cache["xgb_local_to_global"] = local_to_global
        except Exception:
            g_cache["xgb_local_to_global"] = {i: i for i in range(6)}

        _CACHE[group_name] = g_cache
        _CACHE[f"{group_name}_mtime"] = payload_mtime

    return _CACHE[group_name]


def get_all_groups_payload(force_reload=False):
    path = os.path.join(MODELS_DIR, "all_groups_evaluation.json")
    if not os.path.exists(path):
        return {}
    current_mtime = os.path.getmtime(path)
    if force_reload or "all_groups" not in _CACHE or _CACHE.get("all_groups_mtime") != current_mtime:
        try:
            with open(path) as f:
                _CACHE["all_groups"] = json.load(f)
                _CACHE["all_groups_mtime"] = current_mtime
        except Exception as e:
            print(f"Error reading {path}: {e}")
            if "all_groups" not in _CACHE:
                _CACHE["all_groups"] = {}
    return _CACHE["all_groups"]


# ── Live Telemetry Buffer Storage (Per-City CSVs) ──────────────────────────────
def _get_buffer_path(city: str) -> str:
    return os.path.join(DATA_DIR, f"live_buffer_{city}.csv")


def append_to_live_buffer(city: str, date_str: str, readings: dict, station_count: int = 1):
    """Save or update today's telemetry record in the city's live buffer CSV."""
    path = _get_buffer_path(city)
    row_dict = {
        "Date": date_str,
        "PM2.5": round(float(readings.get("PM2.5", 0.0)), 2),
        "PM10":  round(float(readings.get("PM10", 0.0)), 2),
        "NO2":   round(float(readings.get("NO2", 0.0)), 2),
        "SO2":   round(float(readings.get("SO2", 0.0)), 2),
        "CO":    round(float(readings.get("CO", 0.0)), 2),
        "O3":    round(float(readings.get("O3", 0.0)), 2),
        "Station_Count": int(station_count),
    }

    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            # Remove existing row for today if already present, then append
            df = df[df["Date"] != date_str]
            df = pd.concat([df, pd.DataFrame([row_dict])], ignore_index=True)
            df.sort_values("Date").to_csv(path, index=False)
        except Exception:
            pd.DataFrame([row_dict]).to_csv(path, index=False)
    else:
        pd.DataFrame([row_dict]).to_csv(path, index=False)


def get_city_history(city: str, n_days: int = 6) -> list:
    """Return the last n_days records for the city from its live buffer."""
    path = _get_buffer_path(city)
    if not os.path.exists(path):
        return []
    try:
        df = pd.read_csv(path).sort_values("Date").reset_index(drop=True)
        if len(df) == 0:
            return []
        tail_df = df.tail(n_days)
        return tail_df.to_dict(orient="records")
    except Exception:
        return []


# ── Live CPCB Ingestion Helper ─────────────────────────────────────────────────
def fetch_cpcb_live_data(city: str):
    """
    Query data.gov.in for live telemetry of the specified city.
    Maps OZONE -> O3, averages station records, applies median imputation for missing sensors.
    """
    now = time.time()
    cache_entry = _API_CACHE.get(city)
    if cache_entry and (now - cache_entry["time"] < 900):  # 15-minute cache
        return cache_entry["data"]

    url = (
        "https://api.data.gov.in/resource/3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69"
        f"?api-key={DATA_GOV_API_KEY}&format=json&limit=50&filters%5Bcity%5D={city}"
    )

    resp = None
    for attempt in range(2):
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                break
        except requests.RequestException:
            if attempt == 0:
                time.sleep(1)
            else:
                raise

    if resp is None or resp.status_code != 200:
        status = resp.status_code if resp else "timeout"
        raise RuntimeError(f"CPCB API unavailable ({status}).")

    data = resp.json()
    records = data.get("records", [])
    if not records:
        raise RuntimeError(f"No active station records returned by CPCB for {city}.")

    # Parse and group by pollutant
    pollutant_vals = {}
    stations = set()
    last_update = None

    for r in records:
        pid = r.get("pollutant_id", "")
        if pid == "OZONE":
            pid = "O3"
        val_str = r.get("avg_value", "NA")
        station = r.get("station", "")
        if station:
            stations.add(station)
        if not last_update and r.get("last_update"):
            last_update = r.get("last_update")

        if val_str not in ("NA", "None", "", None):
            try:
                v = float(val_str)
                pollutant_vals.setdefault(pid, []).append(v)
            except ValueError:
                pass

    # Compute averages across reporting stations
    city_medians = CITY_MEDIANS.get(city, CITY_MEDIANS["Mumbai"])
    parsed_readings = {}

    for p in BASE_FEATURES:
        if p in pollutant_vals and len(pollutant_vals[p]) > 0:
            mean_val = float(np.mean(pollutant_vals[p]))
            # Handle CO unit adjustment if reported in ug/m3 or integer index > 5
            if p == "CO" and mean_val > 5.0:
                mean_val = round(mean_val / 50.0, 2)  # Scale to standard urban mg/m3
            parsed_readings[p] = round(mean_val, 2)
        else:
            # Impute missing pollutant with verified city historical median
            parsed_readings[p] = city_medians[p]

    today_str = datetime.now(timezone(timedelta(hours=5, minutes=30))).strftime("%Y-%m-%d")

    # Log to city's live buffer
    append_to_live_buffer(city, today_str, parsed_readings, len(stations))

    result = {
        "status": "success",
        "city": city,
        "date": today_str,
        "last_update": last_update or today_str,
        "readings": parsed_readings,
        "stations_count": len(stations),
        "stations": sorted(list(stations))[:5],
        "buffer_file": f"data/live_buffer_{city}.csv",
    }
    _API_CACHE[city] = {"time": now, "data": result}
    return result


# ── Daily Background Collector Thread (4:30 PM IST) ───────────────────────────
def _daily_collector_loop():
    """Runs continuously in the background, waking daily at 4:30 PM IST to fetch target cities."""
    while True:
        try:
            # Current IST time (UTC + 5:30)
            now_ist = datetime.now(timezone(timedelta(hours=5, minutes=30)))
            target_time = now_ist.replace(hour=16, minute=30, second=0, microsecond=0)
            if now_ist >= target_time:
                target_time += timedelta(days=1)

            seconds_to_wait = (target_time - now_ist).total_seconds()
            time.sleep(min(seconds_to_wait, 3600))  # sleep in slices

            # Check if within 5-min window of 4:30 PM IST
            curr_ist = datetime.now(timezone(timedelta(hours=5, minutes=30)))
            if curr_ist.hour == 16 and 25 <= curr_ist.minute <= 35:
                print(f"[{curr_ist.strftime('%Y-%m-%d %H:%M:%S IST')}] Running scheduled daily CPCB telemetry fetch...")
                for target_city in ("Mumbai", "Lucknow", "Hyderabad"):
                    try:
                        fetch_cpcb_live_data(target_city)
                    except Exception as err:
                        print(f"  ⚠ Background fetch failed for {target_city}: {err}")
                time.sleep(600)  # avoid duplicate trigger within the same hour
        except Exception as e:
            print(f"Daily collector loop error: {e}")
            time.sleep(60)


threading.Thread(target=_daily_collector_loop, daemon=True).start()


# ── Dataset Explorer Payload ───────────────────────────────────────────────────
def get_dataset_payload():
    if "dataset_payload" in _CACHE:
        return _CACHE["dataset_payload"]

    city_frames = {}
    for city in CITY_ORDER:
        path = os.path.join(DATA_DIR, f"{city}.csv")
        df = pd.read_csv(path)
        df["Date"] = pd.to_datetime(df["Date"])
        city_frames[city] = df.sort_values("Date").reset_index(drop=True)

    feature_means = {
        city: {feat: round(float(df[feat].mean()), 2) for feat in BASE_FEATURES}
        for city, df in city_frames.items()
    }
    feature_max = {
        feat: max(feature_means[city][feat] for city in CITY_ORDER) or 1.0
        for feat in BASE_FEATURES
    }

    monthly_by_city = {}
    months = set()
    for city, df in city_frames.items():
        monthly = (
            df.assign(Month=df["Date"].dt.to_period("M").astype(str))
              .groupby("Month", sort=True)["AQI"]
              .mean()
        )
        monthly_by_city[city] = {month: round(float(v), 2) for month, v in monthly.items()}
        months.update(monthly_by_city[city].keys())

    monthly_rows = [{"month": m, **{c: monthly_by_city[c].get(m) for c in CITY_ORDER}} for m in sorted(months)]
    pm25_dates = city_frames[CITY_ORDER[0]]["Date"].dt.strftime("%Y-%m-%d").tolist()

    pm25_trend = {
        city: city_frames[city]["PM2.5"].rolling(window=PM25_TREND_WINDOW, min_periods=1).mean().round(2).tolist()
        for city in CITY_ORDER
    }
    pm25_pm10_scatter = {
        city: [{"x": round(float(r["PM2.5"]), 2), "y": round(float(r["PM10"]), 2)}
               for _, r in city_frames[city].iloc[::SCATTER_STRIDE].iterrows()]
        for city in CITY_ORDER
    }
    pm25_pm10_corr = {
        city: round(float(city_frames[city]["PM2.5"].corr(city_frames[city]["PM10"])), 3)
        for city in CITY_ORDER
    }

    cities = {}
    category_mix = {}
    category_share = {}
    feature_normalized = {}
    feature_contribution_share = {}

    for city, df in city_frames.items():
        counts = df["AQI_Category"].value_counts()
        aligned_counts = {band: int(counts.get(band, 0)) for band in AQI_BANDS}
        peak_row = df.loc[df["AQI"].idxmax()]
        clean_days = int((df["AQI"] <= 100).sum())
        unhealthy_days = int((df["AQI"] > 200).sum())
        severe_days = int((df["AQI"] > 300).sum())

        cities[city] = {
            "samples": int(len(df)),
            "start_date": df["Date"].min().strftime("%Y-%m-%d"),
            "end_date": df["Date"].max().strftime("%Y-%m-%d"),
            "avg_aqi": round(float(df["AQI"].mean()), 1),
            "median_aqi": round(float(df["AQI"].median()), 1),
            "aqi_volatility": round(float(df["AQI"].std()), 1),
            "peak_aqi": round(float(peak_row["AQI"]), 1),
            "peak_date": peak_row["Date"].strftime("%Y-%m-%d"),
            "clean_days": clean_days,
            "clean_days_pct": round(clean_days / len(df) * 100, 1),
            "unhealthy_days": unhealthy_days,
            "unhealthy_days_pct": round(unhealthy_days / len(df) * 100, 1),
            "severe_days": severe_days,
            "dominant_category": max(aligned_counts, key=aligned_counts.get),
            "avg_category": _aqi_to_category(float(df["AQI"].mean())),
        }
        category_mix[city] = aligned_counts
        category_share[city] = {band: round(aligned_counts[band] / len(df) * 100, 1) for band in AQI_BANDS}
        feature_normalized[city] = {feat: round(feature_means[city][feat] / feature_max[feat], 3) for feat in BASE_FEATURES}
        norm_total = sum(feature_normalized[city].values()) or 1.0
        feature_contribution_share[city] = {feat: round(feature_normalized[city][feat] / norm_total * 100, 1) for feat in BASE_FEATURES}

    _CACHE["dataset_payload"] = {
        "city_order": list(CITY_ORDER),
        "feature_order": list(BASE_FEATURES),
        "category_order": list(AQI_BANDS),
        "cities": cities,
        "monthly_aqi": monthly_rows,
        "category_mix": category_mix,
        "category_share": category_share,
        "feature_means": feature_means,
        "feature_normalized": feature_normalized,
        "feature_contribution_share": feature_contribution_share,
        "pm25_dates": pm25_dates,
        "pm25_trend_window": PM25_TREND_WINDOW,
        "pm25_trend": pm25_trend,
        "pm25_pm10_scatter": pm25_pm10_scatter,
        "pm25_pm10_corr": pm25_pm10_corr,
    }
    return _CACHE["dataset_payload"]


def _aqi_to_category(v: float) -> str:
    if v <= 50:   return "Good"
    if v <= 100:  return "Satisfactory"
    if v <= 200:  return "Moderate"
    if v <= 300:  return "Poor"
    if v <= 400:  return "Very Poor"
    return "Severe"


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    default_group = "Group1_Pune_Chennai_Mumbai"
    cache = get_group_models(default_group)
    return render_template(
        "index.html",
        payload_json=json.dumps(cache["payload"]),
        dataset_json=json.dumps(get_dataset_payload()),
        all_groups_json=json.dumps(get_all_groups_payload()),
    )


@app.route("/api/fetch_live", methods=["GET"])
def api_fetch_live():
    """
    Endpoint to fetch real-time air quality telemetry from data.gov.in.
    Usage: /api/fetch_live?city=Mumbai
    """
    city = request.args.get("city", "Mumbai").strip()
    try:
        telemetry = fetch_cpcb_live_data(city)
        return jsonify(telemetry)
    except Exception as e:
        print(f"  [CPCB Notice] {city}: {e}")
        # Check if we already have today's buffered telemetry in live_buffer_<city>.csv
        history = get_city_history(city, n_days=1)
        if history and len(history) > 0:
            latest = history[-1]
            return jsonify({
                "status": "buffered",
                "message": f"CPCB server busy; loaded verified telemetry for {city} from local buffer.",
                "city": city,
                "readings": {
                    "PM2.5": latest.get("PM2.5", 38.26),
                    "PM10":  latest.get("PM10", 103.92),
                    "NO2":   latest.get("NO2", 23.44),
                    "SO2":   latest.get("SO2", 2.72),
                    "CO":    latest.get("CO", 0.58),
                    "O3":    latest.get("O3", 25.11),
                },
                "stations_count": latest.get("Station_Count", 1),
                "last_update": latest.get("Date", "Today"),
                "buffer_file": f"data/live_buffer_{city}.csv",
                "fallback": "live_buffer",
            }), 200

        # Otherwise fall back to verified city historical median
        city_medians = CITY_MEDIANS.get(city, CITY_MEDIANS["Mumbai"])
        return jsonify({
            "status": "warning",
            "message": f"CPCB National Portal busy; loaded verified historical baseline for {city}.",
            "city": city,
            "readings": city_medians,
            "stations_count": 0,
            "fallback": "historical_median",
        }), 200


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    POST JSON: { pm25, pm10, no2, so2, co, o3, city, group }
    Returns predictions from XGBoost, LightGBM, and BiLSTM (if 6-day history available).
    """
    body  = request.get_json(force=True)
    city  = body.get("city", "Mumbai")
    group_name = body.get("group") or CITY_TO_GROUP.get(city, "Group1_Pune_Chennai_Mumbai")
    cache = get_group_models(group_name)

    pm25 = float(body.get("pm25", 0))
    pm10 = float(body.get("pm10", 0))
    no2  = float(body.get("no2",  0))
    so2  = float(body.get("so2",  0))
    co   = float(body.get("co",   0))
    o3   = float(body.get("o3",   0))
    pm_ratio = pm25 / (pm10 + 1e-6)

    # Clean 7 features
    feat_map = {
        "PM2.5": pm25, "PM10": pm10, "NO2": no2, "SO2": so2, "CO": co, "O3": o3,
        "PM_ratio": pm_ratio,
    }
    cfg = cache["config"]
    model_features = cfg.get("features", ENG_FEATURES)
    raw = np.array([feat_map.get(f, 0.0) for f in model_features], dtype=np.float32).reshape(1, -1)

    X_s = cache["scaler"].transform(raw).astype(np.float32)
    le  = cache["le"]
    n_classes = len(le.classes_)

    # ── XGBoost ────────────────────────────────────────────────────────────
    xgb_enc_local = cache["xgb_clf"].predict(X_s)[0]
    xgb_enc  = cache["xgb_local_to_global"].get(int(xgb_enc_local), 0)
    xgb_cat  = le.inverse_transform([xgb_enc])[0]

    xgb_proba_local = cache["xgb_clf"].predict_proba(X_s)[0]
    xgb_proba = np.zeros(n_classes, dtype=np.float32)
    for local_idx, global_idx in cache["xgb_local_to_global"].items():
        if local_idx < len(xgb_proba_local):
            xgb_proba[global_idx] = xgb_proba_local[local_idx]

    xgb_aqi  = float(cache["xgb_reg"].predict(X_s)[0])

    # ── LightGBM ───────────────────────────────────────────────────────────
    lgb_enc  = cache["lgb_clf"].predict(X_s)[0]
    lgb_cat  = le.inverse_transform([lgb_enc])[0]

    lgb_proba_local = cache["lgb_clf"].predict_proba(X_s)[0]
    lgb_proba = np.zeros(n_classes, dtype=np.float32)
    for local_idx, global_idx in enumerate(cache["lgb_clf"].classes_.astype(int)):
        lgb_proba[global_idx] = lgb_proba_local[local_idx]

    lgb_aqi  = float(cache["lgb_reg"].predict(X_s)[0])

    # ── BiLSTM: Check for genuine 6-day history from live buffer ───────────
    history_rows = get_city_history(city, n_days=6)
    use_lstm = False
    lstm_cat = "Buffer building"
    lstm_aqi = None
    lstm_proba_v = np.zeros(n_classes, dtype=np.float32)

    if len(history_rows) >= 6 and cache["lstm_clf"] is not None and cache["lstm_reg"] is not None:
        # Build genuine 7-day contiguous matrix: [t-6, t-5, ..., t-1, t_today]
        hist_feats = []
        for r in history_rows[-6:]:
            p25 = float(r.get("PM2.5", pm25))
            p10 = float(r.get("PM10", pm10))
            hist_feats.append([
                p25, p10,
                float(r.get("NO2", no2)),
                float(r.get("SO2", so2)),
                float(r.get("CO", co)),
                float(r.get("O3", o3)),
                p25 / (p10 + 1e-6),
            ])
        hist_feats.append([pm25, pm10, no2, so2, co, o3, pm_ratio])

        X_seq_raw = np.array(hist_feats, dtype=np.float32)
        X_seq_s   = cache["scaler"].transform(X_seq_raw).astype(np.float32)[np.newaxis, :, :]  # (1, 7, 7)

        lstm_proba_v = cache["lstm_clf"].predict(X_seq_s, verbose=0)[0]
        lstm_enc     = int(np.argmax(lstm_proba_v))
        lstm_cat     = le.inverse_transform([lstm_enc])[0]
        lstm_aqi     = round(float(cache["lstm_reg"].predict(X_seq_s, verbose=0)[0][0]), 2)
        use_lstm     = True

    # ── Ensemble Calculation ───────────────────────────────────────────────
    if use_lstm:
        vote_proba = (xgb_proba + lgb_proba + lstm_proba_v) / 3.0
        vote_aqi   = round((xgb_aqi + lgb_aqi + lstm_aqi) / 3.0, 2)
        ensemble_mode = "Full 3-Model Ensemble (BiLSTM + XGBoost + LightGBM)"
    else:
        vote_proba = (xgb_proba + lgb_proba) / 2.0
        vote_aqi   = round((xgb_aqi + lgb_aqi) / 2.0, 2)
        ensemble_mode = f"Tree Ensemble (XGBoost + LightGBM) — Live Buffer: {len(history_rows)}/6 days"

    vote_enc = int(np.argmax(vote_proba))
    vote_cat = le.inverse_transform([vote_enc])[0]
    proba_dict = {cls: round(float(p), 4) for cls, p in zip(le.classes_.tolist(), vote_proba)}

    # ── Tomorrow Forecast (Autoregressive persistence) ─────────────────────
    alpha = 0.8
    mean_unscaled = cache["scaler"].quantiles_.mean(axis=0).reshape(1, -1)
    tomorrow_raw  = alpha * raw + (1.0 - alpha) * mean_unscaled
    X_s_tomorrow  = cache["scaler"].transform(tomorrow_raw).astype(np.float32)

    xgb_fc = float(cache["xgb_reg"].predict(X_s_tomorrow)[0])
    lgb_fc = float(cache["lgb_reg"].predict(X_s_tomorrow)[0])
    fc_aqi = round((xgb_fc + lgb_fc) / 2.0, 2)
    fc_cat = _aqi_to_category(fc_aqi)

    # ── Local TreeSHAP Explanations (Clean 7 features) ─────────────────────
    import shap
    shap_explain = {}
    try:
        expl_xgb = shap.TreeExplainer(cache["xgb_reg"])
        sv_xgb   = expl_xgb.shap_values(X_s)
        shap_xgb = sv_xgb[0] if len(sv_xgb.shape) == 2 else sv_xgb

        expl_lgb = shap.TreeExplainer(cache["lgb_reg"])
        sv_lgb   = expl_lgb.shap_values(X_s)
        if isinstance(sv_lgb, list): sv_lgb = sv_lgb[0]
        shap_lgb = sv_lgb[0] if len(sv_lgb.shape) == 2 else sv_lgb

        mean_shap = (shap_xgb + shap_lgb) / 2.0
        base_val  = float(expl_lgb.expected_value[0] if isinstance(expl_lgb.expected_value, (list, np.ndarray)) else expl_lgb.expected_value)

        shap_explain = {
            "base_value": round(base_val, 2),
            "contributions": dict(zip(model_features, [round(float(x), 3) for x in mean_shap])),
        }
    except Exception:
        shap_explain = {"base_value": 0.0, "contributions": {f: 0.0 for f in model_features}}

    return jsonify({
        "xgb_cat":       xgb_cat,
        "xgb_aqi":       round(xgb_aqi, 2),
        "lgb_cat":       lgb_cat,
        "lgb_aqi":       round(lgb_aqi, 2),
        "lstm_cat":      lstm_cat,
        "lstm_aqi":      lstm_aqi,
        "vote_cat":      vote_cat,
        "vote_aqi":      vote_aqi,
        "class_probas":  proba_dict,
        "forecast_aqi":  fc_aqi,
        "forecast_cat":  fc_cat,
        "shap":          shap_explain,
        "ensemble_mode": ensemble_mode,
        "history_days":  len(history_rows),
    })


@app.route("/api/live_metrics")
def live_metrics():
    group = request.args.get("group", "Group1_Pune_Chennai_Mumbai")
    all_groups_data = get_all_groups_payload(force_reload=True)
    try:
        g_cache = get_group_models(group, force_reload=True)
        active_payload = g_cache.get("payload", all_groups_data.get(group, {}))
    except Exception:
        active_payload = all_groups_data.get(group, {})

    mtime = 0
    all_path = os.path.join(MODELS_DIR, "all_groups_evaluation.json")
    if os.path.exists(all_path):
        mtime = os.path.getmtime(all_path)

    return jsonify({
        "all_groups": all_groups_data,
        "current_group": group,
        "payload": active_payload,
        "timestamp": mtime,
        "features": active_payload.get("feature_names", FEATURES),
        "results": active_payload.get("results", {}),
    })


@app.route("/api/all_groups")
def all_groups():
    return jsonify(get_all_groups_payload(force_reload=True))


@app.route("/api/status")
def status():
    return jsonify({"models_ready": _models_exist()})


@app.route("/api/model_info")
def model_info():
    group_name = request.args.get("group") or "Group1_Pune_Chennai_Mumbai"
    paths = _paths_for_group(group_name)
    info = {}
    for key, full_path in paths.items():
        exists = os.path.exists(full_path)
        rel_path = os.path.relpath(full_path, MODELS_DIR)
        size_kb = round(os.path.getsize(full_path) / 1024, 1) if exists else None
        info[key] = {
            "file": rel_path,
            "size_kb": size_kb,
            "missing": not exists,
        }
    return jsonify(info)


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  AQI Cross-City Generalization & Live Telemetry Server")
    print("  Port:", PORT)
    print("  Models: XGBoost · LightGBM · BiLSTM (Keras)")
    print("=" * 60 + "\n")

    app.run(host="0.0.0.0", port=PORT, debug=False)
