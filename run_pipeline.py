"""
Standalone AQI Pipeline — Fetch May–Jun 2026 for Karachi/Lahore/Islamabad,
train per-city models, run EDA. No Hopsworks required.
"""

import os, json, warnings, re
import pandas as pd
import numpy as np
import requests
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import List

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
BDATA  = SCRIPT_DIR / "backend" / "data"
BMOD   = SCRIPT_DIR / "backend" / "models"
BCFG   = SCRIPT_DIR / "backend" / "config"
EDADIR = SCRIPT_DIR / "eda_output"
for p in [BDATA, BMOD, BCFG, EDADIR]: p.mkdir(parents=True, exist_ok=True)

# ── City config ───────────────────────────────────────────────────────────────
CITIES = {
    "karachi":   {"lat": 24.8608, "lon": 67.0104, "state": "Sindh"},
    "lahore":    {"lat": 31.5204, "lon": 74.3587, "state": "Punjab"},
    "islamabad": {"lat": 33.6844, "lon": 73.0479, "state": "ICT"},
}

HORIZON_MAX_FEATURES = {24: 10, 48: 15, 72: 8}

# ── 1. Fetch ──────────────────────────────────────────────────────────────────

def fetch_city_data(city: str, lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    print(f"  [FETCH] {city.title()}  {start} → {end}")

    # air quality
    aq = requests.get("https://air-quality-api.open-meteo.com/v1/air-quality", params={
        "latitude": lat, "longitude": lon,
        "start_date": start, "end_date": end,
        "hourly": ["pm10","pm2_5","carbon_monoxide","nitrogen_dioxide","sulphur_dioxide","ozone"],
        "timezone": "UTC"
    }, timeout=60).json()

    times = pd.to_datetime(aq["hourly"]["time"])
    aq_df = pd.DataFrame({
        "timestamp": times,
        "pm10_raw": aq["hourly"]["pm10"],
        "pm25_raw": aq["hourly"]["pm2_5"],
        "co_raw":   aq["hourly"]["carbon_monoxide"],
        "no2_raw":  aq["hourly"]["nitrogen_dioxide"],
        "so2_raw":  aq["hourly"]["sulphur_dioxide"],
        "o3_raw":   aq["hourly"]["ozone"],
    })

    # weather
    wx = requests.get("https://archive-api.open-meteo.com/v1/archive", params={
        "latitude": lat, "longitude": lon,
        "start_date": start, "end_date": end,
        "hourly": ["temperature_2m","relative_humidity_2m","rain",
                   "wind_speed_10m","wind_direction_10m","surface_pressure"],
        "timezone": "UTC"
    }, timeout=60).json()

    wx_df = pd.DataFrame({
        "timestamp":     pd.to_datetime(wx["hourly"]["time"]),
        "temperature":   wx["hourly"]["temperature_2m"],
        "humidity":      wx["hourly"]["relative_humidity_2m"],
        "precipitation": wx["hourly"]["rain"],
        "wind_speed":    [v * 3.6 if v is not None else None for v in wx["hourly"]["wind_speed_10m"]],
        "wind_direction":wx["hourly"]["wind_direction_10m"],
        "pressure":      wx["hourly"]["surface_pressure"],
    })

    df = pd.merge(aq_df, wx_df, on="timestamp", how="inner")
    print(f"         raw rows: {len(df)}")
    return df


def calc_aqi(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pm25"] = df["pm25_raw"].rolling(24, min_periods=1).mean()
    df["pm10"] = df["pm10_raw"].rolling(24, min_periods=1).mean()
    df["co"]   = (df["co_raw"] / (28.01/24.45) / 1000).rolling(8, min_periods=1).mean()
    df["no2"]  = df["no2_raw"] / (46.01/24.45)
    df["so2"]  = df["so2_raw"] / (64.07/24.45)
    df["o3"]   = df["o3_raw"]  / (48.00/24.45)
    df["o3_8h"]= df["o3"].rolling(8, min_periods=1).mean()

    bp = {
        "pm25": [(0,12,0,50),(12.1,35.4,51,100),(35.5,55.4,101,150),
                 (55.5,150.4,151,200),(150.5,250.4,201,300),(250.5,500.4,301,500)],
        "pm10": [(0,54,0,50),(55,154,51,100),(155,254,101,150),
                 (255,354,151,200),(355,504,201,300),(505,604,301,500)],
        "co":   [(0,4.4,0,50),(4.5,9.4,51,100),(9.5,12.4,101,150),(12.5,15.4,151,200),(15.5,30.4,201,300)],
        "so2":  [(0,35,0,50),(36,75,51,100),(76,185,101,150),(186,304,151,200),(305,604,201,300)],
        "o3":   [(0,54,0,50),(55,70,51,100),(71,85,101,150),(86,105,151,200),(106,200,201,300)],
        "no2":  [(0,53,0,50),(54,100,51,100),(101,360,101,150),(361,649,151,200),(650,1249,201,300)],
    }
    def _aqi(c, breaks):
        if pd.isna(c): return np.nan
        for lo,hi,ilo,ihi in breaks:
            if lo <= c <= hi:
                return round(((ihi-ilo)/(hi-lo))*(c-lo)+ilo, 2)
        return np.nan

    for pol, col in [("pm25","pm25"),("pm10","pm10"),("co","co"),
                     ("so2","so2"),("o3","o3_8h"),("no2","no2")]:
        df[f"aqi_{pol}"] = df[col].apply(lambda x: _aqi(x, bp[pol]))

    aqi_cols = ["aqi_pm25","aqi_pm10","aqi_co","aqi_so2","aqi_o3","aqi_no2"]
    df["aqi"] = df[aqi_cols].max(axis=1)
    return df


def build_aqi_rows(df: pd.DataFrame, city: str, cfg: dict) -> pd.DataFrame:
    df = df.copy()
    df["city"]      = city
    df["state"]     = cfg["state"]
    df["country"]   = "Pakistan"
    df["source"]    = "open-meteo"
    df["visibility"]= 10.0
    df["cloud_cover"]= 30.0
    df["latitude"]  = cfg["lat"]
    df["longitude"] = cfg["lon"]
    df["is_deleted"]= 0
    df["created_at"]= df["timestamp"].astype(str)
    df["id"]        = range(1, len(df)+1)

    keep = ["id","city","state","country","timestamp","source",
            "aqi","pm25","pm10","no2","so2","co","o3",
            "temperature","humidity","pressure","wind_speed","wind_direction",
            "visibility","cloud_cover","precipitation","latitude","longitude",
            "created_at","is_deleted"]
    for c in keep:
        if c not in df.columns: df[c] = np.nan
    return df[keep]


def update_aqi_csv(new_df: pd.DataFrame) -> pd.DataFrame:
    path = BDATA / "aqi_data.csv"
    new_df = new_df.copy()
    new_df["timestamp"] = pd.to_datetime(new_df["timestamp"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
    new_df["city"]      = new_df["city"].str.lower().str.strip()

    if path.exists():
        existing = pd.read_csv(path)
        existing["timestamp"] = pd.to_datetime(
            existing["timestamp"], format="mixed", errors="coerce"
        ).dt.strftime("%Y-%m-%dT%H:%M:%S")
        existing["city"] = existing["city"].str.lower().str.strip()
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["timestamp","city"], keep="last")
        combined = combined.sort_values("timestamp").reset_index(drop=True)
    else:
        combined = new_df

    combined.to_csv(path, index=False)
    print(f"  [CSV] aqi_data.csv → {len(combined)} rows  "
          f"({combined['timestamp'].min()} → {combined['timestamp'].max()})")
    return combined


# ── 2. Feature engineering ────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    d = df.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    d = d.sort_values("timestamp").set_index("timestamp")

    core = ["aqi","pm25","pm10","no2","so2","co","o3",
            "temperature","humidity","wind_speed","wind_direction","pressure","precipitation"]
    d = d[[c for c in core if c in d.columns]].copy()

    d = d.ffill(limit=3).bfill(limit=3).interpolate(method="time", limit=6)
    for col in d.columns:
        q1, q3 = d[col].quantile(0.01), d[col].quantile(0.99)
        d[col] = d[col].clip(lower=q1, upper=q3)

    # time
    d["hour"]       = d.index.hour
    d["day"]        = d.index.day
    d["month"]      = d.index.month
    d["weekday"]    = d.index.dayofweek
    d["is_weekend"] = (d.index.dayofweek >= 5).astype(int)
    d["season"]     = d["month"] % 12 // 3 + 1
    d["hour_sin"]   = np.sin(2*np.pi*d["hour"]/24)
    d["hour_cos"]   = np.cos(2*np.pi*d["hour"]/24)

    # interactions
    if "temperature" in d.columns and "humidity" in d.columns:
        d["temp_humidity"] = d["temperature"] * d["humidity"] / 100
    if "wind_speed" in d.columns and "wind_direction" in d.columns:
        d["wind_u"] = d["wind_speed"] * np.cos(np.radians(d["wind_direction"]))
        d["wind_v"] = d["wind_speed"] * np.sin(np.radians(d["wind_direction"]))
    if "pm25" in d.columns and "pm10" in d.columns:
        d["pm_ratio"] = d["pm25"] / (d["pm10"] + 1e-6)

    # lags + rolling
    for col in ["aqi","pm25","pm10","no2","so2","co","o3","temperature","humidity","wind_speed"]:
        if col not in d.columns: continue
        for lag in [horizon, horizon+12, horizon+24, horizon+48]:
            d[f"{col}_lag_{lag}h"] = d[col].shift(lag)
        for win in [24, 48]:
            sh = d[col].shift(horizon)
            d[f"{col}_roll_mean_{win}h_lag{horizon}h"] = sh.rolling(win).mean()
            d[f"{col}_roll_std_{win}h_lag{horizon}h"]  = sh.rolling(win).std()

    for hrs in [24, 48]:
        d[f"aqi_rate_{hrs}h_lag{horizon}h"] = d["aqi"].shift(horizon).diff(hrs)

    # targets
    for h in [24, 48, 72]:
        d[f"aqi_{h}h"] = d["aqi"].shift(-h)

    d = d.dropna(subset=[f"aqi_{horizon}h"])

    windows = re.findall(r"lag_(\d+)h", " ".join(d.columns))
    max_lag = max([int(w) for w in windows], default=0)
    if 0 < max_lag < len(d):
        d = d.iloc[max_lag:]

    d = d.ffill().bfill().dropna(axis=1, how="all")

    num = d.select_dtypes(include=np.number).columns
    low_var = [c for c in num if d[c].var() < 1e-6]
    d = d.drop(columns=low_var)

    # drop highly correlated
    feat_cols = [c for c in d.columns if c not in {"aqi","aqi_24h","aqi_48h","aqi_72h"}]
    corr_mat  = d[feat_cols].corr().abs()
    upper     = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool))
    to_drop   = set()
    tgt       = f"aqi_{horizon}h"
    for col in upper.columns:
        for h in upper[col][upper[col] > 0.85].index:
            if col in to_drop: continue
            c1 = abs(d[col].corr(d[tgt])) if tgt in d else 0
            c2 = abs(d[h].corr(d[tgt]))   if tgt in d else 0
            to_drop.add(h if c1 >= c2 else col)
    d = d.drop(columns=list(to_drop), errors="ignore")
    return d


# ── 3. Feature selection ──────────────────────────────────────────────────────

def select_features(df: pd.DataFrame, target: str, max_k: int) -> List[str]:
    excl = {"aqi","aqi_24h","aqi_48h","aqi_72h","event_time","unique_id"}
    feat_cols = [c for c in df.columns if c not in excl]
    X = df[feat_cols].fillna(df[feat_cols].median())
    y = df[target]
    valid = ~y.isna(); X, y = X[valid], y[valid]

    if len(X) < 100:
        return feat_cols[:max_k]

    tscv = TimeSeriesSplit(n_splits=3)
    mi_s, rf_s = {}, {}
    for tr, _ in tscv.split(X):
        Xtr, ytr = X.iloc[tr], y.iloc[tr]
        sel = SelectKBest(mutual_info_regression, k=min(max_k, Xtr.shape[1]))
        sel.fit(Xtr, ytr)
        for f, s in zip(feat_cols, sel.scores_):
            mi_s[f] = mi_s.get(f, 0) + s
        rf = RandomForestRegressor(50, random_state=42, n_jobs=-1)
        rf.fit(Xtr, ytr)
        for f, s in zip(feat_cols, rf.feature_importances_):
            rf_s[f] = rf_s.get(f, 0) + s

    mi_max = max(mi_s.values()) or 1
    rf_max = max(rf_s.values()) or 1
    combined = {f: (mi_s.get(f,0)/mi_max + rf_s.get(f,0)/rf_max)/2 for f in feat_cols}
    selected = sorted(combined, key=combined.get, reverse=True)[:max_k]
    return selected


# ── 4. Train per-city per-horizon model ───────────────────────────────────────

def train_city_model(df: pd.DataFrame, city: str, horizon: int) -> dict:
    print(f"    [TRAIN] {city.title()} {horizon}h …", end=" ")
    target = f"aqi_{horizon}h"
    max_k  = HORIZON_MAX_FEATURES[horizon]

    selected = select_features(df, target, max_k)
    excl = {"aqi","aqi_24h","aqi_48h","aqi_72h"}
    feat_cols = [c for c in selected if c not in excl]

    data = df[feat_cols + [target]].dropna().copy()
    X, y = data[feat_cols], data[target]

    if len(X) < 100:
        print(f"not enough data ({len(X)} rows) — skipping")
        return {}

    train_size = min(8*7*24, int(len(X)*0.7))
    test_size  = min(4*7*24, int(len(X)*0.2))
    step_size  = 2*24

    windows, start = [], 0
    while start + train_size + test_size <= len(X):
        windows.append((start, start+train_size, start+train_size, start+train_size+test_size))
        start += step_size
    if not windows:
        windows = [(0, int(len(X)*0.8), int(len(X)*0.8), len(X))]

    metrics = []
    for ts, te, vs, ve in windows:
        m = ExtraTreesRegressor(200, max_depth=9, min_samples_split=10,
                                min_samples_leaf=10, max_features=0.6,
                                random_state=42, n_jobs=-1)
        m.fit(X.iloc[ts:te], y.iloc[ts:te])
        p = m.predict(X.iloc[vs:ve])
        metrics.append({"r2": float(r2_score(y.iloc[vs:ve], p)),
                        "rmse": float(np.sqrt(mean_squared_error(y.iloc[vs:ve], p))),
                        "mae":  float(mean_absolute_error(y.iloc[vs:ve], p))})

    avg_r2   = float(np.mean([m["r2"]   for m in metrics]))
    avg_rmse = float(np.mean([m["rmse"] for m in metrics]))
    avg_mae  = float(np.mean([m["mae"]  for m in metrics]))
    print(f"R²={avg_r2:.3f}  RMSE={avg_rmse:.1f}  MAE={avg_mae:.1f}")

    # final model on all data
    final = ExtraTreesRegressor(251, max_depth=9, min_samples_split=10,
                                min_samples_leaf=20, max_features=0.6449,
                                random_state=42, n_jobs=-1)
    final.fit(X, y)

    # save model
    model_path = BMOD / f"model_{city}_{horizon}h.pkl"
    joblib.dump(final, model_path, protocol=4)

    # save selected features
    feat_path = BCFG / f"selected_features_{city}_{horizon}h.json"
    with open(feat_path, "w") as f:
        json.dump(feat_cols, f, indent=2)

    # save horizon data (latest 72 rows for inference)
    horizon_path = BDATA / f"horizon_{city}_{horizon}h_data.csv"
    df[feat_cols].tail(72).to_csv(horizon_path, index=False)

    return {
        "model_type": "ExtraTrees",
        "model_file": f"model_{city}_{horizon}h.pkl",
        "model_format": "pkl",
        "avg_r2": avg_r2, "avg_rmse": avg_rmse, "avg_mae": avg_mae,
        "features_count": len(feat_cols),
        "updated_at": datetime.utcnow().isoformat(),
        "description": f"ExtraTrees {city} {horizon}h  R²={avg_r2:.3f}"
    }


# ── 5. EDA ────────────────────────────────────────────────────────────────────

def run_eda(full_df: pd.DataFrame):
    print("\n[EDA] Generating plots …")
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

    full_df = full_df.copy()
    full_df["timestamp"] = pd.to_datetime(full_df["timestamp"])
    full_df = full_df.sort_values("timestamp")

    mask = (full_df["timestamp"] >= "2026-05-01") & (full_df["timestamp"] <= "2026-06-30")
    df26 = full_df[mask].copy()
    if df26.empty:
        df26 = full_df.copy()

    cities = df26["city"].str.lower().unique()
    city_colors = {"karachi": "#2980b9", "lahore": "#e74c3c", "islamabad": "#27ae60"}

    # ── 1. AQI time series per city ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 4))
    for city in cities:
        cd = df26[df26["city"]==city].copy()
        color = city_colors.get(city, "#888")
        ax.plot(cd["timestamp"], cd["aqi"], lw=0.9, color=color, alpha=0.85, label=city.title())
    ax.set_title("AQI Time Series — May–Jun 2026", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date"); ax.set_ylabel("AQI")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax.legend(); plt.xticks(rotation=30); plt.tight_layout()
    fig.savefig(EDADIR / "01_aqi_timeseries.png", dpi=150); plt.close()

    # ── 2. AQI distribution per city ────────────────────────────────────────
    fig, axes = plt.subplots(1, len(cities), figsize=(5*len(cities), 4), sharey=True)
    if len(cities) == 1: axes = [axes]
    for ax, city in zip(axes, cities):
        cd = df26[df26["city"]==city]["aqi"].dropna()
        ax.hist(cd, bins=30, color=city_colors.get(city,"#888"), edgecolor="white", alpha=0.85)
        ax.set_title(city.title()); ax.set_xlabel("AQI")
        for lim, col in [(50,"#2ecc71"),(100,"#f1c40f"),(150,"#e67e22"),(200,"#e74c3c")]:
            ax.axvline(lim, color=col, lw=1.2, linestyle="--", alpha=0.7)
    axes[0].set_ylabel("Count")
    fig.suptitle("AQI Distribution — May–Jun 2026", fontsize=13, fontweight="bold")
    plt.tight_layout(); fig.savefig(EDADIR / "02_aqi_distribution.png", dpi=150); plt.close()

    # ── 3. Hourly AQI pattern per city ──────────────────────────────────────
    df26["hour"] = df26["timestamp"].dt.hour
    fig, ax = plt.subplots(figsize=(11, 4))
    for city in cities:
        cd = df26[df26["city"]==city]
        ha = cd.groupby("hour")["aqi"].mean()
        ax.plot(ha.index, ha.values, marker="o", ms=5,
                color=city_colors.get(city,"#888"), label=city.title())
    ax.set_title("Average AQI by Hour — May–Jun 2026")
    ax.set_xlabel("Hour (UTC)"); ax.set_ylabel("Average AQI")
    ax.set_xticks(range(0, 24, 2)); ax.legend()
    plt.tight_layout(); fig.savefig(EDADIR / "03_hourly_pattern.png", dpi=150); plt.close()

    # ── 4. Pollutant trends (Karachi only – freshest data) ───────────────────
    kar = df26[df26["city"]=="karachi"]
    pollutants = [p for p in ["pm25","pm10","no2","so2","co","o3"] if p in kar.columns]
    if pollutants:
        fig, axes = plt.subplots(len(pollutants), 1, figsize=(14, 3*len(pollutants)), sharex=True)
        if len(pollutants)==1: axes=[axes]
        colors = ["#e74c3c","#e67e22","#3498db","#1abc9c","#9b59b6","#f39c12"]
        for i, (pol, ax) in enumerate(zip(pollutants, axes)):
            ax.plot(kar["timestamp"], kar[pol], lw=0.7, color=colors[i%len(colors)])
            ax.set_ylabel(pol.upper()); ax.grid(axis="y", alpha=0.4)
        axes[0].set_title("Pollutant Trends — Karachi May–Jun 2026", fontsize=13, fontweight="bold")
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        axes[-1].xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        plt.xticks(rotation=30); plt.tight_layout()
        fig.savefig(EDADIR / "04_pollutant_trends.png", dpi=150); plt.close()

    # ── 5. Correlation heatmap per city ─────────────────────────────────────
    num_cols = ["aqi","pm25","pm10","no2","so2","co","o3","temperature","humidity","wind_speed"]
    for city in cities:
        cd = df26[df26["city"]==city]
        cols = [c for c in num_cols if c in cd.columns]
        if len(cols) < 3: continue
        corr = cd[cols].corr()
        fig, ax = plt.subplots(figsize=(9, 7))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
                    center=0, square=True, linewidths=0.5, ax=ax, annot_kws={"size":8})
        ax.set_title(f"Correlation Matrix — {city.title()} May–Jun 2026", fontsize=12, fontweight="bold")
        plt.tight_layout()
        fig.savefig(EDADIR / f"05_corr_{city}.png", dpi=150); plt.close()

    # ── 6. Risk breakdown per city ───────────────────────────────────────────
    def risk(v):
        if v<=50: return "Good"
        if v<=100: return "Moderate"
        if v<=150: return "USG"
        if v<=200: return "Unhealthy"
        if v<=300: return "Very Unhealthy"
        return "Hazardous"

    order  = ["Good","Moderate","USG","Unhealthy","Very Unhealthy","Hazardous"]
    cmap   = ["#2ecc71","#f1c40f","#e67e22","#e74c3c","#9b59b6","#8e44ad"]
    fig, axes = plt.subplots(1, len(cities), figsize=(5*len(cities), 4))
    if len(cities)==1: axes=[axes]
    for ax, city in zip(axes, cities):
        cd = df26[df26["city"]==city]["aqi"].dropna()
        counts = cd.apply(risk).value_counts().reindex(order, fill_value=0)
        bars = ax.bar(counts.index, counts.values, color=cmap, edgecolor="white")
        ax.bar_label(bars, fmt="%d", padding=2, fontsize=8)
        ax.set_title(city.title()); ax.set_xlabel(""); ax.set_ylabel("Hours")
        plt.setp(ax.get_xticklabels(), rotation=20, fontsize=8)
    fig.suptitle("AQI Risk Breakdown — May–Jun 2026", fontsize=13, fontweight="bold")
    plt.tight_layout(); fig.savefig(EDADIR / "06_risk_breakdown.png", dpi=150); plt.close()

    # ── 7. Temperature vs AQI scatter ───────────────────────────────────────
    if "temperature" in df26.columns:
        fig, axes = plt.subplots(1, len(cities), figsize=(5*len(cities), 4))
        if len(cities)==1: axes=[axes]
        for ax, city in zip(axes, cities):
            cd = df26[df26["city"]==city]
            sc = ax.scatter(cd["temperature"], cd["aqi"],
                            c=cd["humidity"] if "humidity" in cd else city_colors.get(city,"#888"),
                            cmap="coolwarm", alpha=0.4, s=10)
            plt.colorbar(sc, ax=ax, label="Humidity")
            ax.set_title(city.title())
            ax.set_xlabel("Temp (°C)"); ax.set_ylabel("AQI")
        fig.suptitle("Temperature vs AQI", fontsize=13, fontweight="bold")
        plt.tight_layout(); fig.savefig(EDADIR / "07_temp_vs_aqi.png", dpi=150); plt.close()

    # ── 8. City AQI comparison bar ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    city_means = {c: df26[df26["city"]==c]["aqi"].mean() for c in cities}
    bars = ax.bar([c.title() for c in city_means], list(city_means.values()),
                  color=[city_colors.get(c,"#888") for c in city_means], edgecolor="white")
    ax.bar_label(bars, fmt="%.1f", padding=4)
    ax.set_title("Average AQI by City — May–Jun 2026", fontsize=13, fontweight="bold")
    ax.set_ylabel("Average AQI"); plt.tight_layout()
    fig.savefig(EDADIR / "08_city_comparison.png", dpi=150); plt.close()

    # ── 9. Full dataset daily trend all cities ───────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 4))
    all_cities = full_df["city"].str.lower().unique()
    for city in all_cities:
        cd = full_df[full_df["city"]==city].set_index("timestamp")
        daily = cd["aqi"].resample("D").mean().reset_index()
        ax.plot(daily["timestamp"], daily["aqi"], lw=1.1,
                color=city_colors.get(city,"#888"), label=city.title())
    ax.set_title("Daily Average AQI — Full Dataset All Cities", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date"); ax.set_ylabel("AQI")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.legend(); plt.xticks(rotation=30); plt.tight_layout()
    fig.savefig(EDADIR / "09_full_trend.png", dpi=150); plt.close()

    # ── 10. Summary stats ────────────────────────────────────────────────────
    stats_rows = []
    for city in cities:
        cd = df26[df26["city"]==city]["aqi"].dropna()
        stats_rows.append({"City": city.title(), "Mean": f"{cd.mean():.1f}",
                            "Median": f"{cd.median():.1f}", "Std": f"{cd.std():.1f}",
                            "Min": f"{cd.min():.1f}", "Max": f"{cd.max():.1f}",
                            "Count": str(len(cd))})
    stats = pd.DataFrame(stats_rows)
    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.axis("off")
    tbl = ax.table(cellText=stats.values, colLabels=stats.columns,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1.2, 1.6)
    ax.set_title("AQI Summary — May–Jun 2026", fontsize=13, fontweight="bold", pad=20)
    plt.tight_layout()
    fig.savefig(EDADIR / "10_summary_stats.png", dpi=150, bbox_inches="tight"); plt.close()

    print(f"[EDA] 10 plots saved → {EDADIR}/")


# ── 6. Main ───────────────────────────────────────────────────────────────────

def main():
    print("="*65)
    print("  AQI PIPELINE — 3-City: Fetch → Preprocess → Train → EDA")
    print("="*65)

    START = "2026-05-01"
    END   = date.today().strftime("%Y-%m-%d")
    print(f"\nDate range: {START} → {END}")

    # ── Step 1: fetch all 3 cities ───────────────────────────────────────────
    print("\n[STEP 1] Fetching data for all cities")
    all_new_rows = []
    city_dfs = {}   # city → raw DataFrame with aqi column (for training)

    for city, cfg in CITIES.items():
        try:
            raw = fetch_city_data(city, cfg["lat"], cfg["lon"], START, END)
            raw = calc_aqi(raw)
            rows = build_aqi_rows(raw, city, cfg)
            all_new_rows.append(rows)
            # keep a clean version for training (timestamp as index)
            city_dfs[city] = rows.copy()
        except Exception as e:
            print(f"  [ERROR] {city}: {e}")

    if not all_new_rows:
        print("[ERROR] No data fetched. Aborting."); return

    new_df   = pd.concat(all_new_rows, ignore_index=True)
    full_df  = update_aqi_csv(new_df)

    # ── Step 2: EDA ──────────────────────────────────────────────────────────
    print("\n[STEP 2] EDA on updated data")
    run_eda(full_df)

    # ── Step 3: train per-city per-horizon models ────────────────────────────
    print("\n[STEP 3] Training city-specific models")
    meta = {}

    for city in CITIES:
        print(f"\n  City: {city.title()}")
        # Use the freshly fetched rows for this city for training
        city_rows = full_df[full_df["city"] == city].copy()
        city_rows["timestamp"] = pd.to_datetime(city_rows["timestamp"])

        if len(city_rows) < 300:
            print(f"  [SKIP] Only {len(city_rows)} rows for {city} — not enough to train")
            continue

        city_meta = {}
        for horizon in [24, 48, 72]:
            try:
                feat_df = engineer_features(city_rows, horizon)
                result  = train_city_model(feat_df, city, horizon)
                if result:
                    city_meta[str(horizon)] = result
            except Exception as e:
                print(f"    [ERROR] {city} {horizon}h: {e}")

        if city_meta:
            meta[city] = city_meta

    # ── Step 4: save metadata ─────────────────────────────────────────────────
    meta_path = BCFG / "model_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n[META] Saved model_metadata.json — {sum(len(v) for v in meta.values())} models")

    print("\n" + "="*65)
    print("  PIPELINE COMPLETE")
    print(f"  Models    : {BMOD}")
    print(f"  EDA plots : {EDADIR}")
    print("="*65)


if __name__ == "__main__":
    main()
