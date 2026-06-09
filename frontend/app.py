import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from utils.api_client import fetch_api_data

st.set_page_config(page_title="AirLens", page_icon="🌬️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

#MainMenu, footer, header, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stSidebarNav"] { display: none !important; }

/* ── page bg ── */
[data-testid="stAppViewContainer"] > .main {
    background: #0f1117;
}
[data-testid="block-container"] {
    padding-top: 2rem;
    padding-bottom: 4rem;
    max-width: 1100px;
}

/* ── top bar ── */
.topbar {
    display: flex;
    align-items: baseline;
    gap: 0.6rem;
    margin-bottom: 0.2rem;
}
.site-name {
    font-size: 1.5rem;
    font-weight: 600;
    color: #f1f5f9;
    letter-spacing: -0.02em;
}
.site-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #38bdf8;
    display: inline-block;
    margin-bottom: 3px;
}
.site-tagline {
    font-size: 0.82rem;
    color: #475569;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-left: 0.2rem;
}

/* ── section labels ── */
.section-label {
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #475569;
    margin-bottom: 0.9rem;
    margin-top: 2.2rem;
}

/* ── aqi hero ── */
.aqi-hero {
    background: #161b27;
    border: 1px solid #1e293b;
    border-radius: 16px;
    padding: 2.4rem 2rem 2rem;
    display: flex;
    align-items: center;
    gap: 2rem;
}
.aqi-pill {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    width: 110px;
    height: 110px;
    border-radius: 50%;
    flex-shrink: 0;
}
.aqi-pill-num {
    font-family: 'DM Mono', monospace;
    font-size: 2.2rem;
    font-weight: 500;
    color: #fff;
    line-height: 1;
}
.aqi-pill-lbl {
    font-size: 0.65rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: rgba(255,255,255,0.65);
    margin-top: 3px;
}
.aqi-meta { flex: 1; }
.aqi-status {
    font-size: 1.5rem;
    font-weight: 600;
    color: #f1f5f9;
    letter-spacing: -0.02em;
    margin-bottom: 0.25rem;
}
.aqi-city {
    font-size: 0.85rem;
    color: #64748b;
    margin-bottom: 0.9rem;
}
.aqi-desc {
    font-size: 0.9rem;
    color: #94a3b8;
    line-height: 1.55;
}
.aqi-updated {
    font-size: 0.75rem;
    color: #334155;
    font-family: 'DM Mono', monospace;
    margin-top: 0.9rem;
}

/* ── stat tile ── */
.stat-tile {
    background: #161b27;
    border: 1px solid #1e293b;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
}
.stat-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.6rem;
}
.stat-key {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #475569;
    font-weight: 500;
}
.stat-badge {
    font-size: 0.72rem;
    font-weight: 500;
    padding: 0.18rem 0.55rem;
    border-radius: 20px;
    letter-spacing: 0.03em;
}
.stat-num {
    font-family: 'DM Mono', monospace;
    font-size: 2rem;
    font-weight: 400;
    color: #e2e8f0;
    margin-bottom: 0.2rem;
}
.stat-sub {
    font-size: 0.78rem;
    color: #334155;
}

/* ── air quality band ── */
.band {
    border-radius: 12px;
    padding: 1rem 1.3rem;
    border-left: 3px solid transparent;
    margin-top: 0.9rem;
}
.band-title { font-size: 0.88rem; font-weight: 600; color: #e2e8f0; margin-bottom: 0.15rem; }
.band-desc  { font-size: 0.8rem; color: #64748b; }

/* ── recommendations ── */
.rec-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.7rem;
}
.rec-item {
    background: #161b27;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 0.85rem 1rem;
    display: flex;
    gap: 0.7rem;
    align-items: flex-start;
}
.rec-icon { font-size: 1.1rem; margin-top: 1px; }
.rec-title { font-size: 0.82rem; font-weight: 600; color: #cbd5e1; }
.rec-desc  { font-size: 0.75rem; color: #475569; line-height: 1.4; margin-top: 1px; }

/* ── divider ── */
.soft-divider {
    border: none;
    border-top: 1px solid #1e293b;
    margin: 2rem 0;
}

/* ── forecast cards ── */
.fc-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.9rem;
}
.fc-card {
    background: #161b27;
    border: 1px solid #1e293b;
    border-radius: 14px;
    padding: 1.5rem 1rem;
    text-align: center;
}
.fc-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    color: #475569;
    font-weight: 500;
    margin-bottom: 0.3rem;
}
.fc-date {
    font-size: 0.78rem;
    color: #334155;
    font-family: 'DM Mono', monospace;
    margin-bottom: 1.1rem;
}
.fc-num {
    font-family: 'DM Mono', monospace;
    font-size: 2.5rem;
    font-weight: 400;
    color: #f1f5f9;
    line-height: 1;
    margin-bottom: 0.6rem;
}
.fc-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 5px;
    vertical-align: middle;
}
.fc-risk {
    font-size: 0.78rem;
    color: #94a3b8;
}
.fc-horizon {
    font-size: 0.68rem;
    color: #334155;
    font-family: 'DM Mono', monospace;
    margin-top: 0.7rem;
}

/* ── streamlit selectbox tweaks ── */
[data-testid="stSelectbox"] > div > div {
    background: #161b27 !important;
    border: 1px solid #1e293b !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}
div[data-baseweb="select"] * { color: #e2e8f0 !important; }

/* ── no data ── */
.no-data {
    background: #161b27;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    color: #334155;
    font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)

# ── AQI colour palette (muted, intentional) ──────────────────────────────────
def aqi_palette(v):
    if v <= 50:
        return {"bg":"#14532d","ring":"#22c55e","badge_bg":"#166534","badge_txt":"#86efac",
                "status":"Good","desc":"Air quality is clean. Fine for everyone outside."}
    if v <= 100:
        return {"bg":"#713f12","ring":"#eab308","badge_bg":"#854d0e","badge_txt":"#fde047",
                "status":"Moderate","desc":"Acceptable for most. Unusually sensitive people may want to limit prolonged outdoor exertion."}
    if v <= 150:
        return {"bg":"#7c2d12","ring":"#f97316","badge_bg":"#9a3412","badge_txt":"#fdba74",
                "status":"Unhealthy for Sensitive Groups","desc":"General public is fine. People with heart/lung conditions or children should reduce outdoor activity."}
    if v <= 200:
        return {"bg":"#7f1d1d","ring":"#ef4444","badge_bg":"#991b1b","badge_txt":"#fca5a5",
                "status":"Unhealthy","desc":"Everyone may start to feel effects. Sensitive groups should avoid prolonged outdoor exertion."}
    if v <= 300:
        return {"bg":"#581c87","ring":"#a855f7","badge_bg":"#6b21a8","badge_txt":"#d8b4fe",
                "status":"Very Unhealthy","desc":"Health alert. Everyone should avoid outdoor exertion."}
    return {"bg":"#450a0a","ring":"#b91c1c","badge_bg":"#7f1d1d","badge_txt":"#fca5a5",
            "status":"Hazardous","desc":"Emergency health warning. Entire population at risk."}

# ─────────────────────────────────────────────────────────────────────────────
# Top bar
# ─────────────────────────────────────────────────────────────────────────────
c1, c2 = st.columns([3, 1])
with c1:
    st.markdown("""
    <div class="topbar">
        <span class="site-name">AirLens</span>
        <span class="site-dot"></span>
        <span class="site-tagline">Pakistan Air Quality Monitor</span>
    </div>
    """, unsafe_allow_html=True)

with c2:
    city = st.selectbox("city", ["Karachi", "Lahore", "Islamabad"],
                        label_visibility="collapsed")

st.markdown("<hr class='soft-divider'>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Fetch
# ─────────────────────────────────────────────────────────────────────────────
data = fetch_api_data(f"/dashboard/overview?location={city}")

if not data:
    st.markdown("""<div class="no-data">
        <strong style="color:#475569">Backend unreachable</strong><br>
        Make sure the API is running on http://localhost:8000
    </div>""", unsafe_allow_html=True)
    st.stop()

aqi        = data["current_aqi"]
weekly_avg = data["weekly_avg"]
trend      = data["trend_direction"].title()
updated    = data["last_updated"]
pal        = aqi_palette(aqi)

# ─────────────────────────────────────────────────────────────────────────────
# Hero + Stats row
# ─────────────────────────────────────────────────────────────────────────────
left, right = st.columns([3, 2], gap="large")

with left:
    st.markdown('<div class="section-label">Current Conditions</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="aqi-hero">
        <div class="aqi-pill" style="background:{pal['bg']}; box-shadow: 0 0 0 3px {pal['ring']}22, 0 0 0 6px {pal['ring']}11;">
            <div class="aqi-pill-num">{aqi:.0f}</div>
            <div class="aqi-pill-lbl">AQI</div>
        </div>
        <div class="aqi-meta">
            <div class="aqi-status">{pal['status']}</div>
            <div class="aqi-city">{city}, Pakistan</div>
            <div class="aqi-desc">{pal['desc']}</div>
            <div class="aqi-updated">last updated {updated}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with right:
    st.markdown('<div class="section-label">This Week</div>', unsafe_allow_html=True)

    if trend == "Improving":
        badge_bg, badge_txt, badge_label = "#14532d", "#86efac", "↑ Improving"
    elif trend == "Stable":
        badge_bg, badge_txt, badge_label = "#1e3a5f", "#93c5fd", "→ Stable"
    else:
        badge_bg, badge_txt, badge_label = "#7f1d1d", "#fca5a5", "↓ Worsening"

    st.markdown(f"""
    <div class="stat-tile">
        <div class="stat-row">
            <span class="stat-key">7-day average</span>
            <span class="stat-badge" style="background:{badge_bg}; color:{badge_txt};">{badge_label}</span>
        </div>
        <div class="stat-num">{weekly_avg:.0f}</div>
        <div class="stat-sub">avg AQI over the past week</div>
    </div>
    """, unsafe_allow_html=True)

    # air quality band strip
    if aqi <= 50:
        band_bg, band_border, band_t, band_d = "#0f2318","#22c55e","Good","Safe for everyone"
    elif aqi <= 100:
        band_bg, band_border, band_t, band_d = "#1f1a0a","#eab308","Moderate","May affect very sensitive people"
    elif aqi <= 150:
        band_bg, band_border, band_t, band_d = "#1f1108","#f97316","Sensitive Groups","Reduce prolonged outdoor activity"
    elif aqi <= 200:
        band_bg, band_border, band_t, band_d = "#1f0808","#ef4444","Unhealthy","Limit outdoor time"
    else:
        band_bg, band_border, band_t, band_d = "#1a0820","#a855f7","Very Unhealthy / Hazardous","Avoid going outside"

    st.markdown(f"""
    <div class="band" style="background:{band_bg}; border-left-color:{band_border}; margin-top:0.9rem;">
        <div class="band-title">{band_t}</div>
        <div class="band-desc">{band_d}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr class='soft-divider'>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Recommendations
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Recommendations</div>', unsafe_allow_html=True)

if aqi <= 50:
    recs = [("🏃","Exercise","Great day for a run or outdoor sport"),
            ("🪟","Ventilation","Open windows — let the fresh air in"),
            ("👧","Children","Outdoor play is safe today"),
            ("🌿","Activities","Gardening and outdoor work are fine")]
elif aqi <= 100:
    recs = [("🚶","Light Exercise","Walking and casual activity is fine"),
            ("🏠","Sensitive People","Consider staying in if you have asthma"),
            ("🕐","Timing","Morning hours tend to be cleaner"),
            ("💧","Hydration","Drink water to support respiratory health")]
elif aqi <= 150:
    recs = [("🚫","Heavy Exercise","Avoid intense outdoor workouts"),
            ("😷","Masks","N95 mask recommended if going out"),
            ("👴","Elderly & Kids","Best to stay indoors today"),
            ("🪟","Windows","Keep windows closed, use AC if possible")]
else:
    recs = [("🏠","Stay Inside","Avoid outdoor activity entirely"),
            ("😷","Mask Up","N95 is essential for any trip outside"),
            ("🌀","Air Purifier","Run HEPA filter at home"),
            ("🩺","Watch Health","Check for respiratory discomfort")]

st.markdown('<div class="rec-grid">', unsafe_allow_html=True)
for icon, title, desc in recs:
    st.markdown(f"""
    <div class="rec-item">
        <div class="rec-icon">{icon}</div>
        <div>
            <div class="rec-title">{title}</div>
            <div class="rec-desc">{desc}</div>
        </div>
    </div>""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<hr class='soft-divider'>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Trend chart
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Historical Trend</div>', unsafe_allow_html=True)

hist = fetch_api_data(f"/historical/{city}?days=21")
if hist and hist.get("data"):
    df = pd.DataFrame(hist["data"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    fig = go.Figure()

    # fill area
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["aqi"],
        fill="tozeroy",
        fillcolor="rgba(56,189,248,0.06)",
        line=dict(width=0),
        showlegend=False, hoverinfo="skip"
    ))
    # main line
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["aqi"],
        mode="lines",
        line=dict(color="#38bdf8", width=1.8),
        name="AQI",
        hovertemplate="<b>%{y:.0f}</b><br>%{x|%b %d, %H:%M}<extra></extra>"
    ))

    # threshold lines — subtle
    for threshold, label, col in [
        (50,  "Good",      "#22c55e"),
        (100, "Moderate",  "#eab308"),
        (150, "Unhealthy", "#ef4444"),
    ]:
        fig.add_hline(y=threshold, line_dash="dash", line_color=col,
                      line_width=1, opacity=0.3,
                      annotation_text=label,
                      annotation_font_size=10,
                      annotation_font_color=col,
                      annotation_position="right")

    fig.update_layout(
        height=260,
        showlegend=False,
        plot_bgcolor="#0f1117",
        paper_bgcolor="#0f1117",
        font=dict(family="DM Sans", color="#475569", size=11),
        xaxis=dict(
            gridcolor="#1e293b", linecolor="#1e293b",
            tickformat="%b %d", tickfont=dict(size=10)
        ),
        yaxis=dict(
            gridcolor="#1e293b", linecolor="#1e293b",
            tickfont=dict(size=10), title=""
        ),
        margin=dict(l=0, r=60, t=10, b=10),
        hoverlabel=dict(
            bgcolor="#161b27", bordercolor="#1e293b",
            font_color="#e2e8f0", font_size=12
        )
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.markdown(f'<div class="no-data">No historical data available for {city}</div>',
                unsafe_allow_html=True)

st.markdown("<hr class='soft-divider'>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 3-Day Forecast
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">3-Day Forecast</div>', unsafe_allow_html=True)

forecast_data = fetch_api_data(f"/forecast?location={city}")

if forecast_data and forecast_data.get("forecasts"):
    day_labels = {24: "Tomorrow", 48: "In 2 days", 72: "In 3 days"}
    forecasts  = sorted(forecast_data["forecasts"], key=lambda x: x["horizon_hours"])[:3]

    # build html for all 3 cards at once (avoids column spacing issues)
    cards_html = '<div class="fc-grid">'
    for fc in forecasts:
        h       = fc["horizon_hours"]
        aqi_fc  = fc["predicted_aqi"]
        risk_fc = fc["risk_level"]
        date_fc = (datetime.now() + timedelta(hours=h)).strftime("%a, %b %d")
        fp      = aqi_palette(aqi_fc)

        cards_html += f"""
        <div class="fc-card">
            <div class="fc-label">{day_labels.get(h, f'{h}h')}</div>
            <div class="fc-date">{date_fc}</div>
            <div class="fc-num" style="color:{fp['ring']};">{aqi_fc:.0f}</div>
            <div class="fc-risk">
                <span class="fc-dot" style="background:{fp['ring']};"></span>
                {risk_fc}
            </div>
            <div class="fc-horizon">{h}h forecast</div>
        </div>"""
    cards_html += '</div>'
    st.markdown(cards_html, unsafe_allow_html=True)

else:
    st.markdown('<div class="no-data">Forecast unavailable</div>', unsafe_allow_html=True)
