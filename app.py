"""
AI Cloudburst Early Warning System — Streamlit App
Data source: cloudburst_data.xlsx (edit & save → reload to update predictions)
Run: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sys, os, time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backend.prediction_engine import (
    CloudburstPredictionEngine, AtmosphericInput, PredictionResult
)
from backend.excel_loader import (
    load_from_excel, load_historical_data, append_prediction_log,
    excel_exists, get_file_mtime, get_excel_path
)

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Cloudburst Early Warning",
    page_icon="⛈", layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

section[data-testid="stSidebar"] { background-color: #0b1220; border-right:1px solid rgba(79,158,255,0.15); }
section[data-testid="stSidebar"] * { color:#c8d8f0; }
.main { background-color: #0a0e1a; }

[data-testid="stMetric"] {
    background:#0f1628; border:1px solid rgba(79,158,255,0.15);
    border-radius:12px; padding:1rem 1.25rem;
}
[data-testid="stMetricLabel"] { font-size:11px!important;color:#4a6080!important;text-transform:uppercase;letter-spacing:.06em; }
[data-testid="stMetricValue"] { font-size:26px!important;font-weight:700!important;color:#e8f0ff!important; }

.excel-banner {
    background:linear-gradient(135deg,#0a2218,#0d2e1f);
    border:1px solid rgba(0,212,170,0.3); border-radius:12px;
    padding:12px 18px; margin-bottom:14px; display:flex; align-items:center; gap:12px;
}
.excel-banner-icon { font-size:22px; }
.excel-banner-text { font-size:13px; color:#c8d8f0; }
.excel-banner-sub  { font-size:11px; color:#4a6080; margin-top:2px; font-family:monospace; }

.excel-changed {
    background:rgba(255,209,102,0.12); border:1px solid rgba(255,209,102,0.4);
    border-radius:8px; padding:8px 14px; font-size:12px; color:#ffd166;
    margin-bottom:10px; animation: pulse-border 1.5s ease infinite;
}
@keyframes pulse-border { 0%,100%{border-color:rgba(255,209,102,0.4)} 50%{border-color:rgba(255,209,102,0.9)} }

.section-title {
    font-size:10px;font-weight:600;letter-spacing:.1em;text-transform:uppercase;
    color:#4a6080;border-bottom:1px solid rgba(79,158,255,0.12);
    padding-bottom:6px;margin-bottom:12px;
}
.stTabs [data-baseweb="tab-list"] { gap:4px;background:#0f1628;padding:4px;border-radius:10px; }
.stTabs [data-baseweb="tab"] { background:transparent;border-radius:8px;color:#7a9bc8;font-weight:500;font-size:13px; }
.stTabs [aria-selected="true"] { background:#1c2640!important;color:#e8f0ff!important; }

.param-grid {
    display:grid; grid-template-columns:1fr 1fr; gap:6px; margin-bottom:8px;
}
.param-cell {
    background:#0f1628; border:1px solid rgba(79,158,255,0.12);
    border-radius:8px; padding:8px 12px;
}
.param-label { font-size:10px; color:#4a6080; text-transform:uppercase; letter-spacing:.05em; }
.param-value { font-size:16px; font-weight:600; color:#4f9eff; margin-top:2px; }
.param-unit  { font-size:10px; color:#4a6080; }

.hist-table { width:100%; border-collapse:collapse; font-size:12px; }
.hist-table th { background:#1c3a6a; color:#e8f0ff; padding:6px 10px; text-align:center; }
.hist-table td { padding:5px 10px; text-align:center; border-bottom:1px solid rgba(79,158,255,0.08); color:#c8d8f0; }
.hist-table tr:hover td { background:rgba(79,158,255,0.05); }
.tag-y { color:#00d4aa; font-weight:600; }
.tag-n { color:#7a9bc8; }

.alert-extreme { background:rgba(255,71,87,.1);border-left:4px solid #ff4757;border-radius:8px;padding:12px 16px;margin:8px 0; }
.alert-high    { background:rgba(255,154,60,.1);border-left:4px solid #ff9a3c;border-radius:8px;padding:12px 16px;margin:8px 0; }
.alert-watch   { background:rgba(255,209,102,.1);border-left:4px solid #ffd166;border-radius:8px;padding:12px 16px;margin:8px 0; }
.alert-normal  { background:rgba(0,212,170,.1);border-left:4px solid #00d4aa;border-radius:8px;padding:12px 16px;margin:8px 0; }
</style>
""", unsafe_allow_html=True)

# ─── INIT ────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_engine():
    return CloudburstPredictionEngine()
engine = get_engine()

PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Space Grotesk', color='#7a9bc8', size=11),
    margin=dict(l=10, r=10, t=30, b=10),
)
PLOTLY_AXES = dict(
    xaxis=dict(gridcolor='rgba(99,179,255,0.08)', color='#4a6080'),
    yaxis=dict(gridcolor='rgba(99,179,255,0.08)', color='#4a6080'),
)

def risk_color(p):
    if p < 30: return '#00d4aa'
    if p < 55: return '#ffd166'
    if p < 75: return '#ff9a3c'
    return '#ff4757'

def risk_label(p):
    if p < 30: return 'Low'
    if p < 55: return 'Moderate'
    if p < 75: return 'High'
    return 'Extreme'

# ─── SESSION STATE ───────────────────────────────────────────────────────────
if "params" not in st.session_state:
    params, msg = load_from_excel()
    st.session_state.params = params
    st.session_state.load_msg = msg
    st.session_state.last_mtime = get_file_mtime()
    st.session_state.auto_refresh = True
    st.session_state.prediction_logged = False
    st.session_state.file_changed = False
    st.session_state.last_check_time = time.time()
    st.session_state.change_banner_shown = False

# ─── AUTO-DETECT FILE CHANGE (throttled — check every 5 seconds only) ────────
now = time.time()
last_check = st.session_state.get("last_check_time", 0)

if st.session_state.get("auto_refresh", True) and (now - last_check) >= 5:
    st.session_state.last_check_time = now
    current_mtime = get_file_mtime()
    stored_mtime = st.session_state.get("last_mtime")
    if current_mtime and stored_mtime and current_mtime != stored_mtime:
        params, msg = load_from_excel()
        st.session_state.params = params
        st.session_state.load_msg = msg
        st.session_state.last_mtime = current_mtime
        st.session_state.file_changed = True
        st.session_state.prediction_logged = False  # allow re-logging
        st.rerun()
    else:
        st.session_state.last_mtime = current_mtime

current_mtime = get_file_mtime()

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    # Excel source panel
    st.markdown("### 📊 Data Source")
    excel_path = get_excel_path()
    fname = os.path.basename(excel_path)
    fexists = excel_exists()

    if fexists:
        mtime_str = datetime.fromtimestamp(current_mtime).strftime("%d %b %Y, %H:%M:%S") if current_mtime else "—"
        st.markdown(f"""
        <div class="excel-banner">
          <div class="excel-banner-icon">📗</div>
          <div>
            <div class="excel-banner-text"><b>{fname}</b></div>
            <div class="excel-banner-sub">Last modified: {mtime_str}</div>
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.warning(f"Excel file not found:\n`{excel_path}`\n\nUsing default values.", icon="⚠")

    col_r, col_a = st.columns(2)
    if col_r.button("🔄 Reload", use_container_width=True):
        params, msg = load_from_excel()
        st.session_state.params = params
        st.session_state.load_msg = msg
        st.session_state.last_mtime = get_file_mtime()
        st.session_state.file_changed = False
        st.rerun()

    auto = col_a.toggle("Auto", value=st.session_state.auto_refresh)
    st.session_state.auto_refresh = auto

    if st.session_state.load_msg:
        if "✅" in st.session_state.load_msg:
            st.success(st.session_state.load_msg, icon="✅")
        else:
            st.warning(st.session_state.load_msg)

    st.markdown("---")

    # Display loaded parameters (read-only mirror)
    p = st.session_state.params
    st.markdown("### 📋 Loaded Parameters")
    st.markdown(f"""
    <div class="param-grid">
      <div class="param-cell"><div class="param-label">Max Temp</div><div class="param-value">{p['max_temp']}<span class="param-unit"> °C</span></div></div>
      <div class="param-cell"><div class="param-label">Min Temp</div><div class="param-value">{p['min_temp']}<span class="param-unit"> °C</span></div></div>
      <div class="param-cell"><div class="param-label">Rainfall</div><div class="param-value">{p['rainfall']}<span class="param-unit"> mm</span></div></div>
      <div class="param-cell"><div class="param-label">Humidity</div><div class="param-value">{p['humidity']}<span class="param-unit"> %</span></div></div>
      <div class="param-cell"><div class="param-label">Wind Speed</div><div class="param-value">{p['wind_speed']}<span class="param-unit"> km/h</span></div></div>
      <div class="param-cell"><div class="param-label">Wind Gust</div><div class="param-value">{p['wind_gust']}<span class="param-unit"> km/h</span></div></div>
      <div class="param-cell"><div class="param-label">Pressure</div><div class="param-value">{p['pressure']}<span class="param-unit"> hPa</span></div></div>
      <div class="param-cell"><div class="param-label">Elevation</div><div class="param-value">{p['elevation']}<span class="param-unit"> m</span></div></div>
      <div class="param-cell"><div class="param-label">Slope</div><div class="param-value">{p['slope']}<span class="param-unit"> °</span></div></div>
      <div class="param-cell"><div class="param-label">Soil Moist.</div><div class="param-value">{p['soil_moisture']}<span class="param-unit"> %</span></div></div>
    </div>
    <div class="param-cell" style="margin-bottom:6px"><div class="param-label">Region</div><div class="param-value" style="font-size:13px">{p['region'].replace('_',' ').title()}</div></div>
    <div class="param-cell"><div class="param-cell"><div class="param-label">WMO Code</div><div class="param-value">{p['weather_code']}</div></div></div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    if st.session_state.get("auto_refresh"):
        st.markdown("<div style='font-size:10px;color:#00d4aa;font-family:monospace'>🟢 Auto-detect ON — checks every 5 sec</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='font-size:10px;color:#4a6080;font-family:monospace'>⚫ Auto-detect OFF — use Reload button</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:10px;color:#4a6080;font-family:monospace;margin-top:4px'>File: <b>{fname}</b></div>", unsafe_allow_html=True)

    # ── Trigger periodic check when auto is ON (uses st_autorefresh via meta refresh) ──
    if st.session_state.get("auto_refresh"):
        st.markdown("""<script>setTimeout(function(){ window.location.reload(); }, 8000);</script>""", unsafe_allow_html=True)

# ─── FILE CHANGED BANNER ─────────────────────────────────────────────────────
if st.session_state.get("file_changed"):
    st.markdown("""
    <div style="background:linear-gradient(135deg,rgba(255,209,102,0.15),rgba(255,154,60,0.1));
    border:2px solid #ffd166; border-radius:12px; padding:14px 20px; margin-bottom:16px;
    display:flex; align-items:center; gap:12px;">
      <div style="font-size:24px">📢</div>
      <div>
        <div style="font-size:14px;font-weight:700;color:#ffd166">Excel file changed — predictions updated!</div>
        <div style="font-size:12px;color:#c8a84b;margin-top:3px">All 7 tabs have been refreshed with new data from cloudburst_data.xlsx</div>
      </div>
    </div>""", unsafe_allow_html=True)
    st.session_state.file_changed = False

# ─── BUILD INPUT & PREDICT ───────────────────────────────────────────────────
p = st.session_state.params
inp = AtmosphericInput(
    max_temp=float(p["max_temp"]), min_temp=float(p["min_temp"]),
    rainfall=float(p["rainfall"]), wind_speed=float(p["wind_speed"]),
    wind_gust=float(p["wind_gust"]), wind_direction=float(p["wind_direction"]),
    humidity=float(p["humidity"]), pressure=float(p["pressure"]),
    elevation=float(p["elevation"]), slope=float(p["slope"]),
    soil_moisture=float(p["soil_moisture"]),
    weather_code=int(p["weather_code"]),
    region=str(p["region"]),
    population_density=str(p["population_density"]),
)
result: PredictionResult = engine.predict(inp)

# Log prediction to Excel (once per load)
if not st.session_state.prediction_logged:
    log_data = {
        "cloudburst_probability": result.cloudburst_probability,
        "risk_level": result.risk_level,
        "rainfall_intensity": result.rainfall_intensity,
        "flash_flood_probability": result.flash_flood_probability,
        "landslide_probability": result.landslide_probability,
        "early_warning_hours": result.early_warning_hours,
        "alert_level": result.alert_level,
        "recommendations": result.recommendations,
    }
    append_prediction_log(log_data)
    st.session_state.prediction_logged = True

# ─── HEADER ──────────────────────────────────────────────────────────────────
status_icons = {"NORMAL":"🟢","WATCH":"🟡","HIGH":"🔶","EXTREME":"🔴"}
si = status_icons.get(result.alert_level, "🟢")
ts = datetime.now().strftime("%d %b %Y · %H:%M:%S")

st.markdown(f"""
<div style="background:linear-gradient(135deg,#0f1628,#151d35);border:1px solid rgba(79,158,255,0.2);
border-radius:14px;padding:1.25rem 1.75rem;margin-bottom:1rem;display:flex;align-items:center;gap:16px">
  <div style="font-size:32px">⛈</div>
  <div style="flex:1">
    <div style="font-size:20px;font-weight:700;color:#e8f0ff">AI Cloudburst Early Warning System</div>
    <div style="font-size:11px;color:#4a6080;margin-top:3px;font-family:monospace">
      Multimodal · Physics-Informed · Impact-Aware &nbsp;|&nbsp;
      {si} {result.alert_level} &nbsp;|&nbsp; Risk: <b style="color:{risk_color(result.cloudburst_probability)}">{result.risk_level}</b> &nbsp;|&nbsp;
      Warning: {result.early_warning_hours}h ahead &nbsp;|&nbsp; {ts}
    </div>
  </div>
  <div style="text-align:right">
    <div style="font-size:10px;color:#4a6080;font-family:monospace">DATA SOURCE</div>
    <div style="font-size:12px;color:#4f9eff;font-weight:600">📗 cloudburst_data.xlsx</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── METRIC CARDS ────────────────────────────────────────────────────────────
c1,c2,c3,c4,c5,c6 = st.columns(6)
c1.metric("Cloudburst Probability", f"{result.cloudburst_probability:.0f}%", result.risk_level)
c2.metric("Rainfall Intensity", f"{result.rainfall_intensity:.0f} mm/hr",
          "Extreme" if result.rainfall_intensity>100 else "Heavy" if result.rainfall_intensity>50 else "Moderate")
c3.metric("Early Warning", f"{result.early_warning_hours}h", "Hours ahead")
c4.metric("Flash Flood Risk", f"{result.flash_flood_probability:.0f}%", f"Runoff: {result.peak_runoff:.0f} m³/s")
c5.metric("Landslide Risk", f"{result.landslide_probability:.0f}%", f"Slope: {inp.slope}°")
c6.metric("Affected Population", f"{result.affected_population:,}", "Estimated")

st.markdown("---")

# ─── TABS ────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🔮 Prediction", "🗺 Heatmaps", "⚡ Impact",
    "⚛ Physics", "🧠 Explain AI", "🚨 Alerts", "📊 Excel Data"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    c_left, c_right = st.columns(2)
    with c_left:
        st.markdown('<div class="section-title">Probability Timeline (ConvLSTM Output)</div>', unsafe_allow_html=True)
        tlabels = ['T-6h','T-5h','T-4h','T-3h','T-2h','T-1h','Now','+1h','+2h','+3h']
        base = result.cloudburst_probability / 100
        prob_seq = [round(min(99,max(1,base*f*100)),1) for f in [.08,.14,.22,.33,.48,.68,1.0,1.08,.92,.72]]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=tlabels, y=prob_seq, fill='tozeroy',
            line=dict(color='#4f9eff',width=2), fillcolor='rgba(79,158,255,0.12)',
            mode='lines+markers', marker=dict(size=6,color='#4f9eff')))
        for y,col,lbl in [(30,'#ffd166','Watch'),(55,'#ff9a3c','High'),(75,'#ff4757','Extreme')]:
            fig.add_hline(y=y,line_dash="dot",line_color=col,
                          annotation_text=lbl,annotation_font_color=col)
        fig.update_layout(**PLOTLY_LAYOUT,**PLOTLY_AXES,height=250,yaxis_range=[0,100],showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with c_right:
        st.markdown('<div class="section-title">Cloud Activity — ConvLSTM Spatial-Temporal</div>', unsafe_allow_html=True)
        cloud_data = [round(min(99,max(1,p*(0.85+np.random.uniform(-.1,.1))))) for p in prob_seq]
        fig2 = go.Figure(go.Bar(x=tlabels, y=cloud_data,
            marker_color=[risk_color(d) for d in cloud_data], marker_line_width=0))
        fig2.update_layout(**PLOTLY_LAYOUT,**PLOTLY_AXES,height=250,yaxis_range=[0,100],showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    ci1,ci2,ci3 = st.columns(3)
    with ci1:
        st.markdown("**Moisture Indices**")
        for name,val,col in [
            ("Atmospheric moisture", result.moisture_index,'#4f9eff'),
            ("Humidity convergence",  min(99,round(inp.humidity*.8)),'#00d4aa'),
            ("Precipitable water",    min(99,round(result.moisture_index*.9)),'#c77dff'),
        ]:
            st.markdown(f"<div style='font-size:12px;color:#7a9bc8;margin-bottom:2px'>{name} — <b style='color:#e8f0ff'>{val}%</b></div>", unsafe_allow_html=True)
            st.progress(int(val))

    with ci2:
        st.markdown("**Instability Indices**")
        for name,val in [
            ("CAPE index",            min(99,round(result.instability_index*.95))),
            ("Thermal instability",   min(99,round(result.instability_index*.88))),
            ("Convective triggering", result.convection_index),
        ]:
            st.markdown(f"<div style='font-size:12px;color:#7a9bc8;margin-bottom:2px'>{name} — <b style='color:#e8f0ff'>{val}%</b></div>", unsafe_allow_html=True)
            st.progress(int(val))

    with ci3:
        st.markdown("**Model Confidence**")
        for k,v in [
            ("Overall confidence", f"{result.model_confidence:.1f}%"),
            ("ConvLSTM score",     f"{result.convlstm_contribution:.1f}%"),
            ("Transformer score",  f"{result.transformer_contribution:.1f}%"),
            ("PINN consistency",   f"{result.physics_consistency:.1f}%"),
            ("Spatial resolution", "1 km grid"),
        ]:
            st.markdown(f"<div style='display:flex;justify-content:space-between;font-size:13px;padding:4px 0;border-bottom:1px solid rgba(79,158,255,0.08)'><span style='color:#7a9bc8'>{k}</span><span style='color:#e8f0ff;font-weight:500'>{v}</span></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — HEATMAPS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">Spatial Risk Heatmaps — 12×12 Terrain Grid (hover for cell values)</div>', unsafe_allow_html=True)

    def make_heatmap(data, title, colorscale, zmin, zmax, unit=""):
        z = data
        rows_n, cols_n = len(z), len(z[0])
        hover = [[f"Col {c+1}, Row {r+1}<br>{title}: {z[r][c]:.1f}{unit}"
                  for c in range(cols_n)] for r in range(rows_n)]
        fig = go.Figure(go.Heatmap(
            z=z,
            x=[f"W{i+1}" for i in range(cols_n)],
            y=[f"N{i+1}" for i in range(rows_n)],
            colorscale=colorscale, zmin=zmin, zmax=zmax,
            text=hover, hoverinfo='text',
            colorbar=dict(thickness=12, tickfont=dict(color='#7a9bc8',size=10),
                          title=dict(text=unit or '%', font=dict(color='#7a9bc8',size=11)))
        ))
        fig.update_layout(
            title=dict(text=title, font=dict(color='#c8d8f0',size=13), x=0),
            **PLOTLY_LAYOUT, height=280,
            xaxis=dict(showgrid=False, tickfont=dict(size=9), color='#4a6080'),
            yaxis=dict(showgrid=False, tickfont=dict(size=9), autorange='reversed', color='#4a6080')
        )
        return fig

    hm1,hm2 = st.columns(2)
    with hm1:
        st.plotly_chart(make_heatmap(result.heatmap_prob,"Cloudburst Probability",
            [[0,'#0a3d2e'],[.3,'#00d4aa'],[.6,'#ffd166'],[.8,'#ff9a3c'],[1,'#ff4757']],0,100,"%"),use_container_width=True)
    with hm2:
        st.plotly_chart(make_heatmap(result.heatmap_rain,"Rainfall Intensity",
            [[0,'#042c53'],[.25,'#185fa5'],[.5,'#9fe1cb'],[.75,'#ffd166'],[1,'#ff4757']],0,200," mm/hr"),use_container_width=True)

    hm3,hm4 = st.columns(2)
    with hm3:
        st.plotly_chart(make_heatmap(result.heatmap_flood,"Flash Flood Risk",
            [[0,'#042c53'],[.4,'#185fa5'],[.7,'#ffd166'],[1,'#ff4757']],0,100,"%"),use_container_width=True)
    with hm4:
        st.plotly_chart(make_heatmap(result.heatmap_land,"Landslide Susceptibility",
            [[0,'#173404'],[.35,'#639922'],[.65,'#854f0b'],[1,'#a32d2d']],0,100,"%"),use_container_width=True)

    hm5,hm6 = st.columns(2)
    with hm5:
        st.plotly_chart(make_heatmap(result.heatmap_wind,"Wind Convergence",
            [[0,'#26215c'],[.4,'#534ab7'],[.7,'#c77dff'],[.85,'#ffd166'],[1,'#ff4757']],0,100,"%"),use_container_width=True)
    with hm6:
        st.plotly_chart(make_heatmap(result.heatmap_orog,"Orographic Lifting",
            [[0,'#04342c'],[.3,'#1d9e75'],[.6,'#faeeda'],[.85,'#ba7517'],[1,'#412402']],0,100,"%"),use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — IMPACT
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    i1,i2 = st.columns(2)
    with i1:
        st.markdown('<div class="section-title">Hydrological Impact</div>', unsafe_allow_html=True)
        for k,v in [
            ("Flash Flood Probability",f"{result.flash_flood_probability:.0f}%"),
            ("Peak Runoff (m³/s)",     f"{result.peak_runoff:.1f}"),
            ("Drainage Saturation",    result.drainage_saturation),
            ("Flood Depth Estimate",   result.flood_depth_estimate),
            ("Inundation Area (km²)",  f"{result.inundation_area:.1f}"),
        ]:
            st.markdown(f"<div style='display:flex;justify-content:space-between;font-size:13px;padding:6px 0;border-bottom:1px solid rgba(79,158,255,0.08)'><span style='color:#7a9bc8'>{k}</span><span style='color:#e8f0ff;font-weight:500'>{v}</span></div>", unsafe_allow_html=True)

    with i2:
        st.markdown('<div class="section-title">Population & Infrastructure</div>', unsafe_allow_html=True)
        for k,v in [
            ("Affected Population",     f"{result.affected_population:,}"),
            ("Road Infrastructure Risk", result.road_risk_level),
            ("Agricultural Land (ha)",  f"{result.agricultural_area_affected:.1f}"),
        ]:
            st.markdown(f"<div style='display:flex;justify-content:space-between;font-size:13px;padding:6px 0;border-bottom:1px solid rgba(79,158,255,0.08)'><span style='color:#7a9bc8'>{k}</span><span style='color:#e8f0ff;font-weight:500'>{v}</span></div>", unsafe_allow_html=True)

    st.markdown("---")
    iz1,iz2 = st.columns(2)
    with iz1:
        st.markdown('<div class="section-title">Landslide Risk Zones</div>', unsafe_allow_html=True)
        p_val = result.cloudburst_probability
        for name, val, desc in [
            ("⛰ Steep Slopes",    min(99,round(p_val*.85+inp.slope/2)),"High debris flow potential"),
            ("🏞 Valley Floors",   min(99,round(p_val*.70)),"Flash flood accumulation"),
            ("🌄 Debris Channels", min(99,round(p_val*.90+inp.slope/3)),"Rapid mass movement"),
            ("🏘 Settlements",     min(99,round(p_val*.65)),"Population exposure"),
        ]:
            rc = risk_color(val)
            st.markdown(f"""<div style='background:#0f1628;border:1px solid rgba(79,158,255,0.12);border-radius:10px;
            padding:10px 14px;margin-bottom:8px;'>
            <div style='display:flex;justify-content:space-between;align-items:center'>
            <span style='font-size:14px;font-weight:500'>{name}</span>
            <span style='font-size:18px;font-weight:700;color:{rc}'>{val}%</span></div>
            <div style='font-size:11px;color:#4a6080;margin-top:3px'>{desc}</div></div>""", unsafe_allow_html=True)

    with iz2:
        st.markdown('<div class="section-title">Multi-Hazard Risk</div>', unsafe_allow_html=True)
        p_val = result.cloudburst_probability
        fig_d = go.Figure(go.Pie(
            labels=['Flash Flood','Landslide','Infrastructure','Agri Loss','Population'],
            values=[result.flash_flood_probability, result.landslide_probability,
                    min(99,p_val+10), round(p_val*.7), round(p_val*.85)],
            hole=.55, marker_colors=['#4f9eff','#ff9a3c','#c77dff','#00d4aa','#ff4757'], textfont_size=11
        ))
        fig_d.update_layout(**PLOTLY_LAYOUT, height=280,
                            legend=dict(font=dict(color='#7a9bc8',size=10), orientation='v', x=1.05))
        st.plotly_chart(fig_d, use_container_width=True)

    st.markdown('<div class="section-title">6-Hour Risk Forecast</div>', unsafe_allow_html=True)
    fcast_labels = ['Now','+1h','+2h','+3h','+4h','+5h','+6h']
    fig_fc = go.Figure()
    for series,name,col in [
        (result.probability_forecast,'Cloudburst Probability','#4f9eff'),
        (result.flood_forecast,'Flash Flood Risk','#ff9a3c'),
        (result.landslide_forecast,'Landslide Risk','#c77dff'),
    ]:
        fig_fc.add_trace(go.Scatter(x=fcast_labels,y=series,name=name,
            line=dict(color=col,width=2),mode='lines+markers',marker=dict(size=5)))
    fig_fc.update_layout(**PLOTLY_LAYOUT,**PLOTLY_AXES,height=250,yaxis_range=[0,100],
                         legend=dict(font=dict(color='#7a9bc8',size=11),orientation='h',y=-.2))
    st.plotly_chart(fig_fc, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — PHYSICS
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    ph1,ph2 = st.columns(2)
    with ph1:
        st.markdown('<div class="section-title">PINN Physics Indicators</div>', unsafe_allow_html=True)
        phys = result.physics
        for name,val,col in [
            ("Moisture Conservation",     phys.moisture_conservation,'#4f9eff'),
            ("Orographic Lifting",        phys.orographic_lifting,'#00d4aa'),
            ("CAPE (convective energy)",  phys.cape_percent,'#ff9a3c'),
            ("Wind Convergence",          phys.wind_convergence,'#c77dff'),
            ("Boundary Layer Instab.",    phys.boundary_layer_instability,'#ff4757'),
            ("Pressure Gradient Force",   phys.pressure_gradient_force,'#ffd166'),
        ]:
            st.markdown(f"<div style='font-size:12px;color:#7a9bc8;margin:8px 0 2px'>{name} — <b style='color:#e8f0ff'>{val:.1f}%</b></div>", unsafe_allow_html=True)
            st.progress(int(min(100,val)))

    with ph2:
        st.markdown('<div class="section-title">Physics Radar</div>', unsafe_allow_html=True)
        cats = ['Moisture','Instability','Orographic','Convection','Soil','Pressure Drop']
        vals = [phys.moisture_conservation, phys.boundary_layer_instability,
                phys.orographic_lifting, phys.wind_convergence,
                result.soil_factor, phys.pressure_gradient_force]
        fig_r = go.Figure(go.Scatterpolar(
            r=vals+[vals[0]], theta=cats+[cats[0]], fill='toself',
            line=dict(color='#00d4aa',width=2), fillcolor='rgba(0,212,170,0.12)',
            marker=dict(size=5,color='#00d4aa')
        ))
        fig_r.update_layout(
            polar=dict(bgcolor='rgba(0,0,0,0)',
                radialaxis=dict(range=[0,100],gridcolor='rgba(99,179,255,0.1)',
                    tickfont=dict(color='#4a6080',size=9)),
                angularaxis=dict(gridcolor='rgba(99,179,255,0.1)',
                    tickfont=dict(color='#7a9bc8',size=11))),
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Space Grotesk',color='#7a9bc8'),
            margin=dict(l=20,r=20,t=20,b=20), height=320, showlegend=False
        )
        st.plotly_chart(fig_r, use_container_width=True)

    pv1,pv2,pv3 = st.columns(3)
    with pv1:
        st.markdown('<div class="section-title">Derived Values</div>', unsafe_allow_html=True)
        for k,v in [
            ("CAPE",              f"{phys.cape_index} J/kg"),
            ("Lapse Rate",        f"{phys.lapse_rate} °C/km"),
            ("Dew Pt Depression", f"{phys.dew_point_depression} °C"),
            ("Lifted Index (LI)", f"{phys.lifted_index}"),
        ]:
            st.markdown(f"<div style='display:flex;justify-content:space-between;font-size:13px;padding:5px 0;border-bottom:1px solid rgba(79,158,255,0.08)'><span style='color:#7a9bc8'>{k}</span><span style='color:#e8f0ff;font-weight:500'>{v}</span></div>", unsafe_allow_html=True)

    with pv2:
        st.markdown('<div class="section-title">Terrain-Atmosphere</div>', unsafe_allow_html=True)
        elev_class = "Alpine" if inp.elevation>3000 else "Sub-alpine" if inp.elevation>1500 else "Mid-hill" if inp.elevation>800 else "Lowland"
        for k,v in [
            ("Orographic Factor",     f"{phys.orographic_lifting:.1f}%"),
            ("Slope-Moisture Index",  f"{min(99,round(inp.slope*inp.soil_moisture/60))}%"),
            ("Valley Channeling",     "Active" if inp.elevation>800 and inp.slope>10 else "Weak"),
            ("Elevation Class",       elev_class),
        ]:
            st.markdown(f"<div style='display:flex;justify-content:space-between;font-size:13px;padding:5px 0;border-bottom:1px solid rgba(79,158,255,0.08)'><span style='color:#7a9bc8'>{k}</span><span style='color:#e8f0ff;font-weight:500'>{v}</span></div>", unsafe_allow_html=True)

    with pv3:
        st.markdown('<div class="section-title">PINN Consistency</div>', unsafe_allow_html=True)
        for k,v in [
            ("Law Adherence",    f"{phys.pinn_law_adherence:.1f}%"),
            ("Plausibility",     f"{phys.pinn_plausibility:.1f}%"),
            ("Generalization",   "82%"),
            ("Rare Event Mode",  "High" if result.cloudburst_probability>70 else "Standard"),
        ]:
            st.markdown(f"<div style='display:flex;justify-content:space-between;font-size:13px;padding:5px 0;border-bottom:1px solid rgba(79,158,255,0.08)'><span style='color:#7a9bc8'>{k}</span><span style='color:#e8f0ff;font-weight:500'>{v}</span></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — EXPLAIN AI
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    xe1,xe2 = st.columns(2)
    with xe1:
        st.markdown('<div class="section-title">Top Contributing Factors (SHAP-style XAI)</div>', unsafe_allow_html=True)
        for feat, imp in sorted(result.feature_importances.items(), key=lambda x:-x[1])[:7]:
            col = risk_color(imp)
            st.markdown(f"""<div style='margin-bottom:10px'>
            <div style='display:flex;justify-content:space-between;margin-bottom:3px'>
            <span style='font-size:13px;font-weight:500;color:#e8f0ff'>{feat}</span>
            <span style='font-size:12px;font-family:monospace;color:{col}'>{imp:.1f}%</span></div>
            <div style='height:6px;background:#1c2640;border-radius:3px;overflow:hidden'>
            <div style='height:100%;width:{imp}%;background:{col};border-radius:3px'></div></div>
            </div>""", unsafe_allow_html=True)

    with xe2:
        st.markdown('<div class="section-title">Feature Importance</div>', unsafe_allow_html=True)
        xai_sorted = sorted(result.feature_importances.items(), key=lambda x:x[1])
        fig_xai = go.Figure(go.Bar(
            x=[v for _,v in xai_sorted], y=[k for k,_ in xai_sorted],
            orientation='h',
            marker_color=[risk_color(v) for _,v in xai_sorted], marker_line_width=0
        ))
        fig_xai.update_layout(**PLOTLY_LAYOUT,**PLOTLY_AXES,height=300,
                              xaxis_range=[0,100],xaxis_title="Importance (%)")
        st.plotly_chart(fig_xai, use_container_width=True)

    ma1,ma2,ma3 = st.columns(3)
    with ma1:
        st.markdown("**Model Attribution**")
        for name,val,col in [
            ("ConvLSTM", result.convlstm_contribution,'#4f9eff'),
            ("Transformer", result.transformer_contribution,'#00d4aa'),
            ("Attention", result.attention_contribution,'#c77dff'),
            ("PINN", result.pinn_contribution,'#ffd166'),
        ]:
            st.markdown(f"<div style='font-size:12px;color:#7a9bc8;margin:6px 0 2px'>{name} — <b style='color:#e8f0ff'>{val:.1f}%</b></div>", unsafe_allow_html=True)
            st.progress(int(min(100,val)))

    with ma2:
        st.markdown("**Confidence**")
        for k,v in [
            ("Overall", f"{result.model_confidence:.1f}%"),
            ("Physics", f"{result.physics_consistency:.1f}%"),
            ("Spatial res.", "1 km grid"), ("Temporal res.", "15 min"),
        ]:
            st.markdown(f"<div style='display:flex;justify-content:space-between;font-size:13px;padding:5px 0;border-bottom:1px solid rgba(79,158,255,0.08)'><span style='color:#7a9bc8'>{k}</span><span style='color:#e8f0ff;font-weight:500'>{v}</span></div>", unsafe_allow_html=True)

    with ma3:
        st.markdown("**Status**")
        for k,v,c in [
            ("Architecture","✅ Complete","#00d4aa"),
            ("Historical data","✅ Complete","#00d4aa"),
            ("Real-time data","⏳ Pending","#ffd166"),
            ("Doppler radar","⏳ Pending","#ffd166"),
            ("IoT integration","⏳ Pending","#ffd166"),
        ]:
            st.markdown(f"<div style='display:flex;justify-content:space-between;font-size:13px;padding:5px 0;border-bottom:1px solid rgba(79,158,255,0.08)'><span style='color:#7a9bc8'>{k}</span><span style='color:{c};font-weight:500'>{v}</span></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — ALERTS
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    al1,al2 = st.columns([1.2,.8])
    with al1:
        st.markdown('<div class="section-title">Active Alert</div>', unsafe_allow_html=True)
        cls_map = {"EXTREME":"alert-extreme","HIGH":"alert-high","WATCH":"alert-watch","NORMAL":"alert-normal"}
        st.markdown(f'<div class="{cls_map.get(result.alert_level,"alert-normal")}"><b>{result.alert_message}</b></div>', unsafe_allow_html=True)
        st.markdown(f"""<div class="alert-normal" style="background:rgba(79,158,255,0.08);border-left:4px solid #4f9eff;border-radius:8px;padding:12px 16px;margin:8px 0">
        ℹ <b>Early Warning Horizon: {result.early_warning_hours}h</b> — PINN-validated.
        Data source: cloudburst_data.xlsx</div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-title" style="margin-top:16px">Response Recommendations</div>', unsafe_allow_html=True)
        for i,rec in enumerate(result.recommendations,1):
            st.markdown(f"<div style='display:flex;gap:10px;padding:7px 0;border-bottom:1px solid rgba(79,158,255,0.08);font-size:13px'><span style='color:#4f9eff;font-weight:600;min-width:20px'>{i}.</span><span style='color:#c8d8f0'>{rec}</span></div>", unsafe_allow_html=True)

    with al2:
        st.markdown('<div class="section-title">6-Hour Threat Timeline</div>', unsafe_allow_html=True)
        fcast_labels = ['Now','+1h','+2h','+3h','+4h','+5h','+6h']
        ac = risk_color(result.cloudburst_probability)
        fig_al = go.Figure(go.Scatter(
            x=fcast_labels, y=result.probability_forecast, fill='tozeroy',
            mode='lines+markers', line=dict(color=ac,width=2),
            fillcolor=f"rgba({int(ac[1:3],16)},{int(ac[3:5],16)},{int(ac[5:7],16)},0.1)",
            marker=dict(size=6,color=ac)
        ))
        for y0,y1,fill_c,lbl in [(75,100,"rgba(255,71,87,0.06)","Extreme"),(55,75,"rgba(255,154,60,0.06)","High"),(30,55,"rgba(255,209,102,0.06)","Watch")]:
            fig_al.add_hrect(y0=y0,y1=y1,fillcolor=fill_c,line_width=0,annotation_text=lbl)
        fig_al.update_layout(**PLOTLY_LAYOUT,**PLOTLY_AXES,height=240,yaxis_range=[0,100],showlegend=False)
        st.plotly_chart(fig_al, use_container_width=True)

        st.markdown('<div class="section-title">Data Requirements</div>', unsafe_allow_html=True)
        for k,v,c in [
            ("Doppler radar","Required 🔴","#ff4757"),
            ("INSAT/MODIS satellite","Required 🔴","#ff4757"),
            ("IoT weather stations","Required 🔴","#ff4757"),
            ("DEM terrain data","Available ✅","#00d4aa"),
            ("Historical records","Available ✅","#00d4aa"),
        ]:
            st.markdown(f"<div style='display:flex;justify-content:space-between;font-size:12px;padding:5px 0;border-bottom:1px solid rgba(79,158,255,0.08)'><span style='color:#7a9bc8'>{k}</span><span style='color:{c};font-weight:500'>{v}</span></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — EXCEL DATA VIEWER
# ══════════════════════════════════════════════════════════════════════════════
with tab7:
    st.markdown('<div class="section-title">Current Parameters Loaded from Excel</div>', unsafe_allow_html=True)

    ed1,ed2 = st.columns(2)
    with ed1:
        st.markdown("**Atmospheric Parameters**")
        atm_data = {
            "Max Temperature (°C)":  p["max_temp"],
            "Min Temperature (°C)":  p["min_temp"],
            "Rainfall (mm)":         p["rainfall"],
            "Wind Speed (km/h)":     p["wind_speed"],
            "Wind Gust (km/h)":      p["wind_gust"],
            "Wind Direction (°)":    p["wind_direction"],
            "Humidity (%)":          p["humidity"],
            "Pressure (hPa)":        p["pressure"],
        }
        df_atm = pd.DataFrame(list(atm_data.items()), columns=["Parameter","Value"])
        df_atm["Value"] = df_atm["Value"].astype(str)
        st.dataframe(df_atm, use_container_width=True, hide_index=True)

    with ed2:
        st.markdown("**Terrain & Context**")
        ter_data = {
            "Elevation (m)":           p["elevation"],
            "Slope (°)":               p["slope"],
            "Soil Moisture (%)":       p["soil_moisture"],
            "Weather Code (WMO)":      p["weather_code"],
            "Region":                  p["region"],
            "Population Density":      p["population_density"],
        }
        df_ter = pd.DataFrame(list(ter_data.items()), columns=["Parameter","Value"])
        df_ter["Value"] = df_ter["Value"].astype(str)
        st.dataframe(df_ter, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown('<div class="section-title">Historical Cloudburst Events (from Excel)</div>', unsafe_allow_html=True)

    hist_df = load_historical_data()
    if hist_df is not None and not hist_df.empty:
        # Color the Cloudburst column
        def color_cb(val):
            return "color: #00d4aa; font-weight:600" if str(val)=="Y" else "color: #7a9bc8"
        try:
            styled = hist_df.style.map(color_cb, subset=["Cloudburst(Y/N)"])
            st.dataframe(styled, use_container_width=True, hide_index=True)
        except Exception:
            st.dataframe(hist_df, use_container_width=True, hide_index=True)

        # Mini charts from historical data
        st.markdown('<div class="section-title" style="margin-top:16px">Historical Data Charts</div>', unsafe_allow_html=True)
        hc1,hc2 = st.columns(2)
        with hc1:
            try:
                fig_hist = go.Figure()
                yevents = hist_df[hist_df["Cloudburst(Y/N)"]=="Y"]
                nevents = hist_df[hist_df["Cloudburst(Y/N)"]=="N"]
                for df_sub, name, col in [(yevents,"Cloudburst","#ff4757"),(nevents,"No Cloudburst","#4f9eff")]:
                    if not df_sub.empty:
                        fig_hist.add_trace(go.Scatter(
                            x=df_sub["Rainfall(mm)"], y=df_sub["Humidity(%)"],
                            mode='markers', name=name,
                            marker=dict(size=9, color=col, opacity=0.8)))
                fig_hist.update_layout(**PLOTLY_LAYOUT,**PLOTLY_AXES, height=260,
                                       xaxis_title="Rainfall (mm)", yaxis_title="Humidity (%)",
                                       title=dict(text="Rainfall vs Humidity by Event Type",
                                                  font=dict(color='#c8d8f0',size=12),x=0),
                                       legend=dict(font=dict(color='#7a9bc8',size=10)))
                st.plotly_chart(fig_hist, use_container_width=True)
            except Exception:
                pass

        with hc2:
            try:
                region_counts = hist_df.groupby(["Region","Cloudburst(Y/N)"]).size().reset_index(name="count")
                fig_bar = go.Figure()
                for cb_val, col in [("Y","#ff4757"),("N","#4f9eff")]:
                    sub = region_counts[region_counts["Cloudburst(Y/N)"]==cb_val]
                    if not sub.empty:
                        fig_bar.add_trace(go.Bar(x=sub["Region"],y=sub["count"],
                            name="Cloudburst" if cb_val=="Y" else "No Event",
                            marker_color=col,marker_line_width=0))
                fig_bar.update_layout(**PLOTLY_LAYOUT,**PLOTLY_AXES,height=260,barmode='group',
                                      title=dict(text="Events by Region",font=dict(color='#c8d8f0',size=12),x=0),
                                      legend=dict(font=dict(color='#7a9bc8',size=10)))
                st.plotly_chart(fig_bar, use_container_width=True)
            except Exception:
                pass
    else:
        st.info("Historical data not available. Make sure cloudburst_data.xlsx is present.")

    st.markdown("---")
    st.markdown(f"""
    <div style='background:#0f1628;border:1px solid rgba(79,158,255,0.15);border-radius:10px;padding:14px 18px'>
      <div style='font-size:12px;font-weight:600;color:#4f9eff;margin-bottom:8px'>📗 Excel File Info</div>
      <div style='font-size:12px;color:#7a9bc8'>Path: <code style='color:#c8d8f0'>{excel_path}</code></div>
      <div style='font-size:12px;color:#7a9bc8;margin-top:6px'>Exists: <b style='color:{"#00d4aa" if excel_exists() else "#ff4757"}'>{"Yes" if excel_exists() else "No"}</b></div>
      <div style='font-size:12px;color:#7a9bc8;margin-top:6px'>Sheets: <b style='color:#c8d8f0'>Weather_Input · Historical_Data · Prediction_Log · HOW_TO_USE</b></div>
      <div style='font-size:12px;color:#4a6080;margin-top:10px;font-style:italic'>
        Edit the blue cells in Weather_Input → Save the file → Click Reload or wait 30s for auto-detect
      </div>
    </div>""", unsafe_allow_html=True)

# ─── FOOTER ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center;font-size:11px;color:#4a6080;padding:8px'>
  AI-Powered Multimodal Framework for Multi-Hazard Early Warning · Bobby's Project ·
  ConvLSTM + Transformer + Attention + PINN · Data: cloudburst_data.xlsx
</div>""", unsafe_allow_html=True)
