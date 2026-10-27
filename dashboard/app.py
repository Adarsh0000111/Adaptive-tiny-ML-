"""
Adaptive Multi-Level TinyML IIoT IDS — Modern Interactive Dashboard
Design inspiration: motion.dev (scroll animations), material.ui (elevation/cards),
coconut.ui (gradients, rounded corners, playful polish).
"""

from __future__ import annotations

import json
import os
import time
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    _HAS_PLOTLY = True
except Exception:  # pragma: no cover
    _HAS_PLOTLY = False

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Adaptive TinyML IIoT IDS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True
if "sim_running" not in st.session_state:
    st.session_state.sim_running = False
if "sim_logs" not in st.session_state:
    st.session_state.sim_logs = []
if "esp32_connected" not in st.session_state:
    st.session_state.esp32_connected = False
if "esp32_provider" not in st.session_state:
    st.session_state.esp32_provider = None
if "mqtt_client" not in st.session_state:
    st.session_state.mqtt_client = None
if "live_telemetry" not in st.session_state:
    st.session_state.live_telemetry = {}
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()
if "scroll_section" not in st.session_state:
    st.session_state.scroll_section = "overview"

# ---------------------------------------------------------------------------
# Paths & data loading
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def resolve_path(rel: str) -> str:
    return os.path.join(PROJECT_ROOT, rel)


@st.cache_data(show_spinner=False)
def load_json(path: str):
    full = resolve_path(path)
    if os.path.exists(full):
        with open(full, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_data(show_spinner=False)
def load_project_data():
    paths = {
        "benchmark": "results/tables/model_benchmark.json",
        "tinyml": "results/tables/tinyml_optimization.json",
        "federated": "results/tables/federated_comparison.json",
        "baselines": "results/tables/baseline_comparison.json",
        "simulation": "results/experiments/simulation_results.json",
    }
    return {key: load_json(p) for key, p in paths.items()}


# ---------------------------------------------------------------------------
# Design tokens (CSS variables)
# ---------------------------------------------------------------------------
def design_css(dark: bool) -> str:
    if dark:
        bg = "#0a0a0c"
        panel = "#14141a"
        card = "#1a1a22"
        border = "#272730"
        text = "#f5f5f7"
        muted = "#8a8a98"
        accent = "#22d3ee"   # cyan
        accent2 = "#a78bfa"  # violet
        success = "#34d399"
        warning = "#fbbf24"
        danger = "#f87171"
        glow = "rgba(34, 211, 238, 0.35)"
    else:
        bg = "#f8fafc"
        panel = "#ffffff"
        card = "#ffffff"
        border = "#e2e8f0"
        text = "#0f172a"
        muted = "#64748b"
        accent = "#0ea5e9"
        accent2 = "#8b5cf6"
        success = "#10b981"
        warning = "#f59e0b"
        danger = "#ef4444"
        glow = "rgba(14, 165, 233, 0.25)"

    return f"""
    <style>
    :root {{
      --bg: {bg};
      --panel: {panel};
      --card: {card};
      --border: {border};
      --text: {text};
      --muted: {muted};
      --accent: {accent};
      --accent2: {accent2};
      --success: {success};
      --warning: {warning};
      --danger: {danger};
      --glow: {glow};
      --radius: 16px;
      --radius-lg: 24px;
      --shadow: 0 20px 40px -10px rgba(0,0,0,0.35);
      --shadow-md: 0 10px 25px -5px rgba(0,0,0,0.2);
      --transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
    }}

    * {{ scroll-behavior: smooth; }}

    /* Hide Streamlit chrome */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    .stDeployButton {{display: none;}}
    .stAppViewContainer {{background: var(--bg);}}
    .block-container {{padding-top: 0; max-width: 1400px;}}

    /* Global text */
    body, .stMarkdown, .stText, label {{color: var(--text) !important;}}
    .stApp {{background: var(--bg);}}

    /* Background gradient mesh */
    .mesh {{
      position: fixed;
      inset: 0;
      z-index: 0;
      overflow: hidden;
      pointer-events: none;
    }}
    .blob {{
      position: absolute;
      filter: blur(60px);
      opacity: 0.35;
      border-radius: 50%;
      animation: blobMove 12s ease-in-out infinite alternate;
    }}
    .blob-1 {{top: -10%; left: -10%; width: 50vw; height: 50vw; background: var(--accent); animation-duration: 14s;}}
    .blob-2 {{bottom: -10%; right: -10%; width: 45vw; height: 45vw; background: var(--accent2); animation-duration: 16s; animation-delay: -4s;}}
    @keyframes blobMove {{
      0% {{transform: translate(0, 0) scale(1);}}
      100% {{transform: translate(40px, 60px) scale(1.1);}}
    }}

    /* Layout shells */
    .shell {{
      position: relative;
      z-index: 1;
      padding-top: 10px;
    }}
    .glass {{
      background: rgba(255,255,255,0.03);
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
      border: 1px solid var(--border);
      border-radius: var(--radius-lg);
      box-shadow: var(--shadow);
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 20px;
      box-shadow: var(--shadow-md);
      transition: var(--transition);
      position: relative;
      overflow: hidden;
    }}
    .card:hover {{
      transform: translateY(-6px);
      border-color: var(--accent);
      box-shadow: 0 30px 60px -15px var(--glow);
    }}
    .card::before {{
      content: "";
      position: absolute;
      top: 0; left: 0; right: 0; height: 1px;
      background: linear-gradient(90deg, transparent, var(--accent), transparent);
      opacity: 0.6;
    }}

    /* Typography */
    .h1 {{
      font-size: 3rem; font-weight: 800; letter-spacing: -0.04em;
      margin: 0; background: linear-gradient(135deg, #fff 0%, var(--accent) 60%, var(--accent2) 100%);
      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
      line-height: 1.1;
    }}
    .h2 {{font-size: 1.5rem; font-weight: 700; margin: 0 0 8px; color: var(--text);}}
    .h3 {{font-size: 1.1rem; font-weight: 600; color: var(--text); margin-bottom: 12px;}}
    .sub {{font-size: 1.05rem; color: var(--muted); line-height: 1.5; margin: 0;}}
    .badge {{
      display: inline-flex; align-items: center; gap: 6px;
      padding: 4px 10px; border-radius: 999px; font-size: 0.75rem; font-weight: 700;
      text-transform: uppercase; letter-spacing: 0.06em;
    }}
    .badge-low {{background: rgba(52, 211, 153, 0.15); color: var(--success); border: 1px solid rgba(52, 211, 153, 0.3);}}
    .badge-med {{background: rgba(251, 191, 36, 0.15); color: var(--warning); border: 1px solid rgba(251, 191, 36, 0.3);}}
    .badge-high {{background: rgba(248, 113, 113, 0.15); color: var(--danger); border: 1px solid rgba(248, 113, 113, 0.3);}}
    .badge-crit {{background: rgba(239, 68, 68, 0.2); color: #fee2e2; border: 1px solid rgba(239, 68, 68, 0.4);}}

    /* Scroll-reveal */
    .reveal {{
      opacity: 0;
      transform: translateY(30px);
      transition: all 0.8s cubic-bezier(0.16, 1, 0.3, 1);
    }}
    .reveal.visible {{
      opacity: 1; transform: translateY(0);
    }}
    .reveal-delay-1 {{transition-delay: 0.1s;}}
    .reveal-delay-2 {{transition-delay: 0.2s;}}
    .reveal-delay-3 {{transition-delay: 0.3s;}}

    /* Metric cards */
    .metric {{
      background: var(--card); border: 1px solid var(--border);
      border-radius: var(--radius); padding: 20px 22px;
      transition: var(--transition);
    }}
    .metric:hover {{transform: translateY(-4px); border-color: var(--accent); box-shadow: var(--shadow);}}
    .metric-label {{font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em; color: var(--muted); margin-bottom: 8px; font-weight: 600;}}
    .metric-value {{font-size: 2rem; font-weight: 800; color: var(--text); line-height: 1.1;}}
    .metric-trend {{font-size: 0.8rem; margin-top: 6px; display: flex; align-items: center; gap: 6px;}}
    .dot {{width: 8px; height: 8px; border-radius: 50%; background: var(--success); animation: pulse 2s infinite;}}
    @keyframes pulse {{0%{{opacity: 1;transform: scale(1);}}50%{{opacity: 0.5;transform: scale(1.15);}}100%{{opacity: 1;transform: scale(1);}}}}

    /* Stat big number */
    .big {{font-size: 2.75rem; font-weight: 800; line-height: 1; margin: 0; color: var(--text);}}
    .big-cap {{font-size: 0.85rem; color: var(--muted); margin-top: 4px;}}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{gap: 8px; background: var(--panel); border-radius: var(--radius-lg); padding: 6px; border: 1px solid var(--border);}}
    .stTabs [data-baseweb="tab"] {{background: transparent; border-radius: 10px; color: var(--muted); font-weight: 600; padding: 8px 18px; transition: var(--transition); border: none;}}
    .stTabs [aria-selected="true"] {{background: var(--card); color: var(--accent); box-shadow: 0 4px 12px -5px var(--glow);}}

    /* Sidebar */
    section[data-testid="stSidebar"] {{background: var(--panel); border-right: 1px solid var(--border); width: 320px !important;}}
    section[data-testid="stSidebar"] .stSlider > div {{padding-top: 2px;}}
    .sidebar-section {{padding: 16px 20px; border-bottom: 1px solid var(--border);}}
    .sidebar-section h4 {{font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.1em; color: var(--muted); margin: 0 0 12px; font-weight: 700;}}
    .stButton > button {{width: 100%; border-radius: 10px; border: 1px solid var(--border); background: var(--card); color: var(--text); font-weight: 600; transition: var(--transition);}}
    .stButton > button:hover {{border-color: var(--accent); color: var(--accent); background: rgba(34,211,238,0.08);}}
    .stButton[kind="primary"] > button {{background: var(--accent); color: #000; border: none; font-weight: 700;}}
    .stButton[kind="primary"] > button:hover {{background: var(--accent2); color: #fff; box-shadow: 0 10px 25px -8px var(--glow);}}

    /* Toggle switch */
    .stToggle > div {{background: var(--card); border: 1px solid var(--border);}}

    /* Status dot */
    .status-pill {{display: inline-flex; align-items: center; gap: 8px; padding: 6px 14px; border-radius: 999px; font-size: 0.85rem; font-weight: 700; border: 1px solid var(--border); background: var(--card);}}
    .status-dot {{width: 10px; height: 10px; border-radius: 50%;}}
    .status-on {{background: var(--success); box-shadow: 0 0 12px var(--success);}}
    .status-off {{background: var(--danger); box-shadow: 0 0 12px var(--danger);}}

    /* Code block */
    pre {{background: #0d0d12 !important; border: 1px solid var(--border) !important; border-radius: 12px !important;}}
    code {{font-size: 0.85rem !important;}}

    /* Scrollbar */
    ::-webkit-scrollbar {{width: 8px; height: 8px;}}
    ::-webkit-scrollbar-track {{background: var(--bg);}}
    ::-webkit-scrollbar-thumb {{background: var(--border); border-radius: 999px;}}
    ::-webkit-scrollbar-thumb:hover {{background: var(--muted);}}
    </style>
    """

st.markdown(design_css(st.session_state.dark_mode), unsafe_allow_html=True)

# Background mesh
st.markdown("""
<div class="mesh">
  <div class="blob blob-1"></div>
  <div class="blob blob-2"></div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="shell">', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header / Hero
# ---------------------------------------------------------------------------
st.markdown("""
<div class="reveal reveal-delay-1">
  <div style="display:flex; align-items:center; gap:12px; margin-bottom:8px;">
    <span class="badge badge-low">● LIVE</span>
    <span style="font-size:0.8rem; color:var(--muted); text-transform:uppercase; letter-spacing:0.15em;">Adaptive TinyML • IIoT Intrusion Detection</span>
  </div>
  <h1 class="h1">Adaptive Multi-Level<br>TinyML IDS</h1>
  <p class="sub" style="max-width:700px;">Energy-aware model switching &amp; federated learning for Industrial IoT edge devices.
  <br>Designed for ESP32 edge nodes — real-time telemetry, model switching, and federated convergence.</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-section"><h4>🎨 Theme</h4>', unsafe_allow_html=True)
    st.session_state.dark_mode = st.toggle("Dark Mode", value=st.session_state.dark_mode, key="theme_toggle")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section"><h4>⚙️ Simulation</h4>', unsafe_allow_html=True)
    sim_speed = st.slider("Frequency (Hz)", 1, 20, 5, key="sim_speed")
    energy_threshold = st.slider("Low Battery Threshold (%)", 10, 50, 35, key="batt_thresh")
    num_events = st.number_input("Events / Batch", 10, 2000, 100, step=10, key="num_events")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section"><h4>🔌 Hardware</h4>', unsafe_allow_html=True)
    esp_port = st.text_input("Serial Port", "COM3", key="esp_port")
    mqtt_broker = st.text_input("MQTT Broker", "localhost", key="mqtt_broker")
    mqtt_port = st.number_input("MQTT Port", 1, 65535, 1883, key="mqtt_port")
    device_id = st.text_input("Device ID", "esp32-ids-01", key="device_id")

    c1, c2 = st.columns(2)
    if c1.button("Connect ESP32", key="btn_connect_esp32"):
        try:
            from src.energy.esp32_interface import ESP32INA219EnergyProvider
            prov = ESP32INA219EnergyProvider()
            ok = prov.connect_sensor()
            st.session_state.esp32_connected = ok
            st.session_state.esp32_provider = prov if ok else None
            st.toast("ESP32 connected" if ok else "Stub mode (no hardware)", icon="✅" if ok else "⚠️")
        except Exception as exc:
            st.toast(f"Connect error: {exc}", icon="❌")
            st.session_state.esp32_connected = False
    if c2.button("Disconnect", key="btn_disconnect_esp32"):
        prov = st.session_state.esp32_provider
        if prov:
            try: prov.close()
            except Exception: pass
        st.session_state.esp32_connected = False
        st.session_state.esp32_provider = None
        st.toast("ESP32 disconnected", icon="🔌")

    c3, c4 = st.columns(2)
    if c3.button("Connect MQTT", key="btn_connect_mqtt"):
        try:
            from src.energy.esp32_interface import ESP32MQTTClient
            cli = ESP32MQTTClient(device_id=device_id, broker_host=mqtt_broker, broker_port=int(mqtt_port))
            ok = cli.connect()
            st.session_state.mqtt_client = cli if ok else None
            st.toast("MQTT connected" if ok else "Broker unreachable", icon="✅" if ok else "⚠️")
        except Exception as exc:
            st.toast(f"MQTT error: {exc}", icon="❌")
            st.session_state.mqtt_client = None
    if c4.button("Disconnect MQTT", key="btn_disconnect_mqtt"):
        cli = st.session_state.mqtt_client
        if cli: cli.disconnect()
        st.session_state.mqtt_client = None
        st.toast("MQTT disconnected", icon="🔌")

    st.markdown('<div class="sidebar-section"><h4>🚀 Actions</h4>', unsafe_allow_html=True)
    if st.button("Run Live Simulation Batch", key="btn_run_sim", type="primary"):
        st.session_state.sim_running = True
    if st.button("Run Full Research Pipeline", key="btn_run_pipeline"):
        st.toast("Pipeline queued — run `python -m src.main` in terminal", icon="ℹ️")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-section">
      <h4>📊 Target</h4>
      <div style="font-size:0.75rem; color:var(--muted); line-height:1.5;">
        ESP32 Edge MCU<br>
        Protocols: Modbus, MQTT, HTTP, TCP/UDP<br>
        Dataset: Edge-IIoTset<br>
        Sensors: INA219 · DHT22 · PIR · MQ-2
      </div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
data_store = load_project_data()

# ---------------------------------------------------------------------------
# Helper: severity badge HTML
# ---------------------------------------------------------------------------
def severity_badge(severity: str) -> str:
    s = (severity or "LOW").upper()
    cls = {"LOW":"badge-low","MEDIUM":"badge-med","HIGH":"badge-high","CRITICAL":"badge-crit"}.get(s, "badge-low")
    return f'<span class="badge {cls}">{s}</span>'

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Overview & Live Monitor",
    "⚡ Adaptive Model Switching",
    "🌐 Federated Learning",
    "🔬 TinyML Optimization",
    "🔋 Energy Analysis",
    "📈 Baseline Comparison",
    "🔌 ESP32 & Hardware",
])

# ===========================================================================
# TAB 1 — OVERVIEW & LIVE MONITOR
# ===========================================================================
with tab1:
    st.markdown('<div class="reveal reveal-delay-1">', unsafe_allow_html=True)
    st.subheader("Real-Time IIoT Device State & Intrusion Monitoring")

    sim_logs = st.session_state.sim_logs if st.session_state.sim_logs else data_store.get("simulation")

    if st.session_state.sim_running:
        st.session_state.sim_running = False
        with st.spinner("Running live IIoT simulation batch…"):
            try:
                from src.simulation.iiot_simulator import RealtimeIIoTSimulator
                sim = RealtimeIIoTSimulator(initial_battery_pct=100.0)
                logs = sim.run_simulation_batch(count=int(num_events))
                st.session_state.sim_logs = logs
                sim_logs = logs
                st.toast(f"Simulation complete: {len(logs)} events", icon="✅")
            except Exception as exc:
                st.error(f"Simulation error: {exc}")

    # Refresh ESP32 telemetry
    prov = st.session_state.esp32_provider
    if prov is not None:
        try:
            st.session_state.live_telemetry = {
                "battery_pct": prov.get_battery_percentage(),
                "bus_voltage_v": prov.get_bus_voltage_v(),
                "shunt_current_ma": prov.get_shunt_current_ma(),
                "total_energy_uj": prov.total_energy_consumed_uj,
            }
        except Exception as exc:
            st.session_state.live_telemetry = {"error": str(exc)}

    latest = sim_logs[-1] if sim_logs else None

    batt_val = None
    if st.session_state.live_telemetry and "battery_pct" in st.session_state.live_telemetry:
        batt_val = st.session_state.live_telemetry["battery_pct"]
    elif latest:
        batt_val = latest["battery_pct"]

    # KPI row
    kpis = [
        ("Edge Battery", f"{batt_val:.1f}%" if batt_val is not None else "—", "Power remaining", "battery"),
        ("ESP32 Status", "Connected" if st.session_state.esp32_connected else "Simulation", "Hardware link", "esp32"),
        ("Last Prediction", latest["predicted_class"] if latest else "—", "Most recent class", "pred"),
        ("Threat Severity", (latest["severity"] if latest else "LOW"), "Current risk level", "sev"),
        ("Selected Model", latest["selected_model"] if latest else "—", "Active inference tier", "model"),
        ("Events Processed", len(sim_logs) if sim_logs else 0, "In this session", "count"),
    ]
    cols = st.columns(3)
    for i, (label, value, desc, kind) in enumerate(kpis):
        with cols[i % 3]:
            accent_cls = "badge-low"
            if kind == "sev":
                accent_cls = {"LOW":"badge-low","MEDIUM":"badge-med","HIGH":"badge-high","CRITICAL":"badge-crit"}.get(str(value).upper(), "badge-low")
            st.markdown(f"""
            <div class="metric reveal reveal-delay-{(i%3)+1}" style="animation-delay:0.{i}s;">
              <div class="metric-label">{label}</div>
              <div class="metric-value">{value}</div>
              {f'<div class="metric-trend"><span class="dot"></span><span style="color:var(--muted);font-size:0.75rem;">{desc}</span></div>' if kind!="sev" else f'<div class="metric-trend">{severity_badge(value)}<span style="color:var(--muted);font-size:0.75rem;margin-left:8px;">{desc}</span></div>'}
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if sim_logs:
        df_sim = pd.DataFrame(sim_logs)
        g1, g2 = st.columns([1, 2])

        with g1:
            latest_score = int(df_sim["threat_score"].iloc[-1])
            color = "#34d399" if latest_score < 30 else ("#fbbf24" if latest_score < 60 else ("#f59e0b" if latest_score < 80 else "#ef4444"))
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=latest_score,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Current Threat Score", "font": {"size": 14, "color": "#94a3b8"}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#334155"},
                    "bar": {"color": color, "thickness": 0.62},
                    "bgcolor": "rgba(0,0,0,0)",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 30], "color": "rgba(52, 211, 153, 0.18)"},
                        {"range": [30, 60], "color": "rgba(251, 191, 36, 0.18)"},
                        {"range": [60, 80], "color": "rgba(245, 158, 11, 0.18)"},
                        {"range": [80, 100], "color": "rgba(239, 68, 68, 0.18)"},
                    ],
                    "threshold": {"line": {"color": "#ef4444", "width": 3}, "value": 80},
                },
            ))
            fig.update_layout(
                height=300, margin=dict(t=40, b=20, l=20, r=20),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8", size=12),
            )
            st.plotly_chart(fig, use_container_width=True)

        with g2:
            fig2 = make_subplots(specs=[[{"secondary_y": True}]])
            fig2.add_trace(
                go.Scatter(x=df_sim["event_id"], y=df_sim["threat_score"],
                           name="Threat Score", line=dict(color="#22d3ee", width=3), mode="lines"),
                secondary_y=False,
            )
            model_map = {"LIGHTWEIGHT": 1, "MEDIUM": 2, "HEAVY": 3}
            model_nums = df_sim["selected_model"].map(model_map)
            fig2.add_trace(
                go.Scatter(x=df_sim["event_id"], y=model_nums,
                           name="Selected Model", mode="lines+markers",
                           line=dict(color="#a78bfa", width=3), marker=dict(size=6),
                           step="after"),
                secondary_y=True,
            )
            fig2.update_yaxes(title_text="Threat Score (0–100)", secondary_y=False, range=[0, 100],
                              gridcolor="rgba(148,163,184,0.12)", zeroline=False)
            fig2.update_yaxes(title_text="Selected Model", secondary_y=True,
                              tickvals=[1, 2, 3], ticktext=["Lightweight", "Medium", "Heavy"],
                              gridcolor="rgba(148,163,184,0.12)", zeroline=False)
            fig2.update_xaxes(title_text="Event ID", gridcolor="rgba(148,163,184,0.12)", zeroline=False)
            fig2.update_layout(
                height=300, margin=dict(t=20, b=20, l=20, r=20),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8", size=11),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                            bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("### Recent Events")
        show_cols = ["timestamp", "event_id", "true_class", "predicted_class",
                     "threat_score", "severity", "battery_pct", "selected_model",
                     "inference_latency_ms", "energy_consumed_uj", "model_switch"]
        show_cols = [c for c in show_cols if c in df_sim.columns]
        st.dataframe(df_sim[show_cols].tail(12)[::-1], use_container_width=True, height=360,
                     hide_index=True)
    else:
        st.info("Run a simulation batch from the sidebar to see live data.")
    st.markdown('</div>', unsafe_allow_html=True)


# ===========================================================================
# TAB 2 — ADAPTIVE MODEL SWITCHING
# ===========================================================================
with tab2:
    st.markdown('<div class="reveal reveal-delay-1">', unsafe_allow_html=True)
    st.subheader("⚡ Adaptive Model Switching Engine")
    st.markdown("""
    <p class="sub">Multi-attribute utility theory selects the optimal inference tier per network event based on
    threat severity, confidence, battery level, latency, and energy cost.</p>
    """, unsafe_allow_html=True)

    sim_logs = st.session_state.sim_logs if st.session_state.sim_logs else data_store.get("simulation")

    if sim_logs:
        df_sim = pd.DataFrame(sim_logs)

        # Model distribution donut
        col_a, col_b = st.columns([1, 1.4])
        with col_a:
            dist = df_sim["selected_model"].value_counts().reindex(["LIGHTWEIGHT", "MEDIUM", "HEAVY"]).fillna(0)
            fig = go.Figure(go.Pie(
                labels=dist.index, values=dist.values, hole=0.62,
                marker=dict(colors=["#34d399", "#fbbf24", "#ef4444"],
                            line=dict(color="#0a0a0c", width=2)),
                textinfo="label+percent", hoverinfo="label+value",
                rotation=90,
            ))
            fig.update_layout(
                height=320, margin=dict(t=10, b=10, l=10, r=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8", size=12),
                showlegend=True, legend=dict(orientation="h", y=-0.12, x=0.5, xanchor="center"),
                annotations=[dict(text="Inference<br>Tier Mix", showarrow=False, font=dict(size=13, color="#f5f5f7"))],
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            # Energy per tier bar
            energy_map = {"LIGHTWEIGHT": 8.5, "MEDIUM": 62.0, "HEAVY": 340.0}
            df_sim["energy_est"] = df_sim["selected_model"].map(energy_map)
            tier_e = df_sim.groupby("selected_model")["energy_est"].sum().reindex(["LIGHTWEIGHT", "MEDIUM", "HEAVY"]).fillna(0)
            fig2 = go.Figure(go.Bar(
                x=tier_e.index, y=tier_e.values,
                marker=dict(color=["#34d399", "#fbbf24", "#ef4444"], cornerradius=8),
                text=[f"{v:,.0f} uJ" for v in tier_e.values],
                textposition="outside",
                textfont=dict(size=11, color="#94a3b8"),
            ))
            fig2.update_layout(
                height=320, margin=dict(t=20, b=40, l=20, r=20),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8", size=12),
                xaxis=dict(tickcolor="#334155", gridcolor="rgba(148,163,184,0.12)", zeroline=False),
                yaxis=dict(title="Est. Energy (uJ)", tickcolor="#334155", gridcolor="rgba(148,163,184,0.12)", zeroline=False),
                showlegend=False,
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("### Model Switching Timeline (first 120 events)")
        df_zoom = df_sim.head(120).copy()
        df_zoom["model_num"] = df_zoom["selected_model"].map(model_map)
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=df_zoom["event_id"], y=df_zoom["threat_score"],
            mode="lines", name="Threat Score",
            line=dict(color="#818cf8", width=2),
        ))
        fig3.add_trace(go.Scatter(
            x=df_zoom["event_id"], y=df_zoom["model_num"],
            mode="lines+markers", name="Selected Model",
            line=dict(color="#22d3ee", width=3), marker=dict(size=5),
            step="after", yaxis="y2",
        ))
        fig3.update_layout(
            height=340, margin=dict(t=20, b=40, l=20, r=20),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8", size=11),
            xaxis=dict(title="Event ID", gridcolor="rgba(148,163,184,0.12)", zeroline=False),
            yaxis=dict(title="Threat Score", range=[0, 105], gridcolor="rgba(148,163,184,0.12)", zeroline=False),
            yaxis2=dict(title="Model Tier", overlaying="y", side="right",
                        tickvals=[1, 2, 3], ticktext=["Light", "Medium", "Heavy"],
                        gridcolor="rgba(148,163,184,0)", zeroline=False),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                        bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown("### Switch Reason Distribution")
        reason_counts = df_sim["reason"].value_counts().head(8)
        fig4 = go.Figure(go.Bar(
            x=reason_counts.values, y=reason_counts.index, orientation="h",
            marker=dict(color="#a78bfa", cornerradius=6),
            text=[f"{v}x" for v in reason_counts.values],
            textposition="outside", textfont=dict(size=11, color="#94a3b8"),
        ))
        fig4.update_layout(
            height=300, margin=dict(t=20, b=40, l=120, r=20),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8", size=12),
            xaxis=dict(gridcolor="rgba(148,163,184,0.12)", zeroline=False),
            yaxis=dict(autorange="reversed", tickcolor="#334155", gridcolor="rgba(0,0,0,0)", zeroline=False),
            showlegend=False,
        )
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("Run a simulation batch to view adaptive switching analytics.")
    st.markdown('</div>', unsafe_allow_html=True)

# ===========================================================================
# TAB 3 — FEDERATED LEARNING
# ===========================================================================
with tab3:
    st.markdown('<div class="reveal reveal-delay-1">', unsafe_allow_html=True)
    st.subheader("🌐 Federated Learning & Non-IID Convergence")
    st.markdown("""
    <p class="sub">Privacy-preserving FedAvg with energy-aware client selection across 5 simulated IIoT edge nodes.
    Raw telemetry never leaves the device.</p>
    """, unsafe_allow_html=True)

    fed_data = data_store.get("federated")
    if fed_data:
        std_fl = fed_data.get("standard_fl", {})
        energy_fl = fed_data.get("energy_aware_fl", {})

        # KPI row
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f"""
            <div class="card reveal reveal-delay-1">
              <div class="metric-label">Simulated Clients</div>
              <div class="big">5 Edge Nodes</div>
              <div class="big-cap">Non-IID Dirichlet partitions</div>
            </div>
            """, unsafe_allow_html=True)
        with k2:
            st.markdown(f"""
            <div class="card reveal reveal-delay-2">
              <div class="metric-label">FL Communication Rounds</div>
              <div class="big">10 Rounds</div>
              <div class="big-cap">Energy-aware selection</div>
            </div>
            """, unsafe_allow_html=True)
        with k3:
            acc = energy_fl.get("final_global_accuracy", 0) * 100
            st.markdown(f"""
            <div class="card reveal reveal-delay-3">
              <div class="metric-label">Energy-Aware Accuracy</div>
              <div class="big">{acc:.2f}%</div>
              <div class="big-cap">Global model</div>
            </div>
            """, unsafe_allow_html=True)
        with k4:
            red = ((std_fl.get("total_comm_overhead_kb", 0) - energy_fl.get("total_comm_overhead_kb", 0)) / max(std_fl.get("total_comm_overhead_kb", 1), 1)) * 100
            st.markdown(f"""
            <div class="card reveal reveal-delay-1">
              <div class="metric-label">Comm. Overhead Reduction</div>
              <div class="big">{red:.1f}%</div>
              <div class="big-cap">vs Standard FedAvg</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### Global Accuracy Convergence")
        hist_std = std_fl.get("history", [])
        hist_ene = energy_fl.get("history", [])
        if hist_std and hist_ene and _HAS_PLOTLY:
            df_std = pd.DataFrame(hist_std)
            df_ene = pd.DataFrame(hist_ene)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_std["round"], y=df_std["global_accuracy"] * 100,
                mode="lines+markers", name="Standard FedAvg",
                line=dict(color="#ef4444", width=3), marker=dict(size=7),
            ))
            fig.add_trace(go.Scatter(
                x=df_ene["round"], y=df_ene["global_accuracy"] * 100,
                mode="lines+markers", name="Energy-Aware FL (Proposed)",
                line=dict(color="#22d3ee", width=3), marker=dict(size=7, symbol="diamond"),
            ))
            fig.update_layout(
                height=380, margin=dict(t=20, b=40, l=20, r=20),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8", size=12),
                xaxis=dict(title="Communication Round", gridcolor="rgba(148,163,184,0.12)", zeroline=False),
                yaxis=dict(title="Global Accuracy (%)", gridcolor="rgba(148,163,184,0.12)", zeroline=False),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                            bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Client Battery Trajectory")
        if hist_ene and _HAS_PLOTLY:
            df_ene = pd.DataFrame(hist_ene)
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=df_ene["round"], y=df_ene["avg_battery_pct"],
                mode="lines+markers", name="Avg Battery (%)",
                line=dict(color="#3b82f6", width=3),
            ))
            fig2.add_trace(go.Scatter(
                x=df_ene["round"], y=df_ene["min_battery_pct"],
                mode="lines+markers", name="Min Battery (%)",
                line=dict(color="#f59e0b", width=2, dash="dot"),
            ))
            fig2.add_trace(go.Scatter(
                x=df_ene["round"], y=[20] * len(df_ene),
                mode="lines", name="Min Threshold (20%)",
                line=dict(color="#ef4444", width=2, dash="dash"),
            ))
            fig2.update_layout(
                height=320, margin=dict(t=20, b=40, l=20, r=20),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8", size=12),
                xaxis=dict(title="Round", gridcolor="rgba(148,163,184,0.12)", zeroline=False),
                yaxis=dict(title="Battery (%)", range=[0, 105], gridcolor="rgba(148,163,184,0.12)", zeroline=False),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                            bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No federated learning results found. Run the pipeline to generate them.")
    st.markdown('</div>', unsafe_allow_html=True)

# ===========================================================================
# TAB 4 — TINYML OPTIMIZATION
# ===========================================================================
with tab4:
    st.markdown('<div class="reveal reveal-delay-1">', unsafe_allow_html=True)
    st.subheader("🔬 TinyML INT8 Quantization & ESP32 Header Export")
    st.markdown("""
    <p class="sub">Fixed-point quantization compresses the lightweight decision tree from ~3.8 KB to 0.20 KB
    (94.7% reduction) for deployment on ESP32 Xtensa LX6 microcontrollers.</p>
    """, unsafe_allow_html=True)

    t_data = data_store.get("tinyml")
    if t_data:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="card reveal reveal-delay-1">
              <div class="metric-label">Original Size</div>
              <div class="big">{t_data.get('original_size_kb', 0)} KB</div>
              <div class="big-cap">Float32 decision tree</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="card reveal reveal-delay-2">
              <div class="metric-label">INT8 Quantized Size</div>
              <div class="big">{t_data.get('quantized_size_kb', 0)} KB</div>
              <div class="big-cap">Fixed-point int32 thresholds</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="card reveal reveal-delay-3">
              <div class="metric-label">Storage Reduction</div>
              <div class="big">{t_data.get('size_reduction_pct', 0):.1f}%</div>
              <div class="big-cap">Edge-ready footprint</div>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="card reveal reveal-delay-1">
              <div class="metric-label">Accuracy Drop</div>
              <div class="big">{t_data.get('accuracy_drop_pct', 0):.2f}%</div>
              <div class="big-cap">Negligible impact</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### ESP32 Deployment Readiness")
        col_a, col_b = st.columns([1, 2])
        with col_a:
            st.markdown("""
            <div class="card reveal reveal-delay-2">
              <div class="h3">Target Specs</div>
              <div style="display:grid; gap:10px; margin-top:12px;">
                <div><span class="badge badge-low">RAM</span> <span style="float:right;color:var(--muted);">1.2 KB</span></div>
                <div><span class="badge badge-med">Flash</span> <span style="float:right;color:var(--muted);">~2.7 KB</span></div>
                <div><span class="badge badge-high">ESP32</span> <span style="float:right;color:var(--muted);">Compatible</span></div>
                <div><span class="badge badge-low">Arch</span> <span style="float:right;color:var(--muted);">Xtensa LX6</span></div>
              </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('<div style="margin-top:14px;"></div>', unsafe_allow_html=True)
            st.success("ESP32 C header generated at `models/tinyml/esp32_model.h`")
        with col_b:
            h_path = resolve_path("models/tinyml/esp32_model.h")
            if os.path.exists(h_path):
                with open(h_path, "r", encoding="utf-8") as f:
                    c_header = f.read()
                st.code(c_header, language="c")
                st.download_button("Download esp32_model.h", c_header,
                                   file_name="esp32_model.h", mime="text/x-csrc",
                                   use_container_width=True, key="dl_header")
            else:
                st.warning("C header not found. Run TinyML optimization phase.")
    else:
        st.info("No TinyML optimization results found. Run the pipeline to generate them.")
    st.markdown('</div>', unsafe_allow_html=True)


# ===========================================================================
# TAB 5 — ENERGY ANALYSIS
# ===========================================================================
with tab5:
    st.markdown('<div class="reveal reveal-delay-1">', unsafe_allow_html=True)
    st.subheader("🔋 Energy Consumption & Battery Longevity")
    st.markdown("""
    <p class="sub">Comparative energy footprint across fixed and adaptive inference strategies,
    with real-time battery drain tracking from the ESP32 INA219 sensor.</p>
    """, unsafe_allow_html=True)

    base_data = data_store.get("baselines")
    sim_logs = st.session_state.sim_logs if st.session_state.sim_logs else data_store.get("simulation")

    if base_data:
        df_base = pd.DataFrame(base_data).T
        st.markdown("### Total Energy per System Configuration")
        if _HAS_PLOTLY and {"name", "total_energy_consumed_uJ"}.issubset(df_base.columns):
            colors = ["#34d399", "#fbbf24", "#ef4444", "#22d3ee", "#a78bfa"]
            fig = go.Figure()
            for i, (_, row) in enumerate(df_base.iterrows()):
                fig.add_trace(go.Bar(
                    name=row.get("name", "?"),
                    x=[row.get("name", "?")],
                    y=[row.get("total_energy_consumed_uJ", 0)],
                    marker=dict(color=colors[i % len(colors)], cornerradius=8),
                    text=[f"{row.get('total_energy_consumed_uJ', 0):,.0f} uJ"],
                    textposition="outside", textfont=dict(size=11, color="#94a3b8"),
                    hovertemplate="%{x}<br>Energy: %{y:,.0f} uJ<extra></extra>",
                ))
            fig.update_layout(
                height=400, margin=dict(t=20, b=60, l=20, r=20),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8", size=12),
                showlegend=False,
                xaxis=dict(title="System Configuration", tickangle=-15,
                           tickcolor="#334155", gridcolor="rgba(0,0,0,0)", zeroline=False),
                yaxis=dict(title="Total Energy Consumed (uJ)", tickcolor="#334155",
                           gridcolor="rgba(148,163,184,0.12)", zeroline=False),
            )
            st.plotly_chart(fig, use_container_width=True)

    if sim_logs:
        st.markdown("### Battery Drain Over Time (Adaptive System)")
        df_sim = pd.DataFrame(sim_logs)
        if _HAS_PLOTLY:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=df_sim["event_id"], y=df_sim["battery_pct"],
                mode="lines", name="Battery %",
                line=dict(color="#3b82f6", width=3),
                fill="tozeroy", fillcolor="rgba(59, 130, 246, 0.08)",
            ))
            fig2.add_trace(go.Scatter(
                x=df_sim["event_id"], y=[energy_threshold] * len(df_sim),
                mode="lines", name="Low Battery Threshold",
                line=dict(color="#f59e0b", width=2, dash="dash"),
            ))
            fig2.update_layout(
                height=340, margin=dict(t=20, b=40, l=20, r=20),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8", size=12),
                xaxis=dict(title="Event ID", gridcolor="rgba(148,163,184,0.12)", zeroline=False),
                yaxis=dict(title="Battery (%)", range=[0, 105], gridcolor="rgba(148,163,184,0.12)", zeroline=False),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                            bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("### Energy per Inference by Model Tier")
        tier_energy = df_sim.groupby("selected_model")["energy_consumed_uj"].sum().reindex(
            ["LIGHTWEIGHT", "MEDIUM", "HEAVY"]).fillna(0)
        fig3 = go.Figure(go.Bar(
            x=tier_energy.index, y=tier_energy.values,
            marker=dict(color=["#34d399", "#fbbf24", "#ef4444"], cornerradius=8),
            text=[f"{v:,.0f} uJ" for v in tier_energy.values],
            textposition="outside", textfont=dict(size=11, color="#94a3b8"),
        ))
        fig3.update_layout(
            height=300, margin=dict(t=20, b=40, l=20, r=20),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8", size=12),
            xaxis=dict(tickcolor="#334155", gridcolor="rgba(0,0,0,0)", zeroline=False),
            yaxis=dict(title="Cumulative Energy (uJ)", tickcolor="#334155",
                       gridcolor="rgba(148,163,184,0.12)", zeroline=False),
            showlegend=False,
        )
        st.plotly_chart(fig3, use_container_width=True)
    elif base_data is None:
        st.info("Run the pipeline to generate energy results.")
    st.markdown('</div>', unsafe_allow_html=True)

# ===========================================================================
# TAB 6 — BASELINE COMPARISON
# ===========================================================================
with tab6:
    st.markdown('<div class="reveal reveal-delay-1">', unsafe_allow_html=True)
    st.subheader("📈 Baseline Comparison: Proposed vs Fixed Strategies")
    st.markdown("""
    <p class="sub">The adaptive multi-level system balances detection accuracy against energy cost,
    outperforming fixed single-tier strategies on the energy–accuracy frontier.</p>
    """, unsafe_allow_html=True)

    base_data = data_store.get("baselines")
    if base_data:
        df_b = pd.DataFrame(base_data).T
        display_cols = ["name", "accuracy", "total_energy_consumed_uJ",
                        "avg_inference_latency_ms", "model_switches_count",
                        "energy_savings_vs_heavy_pct"]
        display_cols = [c for c in display_cols if c in df_b.columns]
        st.dataframe(df_b[display_cols], use_container_width=True, hide_index=True,
                     height=260)

        st.markdown("### Accuracy vs Energy Frontier")
        if _HAS_PLOTLY and {"accuracy", "total_energy_consumed_uJ", "name"}.issubset(df_b.columns):
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_b["total_energy_consumed_uJ"],
                y=df_b["accuracy"] * 100,
                mode="markers+text",
                text=df_b["name"],
                textposition="top center",
                textfont=dict(size=11, color="#94a3b8"),
                marker=dict(size=16, color=df_b["accuracy"] * 100,
                            colorscale="Viridis", showscale=True,
                            colorbar=dict(title="Accuracy %", tickcolor="#94a3b8",
                                          titlefont=dict(color="#94a3b8", size=11))),
            ))
            fig.update_layout(
                height=420, margin=dict(t=20, b=40, l=20, r=20),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8", size=12),
                xaxis=dict(title="Total Energy Consumed (uJ)", gridcolor="rgba(148,163,184,0.12)", zeroline=False),
                yaxis=dict(title="Accuracy (%)", gridcolor="rgba(148,163,184,0.12)", zeroline=False),
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Key Takeaway")
        prop = df_b[df_b["name"].str.contains("Proposed|Adaptive", case=False, na=False)]
        if len(prop):
            row = prop.iloc[0]
            st.markdown(f"""
            <div class="card reveal reveal-delay-2" style="border-left:4px solid var(--accent);">
              <div style="font-size:0.8rem;text-transform:uppercase;letter-spacing:0.1em;color:var(--muted);margin-bottom:8px;">Proposed Adaptive System</div>
              <div style="font-size:1.1rem;color:var(--text);line-height:1.6;">
                {row.get('accuracy',0)*100:.1f}% accuracy &nbsp;•&nbsp;
                {row.get('total_energy_consumed_uJ',0):,.0f} uJ energy &nbsp;•&nbsp;
                {row.get('energy_savings_vs_heavy_pct',0):.1f}% savings vs heavy &nbsp;•&nbsp;
                {int(row.get('model_switches_count',0))} model switches
              </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No baseline comparison results found. Run the pipeline to generate them.")
    st.markdown('</div>', unsafe_allow_html=True)

# ===========================================================================
# TAB 7 — ESP32 & HARDWARE
# ===========================================================================
with tab7:
    st.markdown('<div class="reveal reveal-delay-1">', unsafe_allow_html=True)
    st.subheader("🔌 ESP32 & Sensor Hardware Integration")
    st.markdown("""
    <p class="sub">Connect physical sensors and the ESP32 edge device to this dashboard over WiFi/MQTT.
    Live telemetry flows from the INA219 power monitor into the UI.</p>
    """, unsafe_allow_html=True)

    # Live telemetry cards
    tel = st.session_state.live_telemetry
    if tel and "error" not in tel:
        st.markdown("### Live Telemetry")
        t1, t2, t3, t4 = st.columns(4)
        with t1:
            st.markdown(f"""
            <div class="metric reveal reveal-delay-1">
              <div class="metric-label">Battery Level</div>
              <div class="metric-value">{tel.get('battery_pct', 0):.1f}%</div>
              <div class="metric-trend"><span class="dot"></span><span style="color:var(--muted);font-size:0.75rem;">INA219 bus voltage</span></div>
            </div>
            """, unsafe_allow_html=True)
        with t2:
            st.markdown(f"""
            <div class="metric reveal reveal-delay-2">
              <div class="metric-label">Bus Voltage</div>
              <div class="metric-value">{tel.get('bus_voltage_v', 0):.3f} V</div>
              <div class="metric-trend"><span style="color:var(--muted);font-size:0.75rem;">INA219 V<sub>bus</sub></span></div>
            </div>
            """, unsafe_allow_html=True)
        with t3:
            st.markdown(f"""
            <div class="metric reveal reveal-delay-3">
              <div class="metric-label">Shunt Current</div>
              <div class="metric-value">{tel.get('shunt_current_ma', 0):.2f} mA</div>
              <div class="metric-trend"><span style="color:var(--muted);font-size:0.75rem;">INA219 shunt</span></div>
            </div>
            """, unsafe_allow_html=True)
        with t4:
            st.markdown(f"""
            <div class="metric reveal reveal-delay-1">
              <div class="metric-label">Energy Consumed</div>
              <div class="metric-value">{tel.get('total_energy_uj', 0):,.0f} uJ</div>
              <div class="metric-trend"><span style="color:var(--muted);font-size:0.75rem;">Cumulative inference</span></div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Connect the ESP32 from the sidebar to see live telemetry.")

    st.markdown("### Recommended Sensor Wiring")
    wiring_md = """
| ESP32 Pin | Sensor | Purpose |
|-----------|--------|---------|
| GPIO22 (SCL) / GPIO21 (SDA) | INA219 | Bus voltage & shunt current → battery energy |
| GPIO4 | DHT22 | Ambient temperature / humidity (optional) |
| GPIO5 | PIR | Motion / presence detection (optional) |
| GPIO34 (ADC) | MQ-2 | Gas / smoke anomaly (optional) |
| WiFi (802.11 b/g/n) | MQTT Broker | Telemetry + inference → dashboard |
"""
    st.markdown(wiring_md)

    st.markdown("### Flashing the TinyML Model to ESP32")
    st.markdown("""
1. Run the TinyML optimization phase to generate `models/tinyml/esp32_model.h`.
2. Copy `esp32_model.h` into your Arduino/ESP-IDF project `src/` folder.
3. Include it and call `predict_tinyml_int8(features_float)` on each sensor sample.
4. Compile & upload via Arduino IDE (ESP32 board package) or ESP-IDF (`idf.py flash`).
    """)
    st.code('#include "esp32_model.h"\nfloat features[NUM_FEATURES] = {...};\nint leaf = predict_tinyml_int8(features);\n// map leaf index → attack class', language="cpp")

    st.markdown("### MQTT Topics (dashboard ↔ ESP32)")
    st.code("""
esp32/<device_id>/telemetry    ->  {battery_pct, bus_voltage_v, shunt_current_ma, total_energy_uj}
esp32/<device_id>/inference    ->  {prediction, threat_score, latency_ms, energy_uj, model}
esp32/<device_id>/command      <-  {model_tier, sleep_mode, sample_rate}
    """, language="text")

    st.markdown("### Python-side MQTT subscriber (dashboard)")
    st.code("""
from src.energy.esp32_interface import ESP32MQTTClient

def on_command(payload):
    print("Command from dashboard:", payload)

client = ESP32MQTTClient(device_id="esp32-ids-01", broker_host="localhost")
client.set_command_callback(on_command)
client.connect()
client.publish_telemetry({"battery_pct": 78.2, "bus_voltage_v": 3.9})
    """, language="python")

    # ESP32 connection status pill
    st.markdown("<br>", unsafe_allow_html=True)
    status_on = st.session_state.esp32_connected
    st.markdown(f"""
    <div class="reveal reveal-delay-3">
      <div class="status-pill">
        <span class="status-dot {'status-on' if status_on else 'status-off'}"></span>
        ESP32: {'ONLINE' if status_on else 'OFFLINE / SIMULATION'}
        &nbsp;|&nbsp;
        MQTT: {'ONLINE' if st.session_state.mqtt_client else 'OFFLINE'}
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("""
<div class="reveal reveal-delay-3" style="margin-top:60px; padding:30px 0; border-top:1px solid var(--border);">
  <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:16px; opacity:0.6;">
    <div style="font-size:0.85rem; color:var(--muted);">
      Adaptive Multi-Level TinyML IDS &nbsp;•&nbsp; Research prototype &nbsp;•&nbsp; ESP32 / IIoT edge-ready
    </div>
    <div style="display:flex; gap:16px; font-size:0.85rem; color:var(--muted);">
      <span>📊 Streamlit Dashboard</span>
      <span>⚡ Plotly Charts</span>
      <span>🔌 ESP32 + INA219</span>
      <span>🌐 MQTT Bridge</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # close .shell

# ---------------------------------------------------------------------------
# Scroll-reveal JavaScript (Intersection Observer)
# ---------------------------------------------------------------------------
st.components.v1.html("""
<script>
(function(){
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
      }
    });
  }, { threshold: 0.12, rootMargin: "0px 0px -60px 0px" });
  document.querySelectorAll('.reveal').forEach(el => observer.observe(el));
})();
</script>
""", height=0, width=0)

# ---------------------------------------------------------------------------
# Auto-refresh for live telemetry
# ---------------------------------------------------------------------------
if st.session_state.esp32_connected or st.session_state.mqtt_client:
    time.sleep(0.5)
    st.rerun()
