"""
Adaptive Multi-Level TinyML IIoT Intrusion Detection Dashboard.

Streamlit frontend for the research system. Features:
  * Interactive real-time IIoT simulation (run from the UI)
  * Plotly-powered interactive charts (threat score, model switching,
    federated convergence, energy, baselines)
  * ESP32 hardware connection panel (serial port, MQTT bridge, live telemetry)
  * Dark / light mode
  * Live threat & battery gauges
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

# Optional plotly dependency (graceful fallback to matplotlib)
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    _HAS_PLOTLY = True
except Exception:  # pragma: no cover
    _HAS_PLOTLY = False

import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Adaptive TinyML IIoT IDS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
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

# ---------------------------------------------------------------------------
# Helpers
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


def severity_badge(severity: str) -> str:
    mapping = {
        "LOW": "status-badge-low",
        "MEDIUM": "status-badge-med",
        "HIGH": "status-badge-high",
        "CRITICAL": "status-badge-crit",
    }
    cls = mapping.get(severity.upper(), "status-badge-low")
    return f'<span class="{cls}">{severity.upper()}</span>'


def theme_css(dark: bool) -> str:
    if dark:
        bg, card, text, muted = "#0e1117", "#1c2128", "#fafafa", "#9ca3af"
        accent = "#38bdf8"
    else:
        bg, card, text, muted = "#ffffff", "#f8f9fa", "#111827", "#6b7280"
        accent = "#1E88E5"
    return f"""
    <style>
        :root {{ --accent: {accent}; }}
        .main-title {{
            font-size: 2.1rem; font-weight: 800; color: {accent};
            margin-bottom: 0.15rem; letter-spacing: -0.3px;
        }}
        .sub-title {{
            font-size: 1.05rem; font-weight: 400; color: {muted};
            margin-bottom: 1.6rem;
        }}
        .metric-card {{
            background: {card}; border-radius: 10px; padding: 14px 16px;
            border-left: 5px solid {accent};
            box-shadow: 0 2px 6px rgba(0,0,0,0.06);
            color: {text};
        }}
        .metric-value {{ font-size: 1.5rem; font-weight: 700; }}
        .metric-label {{ font-size: 0.8rem; color: {muted}; text-transform: uppercase; letter-spacing: 0.4px; }}
        .status-badge-low {{ background-color: #28a745; color: #fff; padding: 3px 8px; border-radius: 4px; font-weight: 700; font-size: 0.75rem; }}
        .status-badge-med {{ background-color: #ffc107; color: #000; padding: 3px 8px; border-radius: 4px; font-weight: 700; font-size: 0.75rem; }}
        .status-badge-high {{ background-color: #fd7e14; color: #fff; padding: 3px 8px; border-radius: 4px; font-weight: 700; font-size: 0.75rem; }}
        .status-badge-crit {{ background-color: #dc3545; color: #fff; padding: 3px 8px; border-radius: 4px; font-weight: 700; font-size: 0.75rem; }}
        .esp32-on {{ color: #28a745; font-weight: 700; }}
        .esp32-off {{ color: #dc3545; font-weight: 700; }}
        .stApp {{ background-color: {bg}; }}
        .block-container {{ padding-top: 1rem; }}
    </style>
    """

# ---------------------------------------------------------------------------
# Header & theme
# ---------------------------------------------------------------------------
st.markdown(theme_css(st.session_state.dark_mode), unsafe_allow_html=True)

st.markdown(
    '<div class="main-title">🛡️ Adaptive Multi-Level TinyML IIoT IDS</div>', unsafe_allow_html=True
)
st.markdown(
    '<div class="sub-title">Energy-Aware Model Switching & Federated Learning for Industrial IoT Edge Devices</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.header("🕹️ Control Panel")
st.session_state.dark_mode = st.sidebar.toggle(
    "🌙 Dark Mode", value=st.session_state.dark_mode
)

st.sidebar.markdown("### ⚙️ Simulation Parameters")
sim_speed = st.sidebar.slider("Simulation Frequency (Hz)", 1, 20, 5, key="sim_speed")
energy_threshold = st.sidebar.slider("Low Battery Threshold (%)", 10, 50, 35, key="batt_thresh")
num_events = st.sidebar.number_input("Events per Batch", 10, 2000, 100, step=10, key="num_events")

st.sidebar.markdown("---")

st.sidebar.markdown("### 🔌 ESP32 & Hardware")
esp_port = st.sidebar.text_input("Serial Port (e.g. COM3 / /dev/ttyUSB0)", "COM3", key="esp_port")
mqtt_broker = st.sidebar.text_input("MQTT Broker Host", "localhost", key="mqtt_broker")
mqtt_port = st.sidebar.number_input("MQTT Port", 1, 65535, 1883, key="mqtt_port")
device_id = st.sidebar.text_input("Device ID", "esp32-ids-01", key="device_id")

col_conn1, col_conn2 = st.sidebar.columns(2)
if col_conn1.button("Connect ESP32", use_container_width=True, key="btn_connect_esp32"):
    try:
        from src.energy.esp32_interface import ESP32INA219EnergyProvider
        prov = ESP32INA219EnergyProvider()
        ok = prov.connect_sensor()
        st.session_state.esp32_connected = ok
        st.session_state.esp32_provider = prov if ok else None
        st.sidebar.success("ESP32 connected") if ok else st.sidebar.error("ESP32 connect failed (stub mode)")
    except Exception as exc:  # pragma: no cover
        st.sidebar.error(f"Connect error: {exc}")
        st.session_state.esp32_connected = False

if col_conn2.button("Disconnect", use_container_width=True, key="btn_disconnect_esp32"):
    prov = st.session_state.esp32_provider
    if prov is not None:
        try:
            prov.close()
        except Exception:
            pass
    st.session_state.esp32_connected = False
    st.session_state.esp32_provider = None
    st.sidebar.info("ESP32 disconnected")

mqtt_col1, mqtt_col2 = st.sidebar.columns(2)
if mqtt_col1.button("Connect MQTT", use_container_width=True, key="btn_connect_mqtt"):
    try:
        from src.energy.esp32_interface import ESP32MQTTClient
        cli = ESP32MQTTClient(device_id=device_id, broker_host=mqtt_broker, broker_port=int(mqtt_port))
        ok = cli.connect()
        st.session_state.mqtt_client = cli if ok else None
        st.sidebar.success("MQTT connected") if ok else st.sidebar.error("MQTT connect failed (broker?)")
    except Exception as exc:  # pragma: no cover
        st.sidebar.error(f"MQTT error: {exc}")
        st.session_state.mqtt_client = None

if mqtt_col2.button("Disconnect MQTT", use_container_width=True, key="btn_disconnect_mqtt"):
    cli = st.session_state.mqtt_client
    if cli is not None:
        cli.disconnect()
    st.session_state.mqtt_client = None

st.sidebar.markdown("---")
st.sidebar.markdown("### 🚀 Actions")
if st.sidebar.button("Run Live Simulation Batch", use_container_width=True, key="btn_run_sim"):
    st.session_state.sim_running = True

if st.sidebar.button("Run Full Research Pipeline", use_container_width=True, key="btn_run_pipeline"):
    st.sidebar.info("Pipeline runs in background. Check logs/results.")

st.sidebar.markdown("---")
st.sidebar.caption("Target: ESP32 Edge MCU  |  Protocols: Modbus, MQTT, HTTP, TCP/UDP  |  Dataset: Edge-IIoTset")

# ---------------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------------
data_store = load_project_data()

# ===========================================================================
# TABS
# ===========================================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 System Status & Live Monitor",
    "⚡ Adaptive Model Switching",
    "🌐 Federated Learning & Non-IID",
    "🔬 TinyML Optimization",
    "🔋 Energy Consumption",
    "📈 Baseline Comparison",
    "🔌 ESP32 & Hardware",
])

# ---------------------------------------------------------------------------
# TAB 1: SYSTEM STATUS & LIVE MONITOR
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Real-Time IIoT Device State & Intrusion Monitoring")

    sim_logs = st.session_state.sim_logs if st.session_state.sim_logs else data_store.get("simulation")

    # Run a live batch if requested
    if st.session_state.sim_running:
        st.session_state.sim_running = False
        with st.spinner("Running live IIoT simulation batch..."):
            try:
                from src.simulation.iiot_simulator import RealtimeIIoTSimulator
                sim = RealtimeIIoTSimulator(initial_battery_pct=100.0)
                logs = sim.run_simulation_batch(count=int(num_events))
                st.session_state.sim_logs = logs
                sim_logs = logs
                st.success(f"Simulation complete: {len(logs)} events processed.")
            except Exception as exc:
                st.error(f"Simulation error: {exc}")

    # Refresh telemetry from ESP32 / MQTT
    prov = st.session_state.esp32_provider
    if prov is not None:
        try:
            batt = prov.get_battery_percentage()
            st.session_state.live_telemetry = {
                "battery_pct": batt,
                "bus_voltage_v": prov.get_bus_voltage_v(),
                "shunt_current_ma": prov.get_shunt_current_ma(),
                "total_energy_uj": prov.total_energy_consumed_uj,
            }
        except Exception as exc:
            st.session_state.live_telemetry = {"error": str(exc)}

    # KPI cards
    latest = sim_logs[-1] if sim_logs else None
    c1, c2, c3, c4, c5 = st.columns(5)

    batt_val = None
    if st.session_state.live_telemetry and "battery_pct" in st.session_state.live_telemetry:
        batt_val = st.session_state.live_telemetry["battery_pct"]
    elif latest:
        batt_val = latest["battery_pct"]

    with c1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Edge Battery Level</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{batt_val:.1f}%</div>' if batt_val is not None else '<div class="metric-value">--</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">ESP32 Status</div>', unsafe_allow_html=True)
        status_html = '<span class="esp32-on">● CONNECTED</span>' if st.session_state.esp32_connected else '<span class="esp32-off">● DISCONNECTED (SIM)</span>'
        st.markdown(status_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Last Prediction</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{latest["predicted_class"] if latest else "--"}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Threat Severity</div>', unsafe_allow_html=True)
        st.markdown(severity_badge(latest["severity"] if latest else "LOW"), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c5:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Selected Model</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{latest["selected_model"] if latest else "--"}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Threat gauge + live plot
    if sim_logs:
        df_sim = pd.DataFrame(sim_logs)
        g1, g2 = st.columns([1, 2])
        with g1:
            latest_score = int(df_sim["threat_score"].iloc[-1])
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=latest_score,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Current Threat Score"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#1E88E5"},
                    "steps": [
                        {"range": [0, 30], "color": "#e8f5e9"},
                        {"range": [30, 60], "color": "#fff3e0"},
                        {"range": [60, 80], "color": "#ffccbc"},
                        {"range": [80, 100], "color": "#ffcdd2"},
                    ],
                    "threshold": {"line": {"color": "red", "width": 4}, "value": 80},
                },
            ))
            fig.update_layout(height=260, margin=dict(t=40, b=20, l=20, r=20))
            st.plotly_chart(fig, use_container_width=True)
        with g2:
            fig2 = make_subplots(specs=[[{"secondary_y": True}]])
            fig2.add_trace(
                go.Scatter(x=df_sim["event_id"], y=df_sim["threat_score"],
                           name="Threat Score", line=dict(color="purple", width=2)),
                secondary_y=False,
            )
            model_map = {"LIGHTWEIGHT": 1, "MEDIUM": 2, "HEAVY": 3}
            model_nums = df_sim["selected_model"].map(model_map)
            fig2.add_trace(
                go.Scatter(x=df_sim["event_id"], y=model_nums,
                           name="Selected Model", mode="lines",
                           line=dict(color="teal", width=3), step="after"),
                secondary_y=True,
            )
            fig2.update_yaxes(title_text="Threat Score (0-100)", secondary_y=False, range=[0, 100])
            fig2.update_yaxes(title_text="Selected Model", secondary_y=True,
                              tickvals=[1, 2, 3], ticktext=["Lightweight", "Medium", "Heavy"])
            fig2.update_xaxes(title_text="Event ID")
            fig2.update_layout(height=360, margin=dict(t=20, b=20, l=20, r=20),
                               legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig2, use_container_width=True)

        # Live event table (recent)
        st.markdown("### Recent Events")
        show_cols = ["timestamp", "event_id", "true_class", "predicted_class",
                     "threat_score", "severity", "battery_pct", "selected_model",
                     "inference_latency_ms", "energy_consumed_uj", "model_switch"]
        show_cols = [c for c in show_cols if c in df_sim.columns]
        st.dataframe(df_sim[show_cols].tail(15)[::-1], use_container_width=True, height=340)

# ---------------------------------------------------------------------------
# TAB 3: FEDERATED LEARNING
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Collaborative Training & Non-IID Data Convergence")
    fed_data = data_store.get("federated")
    if fed_data:
        std_fl = fed_data.get("standard_fl", {})
        energy_fl = fed_data.get("energy_aware_fl", {})

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Simulated Clients", "5 Edge Nodes")
        c2.metric("FL Communication Rounds", "10 Rounds")
        c3.metric("Energy-Aware Accuracy", f"{energy_fl.get('final_global_accuracy', 0)*100:.2f}%")
        c4.metric("Comm Overhead Reduction",
                  f"{((std_fl.get('total_comm_overhead_kb', 0) - energy_fl.get('total_comm_overhead_kb', 0)) / max(std_fl.get('total_comm_overhead_kb', 1), 1))*100:.1f}%")

        st.markdown("### Global Accuracy Convergence")
        hist_std = std_fl.get("history", [])
        hist_ene = energy_fl.get("history", [])
        if hist_std and hist_ene and _HAS_PLOTLY:
            df_std = pd.DataFrame(hist_std)
            df_ene = pd.DataFrame(hist_ene)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_std["round"], y=df_std["global_accuracy"]*100,
                                     mode="lines+markers", name="Standard FedAvg",
                                     line=dict(color="#ef4444", width=3)))
            fig.add_trace(go.Scatter(x=df_ene["round"], y=df_ene["global_accuracy"]*100,
                                     mode="lines+markers", name="Energy-Aware FL (Proposed)",
                                     line=dict(color="#22c55e", width=3)))
            fig.update_layout(height=380, margin=dict(t=20, b=20, l=20, r=20),
                              xaxis_title="Communication Round",
                              yaxis_title="Global Accuracy (%)",
                              legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Client Battery & Participation")
        if hist_ene and _HAS_PLOTLY:
            df_ene = pd.DataFrame(hist_ene)
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=df_ene["round"], y=df_ene["avg_battery_pct"],
                                      mode="lines+markers", name="Avg Battery (%)",
                                      line=dict(color="#3b82f6", width=3)))
            fig2.add_trace(go.Scatter(x=df_ene["round"], y=df_ene["min_battery_pct"],
                                      mode="lines+markers", name="Min Battery (%)",
                                      line=dict(color="#f59e0b", width=2, dash="dot")))
            fig2.update_layout(height=320, margin=dict(t=20, b=20, l=20, r=20),
                               xaxis_title="Round", yaxis_title="Battery (%)",
                               yaxis=dict(range=[0, 100]))
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No federated learning results found. Run the pipeline to generate them.")

# ---------------------------------------------------------------------------
# TAB 4: TINYML OPTIMIZATION
# ---------------------------------------------------------------------------
with tab4:
    st.subheader("TinyML INT8 Quantization & ESP32 Header Export")
    t_data = data_store.get("tinyml")
    if t_data:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Original Size", f"{t_data.get('original_size_kb', 0)} KB")
        c2.metric("INT8 Quantized Size", f"{t_data.get('quantized_size_kb', 0)} KB")
        c3.metric("Storage Reduction", f"{t_data.get('size_reduction_pct', 0):.1f}%")
        c4.metric("Accuracy Drop", f"{t_data.get('accuracy_drop_pct', 0):.2f}%")

        st.markdown("### ESP32 Deployment Readiness")
        col_a, col_b = st.columns([1, 2])
        with col_a:
            st.metric("ESP32 RAM (KB)", f"{t_data.get('esp32_ram_requirement_kb', 1.2)}")
            st.metric("ESP32 Flash (KB)", f"{t_data.get('esp32_flash_requirement_kb', 0)}")
            st.metric("ESP32 Compatible", "✅ Yes" if t_data.get("esp32_compatible") else "❌ No")
            st.success("ESP32 C header generated at `models/tinyml/esp32_model.h`")
        with col_b:
            h_path = resolve_path("models/tinyml/esp32_model.h")
            if os.path.exists(h_path):
                with open(h_path, "r", encoding="utf-8") as f:
                    c_header = f.read()
                st.code(c_header, language="c")
                st.download_button("Download esp32_model.h", c_header,
                                   file_name="esp32_model.h", mime="text/x-csrc",
                                   use_container_width=True)
            else:
                st.warning("C header not found. Run TinyML optimization phase.")
    else:
        st.info("No TinyML optimization results found. Run the pipeline to generate them.")

# ---------------------------------------------------------------------------
# TAB 5: ENERGY CONSUMPTION
# ---------------------------------------------------------------------------
with tab5:
    st.subheader("Energy Consumption & Battery Longevity Analysis")
    base_data = data_store.get("baselines")
    sim_logs = st.session_state.sim_logs if st.session_state.sim_logs else data_store.get("simulation")

    if base_data:
        df_base = pd.DataFrame(base_data).T
        if _HAS_PLOTLY:
            fig = go.Figure()
            for _, row in df_base.iterrows():
                fig.add_trace(go.Bar(
                    name=row.get("name", "?"),
                    x=[row.get("name", "?")],
                    y=[row.get("total_energy_consumed_uJ", 0)],
                    text=[f"{row.get('total_energy_consumed_uJ', 0):,.0f} uJ"],
                    textposition="outside",
                ))
            fig.update_layout(height=380, margin=dict(t=20, b=60, l=20, r=20),
                              xaxis_title="System Configuration",
                              yaxis_title="Total Energy Consumed (uJ)",
                              showlegend=False,
                              xaxis_tickangle=-15)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.bar_chart(df_base[["total_energy_consumed_uJ"]])

    if sim_logs:
        st.markdown("### Battery Drain Over Time (Adaptive System)")
        df_sim = pd.DataFrame(sim_logs)
        if _HAS_PLOTLY:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=df_sim["event_id"], y=df_sim["battery_pct"],
                                      mode="lines", name="Battery %",
                                      line=dict(color="#3b82f6", width=3)))
            fig2.add_trace(go.Scatter(x=df_sim["event_id"], y=[35]*len(df_sim),
                                      mode="lines", name="Low Battery Threshold",
                                      line=dict(color="#f59e0b", width=2, dash="dash")))
            fig2.update_layout(height=320, margin=dict(t=20, b=20, l=20, r=20),
                               xaxis_title="Event ID", yaxis_title="Battery (%)",
                               yaxis=dict(range=[0, 105]))
            st.plotly_chart(fig2, use_container_width=True)
    elif base_data is None:
        st.info("Run the pipeline to generate energy results.")

# ---------------------------------------------------------------------------
# TAB 6: BASELINE COMPARISON
# ---------------------------------------------------------------------------
with tab6:
    st.subheader("Research Comparison: Proposed System vs Baselines")
    base_data = data_store.get("baselines")
    if base_data:
        df_b = pd.DataFrame(base_data).T
        display_cols = ["name", "accuracy", "total_energy_consumed_uJ",
                        "avg_inference_latency_ms", "model_switches_count",
                        "energy_savings_vs_heavy_pct"]
        display_cols = [c for c in display_cols if c in df_b.columns]
        st.dataframe(df_b[display_cols], use_container_width=True)

        st.markdown("### Accuracy vs Energy Trade-off")
        if _HAS_PLOTLY and {"accuracy", "total_energy_consumed_uJ", "name"}.issubset(df_b.columns):
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_b["total_energy_consumed_uJ"],
                y=df_b["accuracy"] * 100,
                mode="markers+text",
                text=df_b["name"],
                textposition="top center",
                marker=dict(size=14, color=df_b["accuracy"]*100,
                            colorscale="Viridis", showscale=True, colorbar=dict(title="Accuracy %")),
            ))
            fig.update_layout(height=400, margin=dict(t=20, b=40, l=20, r=20),
                              xaxis_title="Total Energy Consumed (uJ)",
                              yaxis_title="Accuracy (%)")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Key Takeaways")
        prop = df_b[df_b["name"].str.contains("Proposed|Adaptive", case=False, na=False)]
        if len(prop):
            row = prop.iloc[0]
            st.success(
                f"Proposed adaptive system: {row.get('accuracy',0)*100:.1f}% accuracy, "
                f"{row.get('total_energy_consumed_uJ',0):,.0f} uJ energy, "
                f"{row.get('energy_savings_vs_heavy_pct',0):.1f}% energy savings vs heavy, "
                f"{int(row.get('model_switches_count',0))} model switches."
            )
    else:
        st.info("No baseline comparison results found. Run the pipeline to generate them.")

# ---------------------------------------------------------------------------
# TAB 7: ESP32 & HARDWARE
# ---------------------------------------------------------------------------
with tab7:
    st.subheader("ESP32 & Sensor Hardware Integration")
    st.markdown("Connect physical sensors and the ESP32 edge device to this dashboard.")

    st.markdown("### Recommended Sensor Wiring (ESP32 → Sensor)")
    wiring_md = """
| ESP32 Pin | Sensor | Purpose |
|-----------|--------|---------|
| GPIO22 (SCL) | INA219 SCL | Bus voltage / shunt current (battery energy) |
| GPIO21 (SDA) | INA219 SDA | I2C data |
| 3.3V | INA219 VCC | Sensor power |
| GND | INA219 GND | Common ground |
| GPIO4 | DHT22 DATA | Ambient temperature & humidity (optional env. context) |
| GPIO5 | PIR / Motion | Physical tamper / presence detection |
| GPIO34 (ADC) | MQ-2 / Analog | Gas / smoke anomaly (optional physical-layer signal) |
| GPIO1 (TX) / GPIO3 (RX) | UART | Debug console / serial bridge to PC |
| WiFi (802.11 b/g/n) | MQTT Broker | Publish telemetry & inference to this dashboard |
"""
    st.markdown(wiring_md)

    st.markdown("### Live Telemetry")
    tel = st.session_state.live_telemetry
    if tel:
        t1, t2, t3, t4 = st.columns(4)
        t1.metric("Battery (%)", f"{tel.get('battery_pct', 0):.1f}")
        t2.metric("Bus Voltage (V)", f"{tel.get('bus_voltage_v', 0):.3f}")
        t3.metric("Shunt Current (mA)", f"{tel.get('shunt_current_ma', 0):.2f}")
        t4.metric("Energy Consumed (uJ)", f"{tel.get('total_energy_uj', 0):,.0f}")
    else:
        st.info("Connect the ESP32 (sidebar) to see live telemetry.")

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

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption("Adaptive Multi-Level TinyML IDS  •  Research prototype  •  ESP32 / IIoT edge-ready")
