import os
import json
import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit Page Configuration
st.set_page_config(
    page_title="Adaptive TinyML IIoT Intrusion Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load configuration and experimental results
@st.cache_data
def load_project_data():
    paths = {
        "benchmark": "results/tables/model_benchmark.json",
        "tinyml": "results/tables/tinyml_optimization.json",
        "federated": "results/tables/federated_comparison.json",
        "baselines": "results/tables/baseline_comparison.json",
        "simulation": "results/experiments/simulation_results.json"
    }
    loaded = {}
    for key, path in paths.items():
        if os.path.exists(path):
            with open(path, "r") as f:
                loaded[key] = json.load(f)
        else:
            loaded[key] = None
    return loaded

data_store = load_project_data()

# Custom CSS Styling for Research Dashboard
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E88E5;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        font-weight: 400;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #1E88E5;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .status-badge-low { background-color: #28a745; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    .status-badge-med { background-color: #ffc107; color: black; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    .status-badge-high { background-color: #fd7e14; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    .status-badge-crit { background-color: #dc3545; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🛡️ Adaptive Multi-Level TinyML IIoT Intrusion Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Energy-Aware Model Switching & Federated Learning for Industrial IoT Edge Devices</div>', unsafe_allow_html=True)

# Sidebar Controls
st.sidebar.header("🕹️ Control Panel & Parameters")
simulation_speed = st.sidebar.slider("Simulation Frequency (Hz)", 1, 10, 5)
energy_threshold = st.sidebar.slider("Low Battery Threshold (%)", 10, 50, 35)
st.sidebar.markdown("---")
st.sidebar.info("System Target: ESP32 Edge Microcontroller\nProtocol: Modbus, MQTT, HTTP, TCP/UDP\nDataset: Edge-IIoTset")

# Tab Navigation
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 System Status & Real-Time Monitor",
    "⚡ Adaptive Model Switching",
    "🌐 Federated Learning & Non-IID",
    "🔬 TinyML Optimization",
    "🔋 Energy Consumption Analysis",
    "📈 Baseline Comparison & Research Results"
])

# TAB 1: SYSTEM STATUS & REAL-TIME MONITOR
with tab1:
    st.subheader("Real-Time IIoT Device State & Intrusion Monitoring")
    
    sim_logs = data_store.get("simulation")
    if sim_logs:
        latest = sim_logs[0]
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Battery Level", f"{latest['battery_pct']:.1f}%", delta="-0.2%/min")
        with col2:
            st.metric("Current Threat Score", f"{latest['threat_score']}/100", delta=latest['severity'])
        with col3:
            st.metric("Active Model", latest['selected_model'], delta=latest['model_switch'])
        with col4:
            st.metric("Prediction / Attack", latest['predicted_class'], f"Conf: {latest['confidence']*100:.1f}%")
        with col5:
            st.metric("Inference Latency", f"{latest['inference_latency_ms']:.2f} ms", f"{latest['energy_consumed_uj']} uJ")
            
        st.markdown("### Recent Network Event Logs")
        df_sim = pd.DataFrame(sim_logs)
        st.dataframe(
            df_sim[["event_id", "timestamp", "true_class", "predicted_class", "confidence", "threat_score", "severity", "battery_pct", "selected_model", "model_switch"]],
            height=300,
            use_container_width=True
        )
    else:
        st.warning("No simulation logs found. Please run the simulation pipeline.")

# TAB 2: ADAPTIVE MODEL SWITCHING
with tab2:
    st.subheader("Dynamic Utility Engine & Model Switching Timeline")
    st.write("Models automatically switch level (LIGHTWEIGHT ↔ MEDIUM ↔ HEAVY) based on Threat Urgency vs Energy Availability.")
    
    if sim_logs:
        df_sim = pd.DataFrame(sim_logs)
        fig, ax1 = plt.subplots(figsize=(10, 4))
        
        ax1.plot(df_sim["event_id"], df_sim["threat_score"], color="purple", label="Threat Score")
        ax1.set_xlabel("Event ID")
        ax1.set_ylabel("Threat Score (0-100)", color="purple")
        
        ax2 = ax1.twinx()
        model_nums = df_sim["selected_model"].map({"LIGHTWEIGHT": 1, "MEDIUM": 2, "HEAVY": 3})
        ax2.step(df_sim["event_id"], model_nums, color="teal", where="post", linewidth=2, label="Selected Model")
        ax2.set_yticks([1, 2, 3])
        ax2.set_yticklabels(["Lightweight", "Medium", "Heavy"])
        ax2.set_ylabel("Selected Model Level", color="teal")
        
        st.pyplot(fig)
        plt.close(fig)

# TAB 3: FEDERATED LEARNING
with tab3:
    st.subheader("Collaborative Training & Non-IID Data Convergence")
    
    fed_data = data_store.get("federated")
    if fed_data:
        std_fl = fed_data["standard_fl"]
        energy_fl = fed_data["energy_aware_fl"]
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Simulated Clients", "5 Edge Nodes")
        c2.metric("FL Communication Rounds", "10 Rounds")
        c3.metric("Energy-Aware Accuracy", f"{energy_fl['final_global_accuracy']*100:.2f}%")
        c4.metric("Comm Overhead Reduction", f"{((std_fl['total_comm_overhead_kb'] - energy_fl['total_comm_overhead_kb'])/std_fl['total_comm_overhead_kb'])*100:.1f}%")
        
        # Convergence plot
        df_std = pd.DataFrame(std_fl["history"])
        df_ene = pd.DataFrame(energy_fl["history"])
        
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(df_std["round"], df_std["global_accuracy"]*100, marker="o", label="Standard FedAvg", color="red")
        ax.plot(df_ene["round"], df_ene["global_accuracy"]*100, marker="s", label="Energy-Aware FL (Proposed)", color="green")
        ax.set_title("Federated Learning Global Accuracy Convergence")
        ax.set_xlabel("Communication Round")
        ax.set_ylabel("Global Accuracy (%)")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

# TAB 4: TINYML OPTIMIZATION
with tab4:
    st.subheader("TinyML INT8 Quantization & ESP32 Header Export")
    
    t_data = data_store.get("tinyml")
    if t_data:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Original Size", f"{t_data['original_size_kb']} KB")
        col2.metric("INT8 Quantized Size", f"{t_data['quantized_size_kb']} KB")
        col3.metric("Storage Reduction", f"{t_data['size_reduction_pct']:.1f}%")
        col4.metric("Accuracy Drop", f"{t_data['accuracy_drop_pct']:.2f}%")
        
        st.success("ESP32 C Header generated at `models/tinyml/esp32_model.h` (Ready for Arduino / ESP-IDF build).")

# TAB 5: ENERGY CONSUMPTION
with tab5:
    st.subheader("Energy Consumption & Battery Longevity Analysis")
    
    base_data = data_store.get("baselines")
    if base_data:
        df_base = pd.DataFrame(base_data).T
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(data=df_base, x="name", y="total_energy_consumed_uJ", palette="viridis", ax=ax)
        ax.set_title("Total Energy Consumed (Microjoules uJ)")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=15)
        st.pyplot(fig)
        plt.close(fig)

# TAB 6: BASELINE COMPARISON
with tab6:
    st.subheader("Research Comparison: Proposed System vs Baselines")
    if base_data:
        df_b = pd.DataFrame(base_data).T
        st.table(df_b[["name", "accuracy", "total_energy_consumed_uJ", "avg_inference_latency_ms", "model_switches_count", "energy_savings_vs_heavy_pct"]])
