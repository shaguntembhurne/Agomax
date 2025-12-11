import streamlit as st
import pandas as pd
import time
import plotly.graph_objects as go

st.set_page_config(page_title="Agomax AirSim Monitor", layout="wide", page_icon="🎮")
st.markdown("""
<style>
    .metric-card {background-color: #0E1117; padding: 15px; border-radius: 10px; border: 1px solid #333;}
</style>
""", unsafe_allow_html=True)

st.title("🎮 AGOMAX x AIRSIM LINK")

# File to watch
LIVE_FILE = "data/live_airsim_stream.csv"

# Container for live updates
placeholder = st.empty()

def load_data():
    try:
        # Read only last 300 rows for performance
        df = pd.read_csv(LIVE_FILE)
        return df.tail(300)
    except:
        return pd.DataFrame()

while True:
    df = load_data()
    
    with placeholder.container():
        if df.empty:
            st.warning("Waiting for AirSim Bridge connection... (Run 'python airsim_bridge.py --mode live')")
        else:
            last_row = df.iloc[-1]
            is_anomaly = last_row['is_anomaly'] == 1
            
            # --- METRICS ROW ---
            c1, c2, c3, c4 = st.columns(4)
            
            c1.metric("Altitude", f"{last_row['altitude']:.2f} m")
            c2.metric("Roll Angle", f"{last_row['roll']:.2f}°")
            c3.metric("Battery", f"{last_row['battery_voltage']:.1f} V")
            
            status_html = f"<h2 style='color: {'red' if is_anomaly else '#00FF00'};'>{'🚨 ANOMALY' if is_anomaly else '✅ NOMINAL'}</h2>"
            c4.markdown(status_html, unsafe_allow_html=True)
            
            if is_anomaly:
                st.error(f"ROOT CAUSE: {last_row['root_cause_feature']} (Severity: {last_row['severity_score']})")

            # --- CHARTS ---
            col_graph1, col_graph2 = st.columns(2)
            
            # Graph 1: Altitude & Roll
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(y=df['altitude'], name='Altitude', line=dict(color='cyan')))
            fig1.add_trace(go.Scatter(y=df['roll'], name='Roll', line=dict(color='orange')))
            # Mark anomalies
            anoms = df[df['is_anomaly'] == 1]
            fig1.add_trace(go.Scatter(x=anoms.index, y=anoms['altitude'], mode='markers', marker=dict(color='red', size=8), name='Alert'))
            fig1.update_layout(title="Flight Dynamics", height=350, template="plotly_dark")
            col_graph1.plotly_chart(fig1, use_container_width=True)
            
            # Graph 2: Battery
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(y=df['battery_voltage'], name='Voltage', line=dict(color='#00FF00')))
            fig2.update_layout(title="Battery Telemetry", height=350, template="plotly_dark")
            col_graph2.plotly_chart(fig2, use_container_width=True)

    time.sleep(0.1) # Refresh rate