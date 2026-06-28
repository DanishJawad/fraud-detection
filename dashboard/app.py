"""Streamlit dashboard for monitoring the fraud detection API."""
import streamlit as st
import requests
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Fraud Detection Monitoring", layout="wide")
st.title("🚨 Fraud Detection Monitoring")

API_URL = st.sidebar.text_input("API URL", "http://localhost:8000")

tab1, tab2, tab3 = st.tabs(["Overview", "Details", "Alerts"])

with tab1:
    st.subheader("System Overview")
    try:
        r = requests.get(f"{API_URL}/metrics", timeout=3)
        if r.status_code == 200:
            metrics = r.json()
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Fraud Detection Rate", f"{metrics.get('fraud_detection_rate', 0):.1%}")
            col2.metric("False Alarm Rate", f"{metrics.get('false_alarm_rate', 0):.2%}")
            col3.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.4f}")
            col4.metric("Cost Savings", f"{metrics.get('cost_savings_percent', 0):.1f}%")
        else:
            st.error("Could not fetch metrics from API")
    except Exception as e:
        st.error(f"API error: {e}")

with tab2:
    st.subheader("Metrics Table")
    try:
        r = requests.get(f"{API_URL}/metrics", timeout=3)
        if r.status_code == 200:
            m = r.json()
            df = pd.DataFrame({"metric": list(m.keys()), "value": list(m.values())})
            st.dataframe(df)
    except Exception as e:
        st.error(f"API error: {e}")

with tab3:
    st.subheader("Health")
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        if r.status_code == 200:
            h = r.json()
            if h.get("status") == "healthy":
                st.success("Model healthy")
            else:
                st.warning(f"Model status: {h.get('status')}")
        else:
            st.error("Health endpoint error")
    except Exception as e:
        st.error(f"API error: {e}")
