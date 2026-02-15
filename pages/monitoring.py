#monitoring.py

import streamlit as st  # CORRECT
from langsmith_config import LangSmithMonitor
import plotly.express as px

st.title("LangSmith Monitoring Dashboard")

monitor = LangSmithMonitor()
stats = monitor.get_project_stats("langgraph-chatbot")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Conversations", stats["total_runs"])
col2.metric("Total Tokens", f"{stats['total_tokens']:,}")
col3.metric("Avg Latency", f"{stats['avg_latency_seconds']:.2f}s")
col4.metric("Error Rate", f"{stats['error_rate']*100:.1f}%")