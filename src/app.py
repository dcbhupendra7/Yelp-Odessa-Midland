#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import streamlit as st

# Global env tweaks (quiet + stable)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

st.set_page_config(
    page_title="Odessa & Midland — Yelp Analytics + RAG",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Global CSS (applies to all pages)
st.markdown(
    """
    <style>
      .kpi {padding:18px;border-radius:14px;background:#12151e;border:1px solid #222636}
      .kpi h3{margin:0;font-size:14px;color:#9aa4b2;font-weight:600}
      .kpi p{margin:6px 0 0 0;font-size:26px;font-weight:700}
      .bubble-user   {background:#263559;border:1px solid #2f3e64}
      .bubble-assist {background:#182025;border:1px solid #242c33}
      .bubble {padding:14px 16px;border-radius:14px;margin:8px 0}
      .small-muted {color:#9aa4b2;font-size:12px}
      .source-pill {display:inline-block;margin:0 6px 6px 0;padding:6px 10px;border-radius:999px;background:#1d2330;border:1px solid #2a3140}
      .footer {color:#9aa4b2;text-align:center;margin-top:24px}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🍽️ Odessa & Midland — Yelp Analytics + RAG (Business-only)")
st.markdown(
    """
    Use the left sidebar to switch between pages:
    - **Analytics** — KPIs, charts, map, and CSV export  
    - **Chat** — RAG assistant grounded on your dataset (GPT-4o-mini)
    - **Investor Insights** — Strategic analysis for restaurant investment opportunities
    
    **📚 Documentation:** [Yelp Odessa–Midland Docs](https://dcbhupendra7.github.io/Yelp-Odessa-Midland/)
    """)

with st.sidebar:
    st.markdown("**📚 Documentation**")
    st.markdown("[Yelp Odessa–Midland Docs](https://dcbhupendra7.github.io/Yelp-Odessa-Midland/)")

st.markdown("<div class='footer'>© 2025 Bhupendra Dangi · Built with Yelp API, FAISS RAG, and GPT-4o-mini</div>", unsafe_allow_html=True)
