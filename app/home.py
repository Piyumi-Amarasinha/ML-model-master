from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Live Vegetable Price Predictor", layout="wide")

base_dir = Path(__file__).resolve().parents[1]
bg_image = base64.b64encode((base_dir / "assert" / "home2.jpg").read_bytes()).decode()

st.markdown(
    f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: linear-gradient(rgba(0, 0, 0, 0.5), rgba(0, 0, 0, 0.5)),
            url("data:image/jpg;base64,{bg_image}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    [data-testid="stHeader"] {{
        background: rgba(0, 0, 0, 0);
    }}
    [data-testid="stPageLink"] p {{
        color: white;
        font-size: 1.05rem;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

_, nav_col1, nav_col2 = st.columns([20, 2, 2])
with nav_col1:
    st.page_link("home.py", label="Home")
with nav_col2:
    st.page_link("pages/app.py", label="Train Model")

st.markdown("<div style='height: 12vh'></div>", unsafe_allow_html=True)
st.markdown(
    "<h1 style='text-align: center; color: white;'>Live Vegetable Price Predictor</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align: center; color: white; font-size: 1.2rem;'>"
    "Forecast wholesale vegetable prices across Sri Lanka using historical climate and market trends."
    "</p>",
    unsafe_allow_html=True,
)

_, button_col, _ = st.columns([3, 2, 3])
with button_col:
    if st.button("Go to Train Model", use_container_width=True, type="primary"):
        st.switch_page("pages/app.py")
