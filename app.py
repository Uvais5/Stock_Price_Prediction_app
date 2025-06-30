import base64
from typing import Optional
from io import StringIO
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from prophet import Prophet
from prophet.plot import plot_components_plotly, plot_plotly
import yfinance as yf

###############################################################################
# Helper – optional background image                                           #
###############################################################################

def _get_base64(bin_file: str) -> str:
    with open(bin_file, "rb") as f:
        return base64.b64encode(f.read()).decode()


def set_background(png_file: str):
    """Set a full‑screen background image inside Streamlit."""
    bin_str = _get_base64(png_file)
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{bin_str}");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

###############################################################################
# Load data – upload or download                                               #
###############################################################################

st.title("📈 Simple Stock‑Price Prediction App (Prophet)")

source = st.sidebar.selectbox(
    "Load data via…",
    (
        "Upload CSV",
        "Download from Yahoo Finance",
    ),
)

data: Optional[pd.DataFrame] = None
symbol_title: str = ""

# ────────────────────────────────────────────────────────────────────────────
# 1. Upload CSV
# ────────────────────────────────────────────────────────────────────────────
if source == "Upload CSV":
    uploaded = st.sidebar.file_uploader(
        "Upload Yahoo Finance CSV", type="csv", accept_multiple_files=False
    )
    if uploaded is not None:
        data = pd.read_csv(uploaded)
        symbol_title = uploaded.name.split(".")[0].upper()
    else:
        st.info("Upload a CSV file or switch to *Download*.")
        st.stop()

# ────────────────────────────────────────────────────────────────────────────
# 2. Download using yfinance
# ────────────────────────────────────────────────────────────────────────────
if source == "Download from Yahoo Finance":
    ticker = st.sidebar.text_input("Ticker", "AAPL", max_chars=8).upper().strip()
    col1, col2 = st.sidebar.columns(2)
    start_date = col1.date_input("Start date", pd.to_datetime("2023-01-01"))
    end_date = col2.date_input("End date", pd.to_datetime("today"))
    fetch = st.sidebar.button("🔽 Fetch data")

    if fetch:
        if start_date >= end_date:
            st.error("❌ *Start* must be before *End*.")
            st.stop()
        with st.spinner("Downloading from Yahoo Finance…"):
            try:
                data = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date + pd.Timedelta(days=1),  # include end day
                    interval="1d",
                    progress=False,
                    auto_adjust=False,
                    threads=True,
                )
            except Exception as e:
                st.error(f"Download failed: {e}")
                st.stop()
        if data.empty:
            st.error("No data returned – check symbol or try again later.")
            st.stop()
        data.reset_index(inplace=True)
        symbol_title = ticker
        st.success(f"✅ Downloaded {len(data)} rows for **{symbol_title}**")

# Guard: we need data to proceed
if data is None:
    st.stop()

###############################################################################
# Prepare dataset for Prophet                                                   #
###############################################################################

data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
# Drop rows with missing dates or prices
clean = data.dropna(subset=["Date", "Close"])
if clean.empty:
    st.error("Dataset has no usable Date/Close columns.")
    st.stop()

prophet_df = (
    clean[["Date", "Close"]]
    .rename(columns={"Date": "ds", "Close": "y"})
    .sort_values("ds")
    .reset_index(drop=True)
)

###############################################################################
# Fit Prophet model                                                             #
###############################################################################

with st.spinner("Fitting Prophet model…"):
    model = Prophet()
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=365)
    forecast = model.predict(future)

###############################################################################
# Sidebar navigation                                                            #
###############################################################################

page = st.sidebar.radio(
    "Navigate",
    (
        "Overview",
        "Historical Candlestick",
        "Forecast Line",
        "Actual vs Predicted",
        "Residuals",
        "Components (Year / Week)",
        "Monthly Forecast (12 mo)",
        "Compare Price",
    ),
)

###############################################################################
# Page functions                                                                #
###############################################################################

def overview():
    url = "https://facebook.github.io/prophet/"
    st.markdown(
        f"""
        ### Prophet forecast for `{symbol_title}`  
        **Rows**: {len(clean)}    •    **Date range**: {clean['Date'].min().date()} → {clean['Date'].max().date()}  
        **Model docs**: [Prophet]({url})
        """
    )
    st.dataframe(clean.head())


def candlestick():
    st.subheader(f"{symbol_title} – Historical prices")
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=clean["Date"],
                open=clean["Open"],
                high=clean["High"],
                low=clean["Low"],
                close=clean["Close"],
                name="OHLC",
            )
        ]
    )
    fig.update_layout(xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)


def forecast_line():
    st.subheader("1‑year forecast")
    st.plotly_chart(plot_plotly(model, forecast), use_container_width=True)


def actual_vs_pred():
    merged = pd.merge(
        prophet_df,
        forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]],
        on="ds",
        how="inner",
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=merged["ds"], y=merged["y"], name="Actual", mode="lines"))
    fig.add_trace(go.Scatter(x=merged["ds"], y=merged["yhat"], name="Predicted", mode="lines"))
    fig.add_trace(
        go.Scatter(
            x=pd.concat([merged["ds"], merged["ds"][::-1]]),
            y=pd.concat([merged["yhat_upper"], merged["yhat_lower"][::-1]]),
            fill="toself",
            fillcolor="rgba(0,100,80,0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    st.plotly_chart(fig, use_container_width=True)


def residuals():
    merged = pd.merge(prophet_df, forecast[["ds", "yhat"]], on="ds", how="inner")
    merged["residual"] = merged["y"] - merged["yhat"]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Residuals over time")
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=merged["ds"], y=merged["residual"], mode="lines", name="Residual"))
        fig1.add_hline(y=0, line_dash="dash")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("Residual distribution")
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(x=merged["residual"], nbinsx=40, name="Histogram"))
        st.plotly_chart(fig2, use_container_width=True)


def components():
    st.subheader("Prophet components")
    st.plotly_chart(plot_components_plotly(model, forecast), use_container_width=True)


def monthly():
    st.subheader("Monthly forecast (12 months)")
    m = Prophet(changepoint_prior_scale=0.01).fit(prophet_df)
    fut = m.make_future_dataframe(periods=12, freq="M")
    fcst = m.predict(fut)
    fig, ax = plt.subplots(figsize=(10, 4))
    m.plot(fcst, ax=ax)
    st.pyplot(fig)
