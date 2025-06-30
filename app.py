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
# Helper – optional background image                                          #
###############################################################################

def _get_base64(bin_file: str) -> str:
    with open(bin_file, "rb") as f:
        return base64.b64encode(f.read()).decode()


def set_background(png_file: str):
    """Set a full-screen background image inside Streamlit."""
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
# Load data – upload or download                                              #
###############################################################################

st.title("📈 Simple Stock-Price Prediction App (Prophet)")

# Initialize session state for downloaded data and symbol
# This is crucial so that the downloaded data persists across reruns
# without needing to re-download if the user navigates pages.
if 'data' not in st.session_state:
    st.session_state.data = None
if 'symbol_title' not in st.session_state:
    st.session_state.symbol_title = ""
if 'prophet_df' not in st.session_state:
    st.session_state.prophet_df = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'forecast' not in st.session_state:
    st.session_state.forecast = None

source = st.sidebar.selectbox(
    "Load data via…",
    (
        "Download from Yahoo Finance", # Prioritize download
        "Upload CSV",
    ),
)

# ────────────────────────────────────────────────────────────────────────────
# 1. Upload CSV
# ────────────────────────────────────────────────────────────────────────────
if source == "Upload CSV":
    uploaded = st.sidebar.file_uploader(
        "Upload Yahoo Finance CSV", type="csv", accept_multiple_files=False
    )
    if uploaded is not None:
        try:
            # Added parse_dates to ensure 'Date' column is parsed correctly
            uploaded_data = pd.read_csv(uploaded, parse_dates=['Date'])
            st.session_state.data = uploaded_data
            st.session_state.symbol_title = uploaded.name.split(".")[0].upper()
            st.success(f"✅ Uploaded {len(st.session_state.data)} rows for **{st.session_state.symbol_title}**")
            # Set a flag to trigger reprocessing
            st.session_state.data_loaded = True
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            st.session_state.data = None
            st.session_state.symbol_title = ""
            st.session_state.data_loaded = False
    else:
        st.info("Upload a CSV file or switch to *Download*.")
        st.session_state.data_loaded = False
        # If no file is uploaded, ensure previous data is cleared to prevent
        # using stale data if the user switches back and forth.
        st.session_state.data = None
        st.session_state.symbol_title = ""
        st.session_state.prophet_df = None
        st.session_state.model = None
        st.session_state.forecast = None

# ────────────────────────────────────────────────────────────────────────────
# 2. Download using yfinance (Ticker.history avoids timezone bug)
# ────────────────────────────────────────────────────────────────────────────
if source == "Download from Yahoo Finance":
    ticker = st.sidebar.text_input("Ticker", "AAPL", max_chars=8).upper().strip()
    col1, col2 = st.sidebar.columns(2)
    # Defaulting start date to a reasonable past date
    start_date = col1.date_input("Start date", pd.to_datetime("2023-01-01"))
    # Ensure end date is not in the future for current data
    end_date = col2.date_input("End date", pd.to_datetime("today"))
    fetch = st.sidebar.button("🔽 Fetch data")

    # Only fetch if the button is clicked OR if it's the first run and data isn't loaded
    if fetch or (st.session_state.data is None and st.session_state.symbol_title == "" and ticker == "AAPL"):
        if start_date >= end_date:
            st.error("❌ *Start* must be before *End*.")
            st.session_state.data_loaded = False
            st.stop()
        with st.spinner(f"Downloading {ticker} data from Yahoo Finance…"):
            try:
                ticker_obj = yf.Ticker(ticker)
                # Using .history() is good. Added interval="1d" for clarity.
                # Adding 1 day to end_date ensures that the end_date itself is included
                # as history() fetches up to (but not including) the end date.
                downloaded_data = ticker_obj.history(start=start_date, end=end_date + pd.Timedelta(days=1), interval="1d")
            except Exception as e:
                st.error(f"Download failed: {e}. Please check the ticker symbol or your internet connection.")
                st.session_state.data_loaded = False
                st.stop()

            if downloaded_data is None or downloaded_data.empty:
                st.error("No data returned – check symbol, date range, or try again later.")
                st.session_state.data_loaded = False
                st.stop()

            # history returns index as DatetimeIndex; reset to column
            downloaded_data.reset_index(inplace=True)
            # Rename 'Date' column to 'Date' if it's 'Datetime'
            if 'Datetime' in downloaded_data.columns:
                downloaded_data.rename(columns={'Datetime': 'Date'}, inplace=True)

            st.session_state.data = downloaded_data
            st.session_state.symbol_title = ticker
            st.success(f"✅ Downloaded {len(st.session_state.data)} rows for **{st.session_state.symbol_title}**")
            st.session_state.data_loaded = True # Flag to indicate data is ready for processing

# Guard: we need data to proceed
if st.session_state.data is None or st.session_state.data.empty:
    st.info("Please upload a CSV or fetch data from Yahoo Finance to proceed.")
    st.stop()

###############################################################################
# Prepare dataset for Prophet                                                 #
###############################################################################

# Only re-process data if new data is loaded or if it's the first time processing
###############################################################################
# Prepare dataset for Prophet                                                 #
###############################################################################

# Only re-process data if new data is loaded or if it's the first time processing
if st.session_state.data_loaded or st.session_state.prophet_df is None:
    data_to_process = st.session_state.data.copy()

    # Ensure 'Date' column is datetime
    data_to_process["Date"] = pd.to_datetime(data_to_process["Date"], errors="coerce")
    # Drop rows with missing dates or prices
    clean_data = data_to_process.dropna(subset=["Date", "Close"])

    if clean_data.empty:
        st.error("Dataset has no usable Date/Close columns after cleaning. Please check your data.")
        st.session_state.data_loaded = False # Reset flag as data isn't usable
        st.stop()

    prophet_df_cleaned = (
        clean_data[["Date", "Close"]]
        .rename(columns={"Date": "ds", "Close": "y"})
        .sort_values("ds")
        .reset_index(drop=True)
    )
    # --- IMPORTANT FIX: Remove timezone from 'ds' column for Prophet ---
    prophet_df_cleaned['ds'] = prophet_df_cleaned['ds'].dt.tz_localize(None)
    # -------------------------------------------------------------------

    st.session_state.prophet_df = prophet_df_cleaned
    # Reset the flag after processing
    st.session_state.data_loaded = False
else:
    prophet_df_cleaned = st.session_state.prophet_df # Use existing processed data

# Now, ensure `clean` is available for pages like `candlestick()`
clean = st.session_state.data.copy() # Use the raw downloaded/uploaded data
clean["Date"] = pd.to_datetime(clean["Date"], errors="coerce")
clean = clean.dropna(subset=["Date", "Close", "Open", "High", "Low"])


###############################################################################
# Fit Prophet model                                                           #
###############################################################################

# Only re-fit model if data changed or model isn't fitted yet
if st.session_state.model is None or st.session_state.forecast is None:
    with st.spinner("Fitting Prophet model and generating forecast…"):
        try:
            model = Prophet()
            model.fit(prophet_df_cleaned)
            future = model.make_future_dataframe(periods=365) # Forecast 1 year ahead
            forecast = model.predict(future)
            st.session_state.model = model
            st.session_state.forecast = forecast
            st.success("✅ Prophet model fitted and forecast generated!")
        except Exception as e:
            st.error(f"Error fitting Prophet model: {e}")
            st.session_state.model = None
            st.session_state.forecast = None
            st.stop()
else:
    # Use existing model and forecast
    model = st.session_state.model
    forecast = st.session_state.forecast

###############################################################################
# Sidebar navigation                                                          #
###############################################################################

page = st.sidebar.radio(
    "Navigate",
    (
        "Overview",
        "Historical Candlestick",
        "Forecast Line",
        "Actual vs Predicted",
        "Residuals",
        "Components (Year / Week)",
        "Monthly Forecast (12 mo)",
        "Compare Price", # This page is not implemented, consider adding or removing
    ),
)

###############################################################################
# Page functions                                                              #
###############################################################################

def overview():
    url = "https://facebook.github.io/prophet/"
    st.markdown(
        f"""
        ### Prophet forecast for `{st.session_state.symbol_title}`  
        **Rows**: {len(clean)}   •   **Date range**: {clean['Date'].min().date()} → {clean['Date'].max().date()}  
        **Model docs**: [Prophet]({url})
        """
    )
    st.dataframe(clean.head())


def candlestick():
    st.subheader(f"{st.session_state.symbol_title} – Historical prices")
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
    st.subheader("1-year forecast")
    st.plotly_chart(plot_plotly(model, forecast), use_container_width=True)


def actual_vs_pred():
    merged = pd.merge(
        prophet_df_cleaned, # Use the cleaned df for actuals
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
    merged = pd.merge(prophet_df_cleaned, forecast[["ds", "yhat"]], on="ds", how="inner")
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
    st.subheader("Monthly forecast (12 months)")
    # Re-fit a new model for monthly forecast to avoid altering the main model
    # It's good you're doing this with `m = Prophet(...)`
    # Also, ensure Prophet handles monthly data correctly by specifying `freq="MS"` (Month Start) or "M" (Month End)
    # The original model is `model`, so `m` is a new one for this specific plot.
    m = Prophet(changepoint_prior_scale=0.01)
    m.fit(prophet_df_cleaned) # Use the same cleaned data

    # make_future_dataframe for 12 months, starting from the last date in prophet_df_cleaned
    # freq='MS' for month start frequency, 'M' for month end frequency
    fut = m.make_future_dataframe(periods=12, freq="MS") # Adjusted frequency to 'MS' for clear monthly starts
    fcst = m.predict(fut)

    # Plotting the monthly forecast using plot_plotly
    fig_monthly = plot_plotly(m, fcst, trend=True, changepoints=True, uncertainty=True,
                              xlabel='Date', ylabel='Predicted Price')

    # You can also add actual data for context if needed
    fig_monthly.add_trace(go.Scatter(x=prophet_df_cleaned['ds'], y=prophet_df_cleaned['y'],
                                     mode='markers', name='Actual Data',
                                     marker=dict(color='blue', size=4)))

    st.plotly_chart(fig_monthly, use_container_width=True)


def compare_price():
    st.subheader("Compare Price (Under Development)")
    st.info("This section is intended for comparing stock prices but is not yet implemented.")
    # You would typically fetch data for another ticker and plot it here
    # Similar to how 'Download from Yahoo Finance' works, but for two tickers.
    # For example:
    # ticker2 = st.text_input("Compare with Ticker", "MSFT").upper().strip()
    # if st.button("Compare"):
    #     try:
    #         data2 = yf.download(ticker2, start=clean['Date'].min(), end=clean['Date'].max())
    #         data2.reset_index(inplace=True)
    #         fig = go.Figure()
    #         fig.add_trace(go.Scatter(x=clean["Date"], y=clean["Close"], name=st.session_state.symbol_title, mode="lines"))
    #         fig.add_trace(go.Scatter(x=data2["Date"], y=data2["Close"], name=ticker2, mode="lines"))
    #         st.plotly_chart(fig, use_container_width=True)
    #     except Exception as e:
    #         st.error(f"Could not compare: {e}")


###############################################################################
# Page routing                                                                #
###############################################################################

if page == "Overview":
    overview()
elif page == "Historical Candlestick":
    candlestick()
elif page == "Forecast Line":
    forecast_line()
elif page == "Actual vs Predicted":
    actual_vs_pred()
elif page == "Residuals":
    residuals()
elif page == "Components (Year / Week)":
    components()
elif page == "Monthly Forecast (12 mo)":
    monthly()
elif page == "Compare Price":
    compare_price()
