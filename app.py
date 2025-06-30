import base64
from io import StringIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from prophet import Prophet
from prophet.plot import plot_components_plotly, plot_plotly
import yfinance as yf  # <-- NEW: automatic data downloader

###############################################################################
# Helper – optional background image                                           #
###############################################################################

def _get_base64(bin_file: str) -> str:
    with open(bin_file, "rb") as f:
        return base64.b64encode(f.read()).decode()


def set_background(png_file: str):
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
# Sidebar – choose data source                                                 #
###############################################################################

st.title("📈 Simple Stock‑Price Prediction App (Prophet)")

source = st.sidebar.selectbox("How would you like to load data?", ("Upload CSV", "Download from Yahoo Finance"))

data = None  # will hold the final DataFrame used everywhere
symbol_title = ""

if source == "Upload CSV":
    uploaded_files = st.sidebar.file_uploader("Upload CSV file(s)", accept_multiple_files=True, type="csv")

    if not uploaded_files:
        st.info("⬆️ Upload a CSV file to get started, **or** switch to the download option in the sidebar.")
        st.stop()

    # Keep only the last uploaded file as the active dataset
    uploaded_file = uploaded_files[-1]
    symbol_title = uploaded_file.name.split(".")[0]
    data = pd.read_csv(uploaded_file)

elif source == "Download from Yahoo Finance":
    ticker = st.sidebar.text_input("Ticker symbol", value="AAPL", max_chars=10)
    col1, col2 = st.sidebar.columns(2)
    start_date = col1.date_input("Start", value=pd.to_datetime("2024-01-01"))
    end_date = col2.date_input("End", value=pd.to_datetime("today"))

    fetch_clicked = st.sidebar.button("🔽 Fetch data")

    if fetch_clicked:
        if start_date >= end_date:
            st.error("Start date must be before end date.")
            st.stop()
        try:
            df = yf.download(ticker.upper(), start=start_date, end=end_date, interval="1d", auto_adjust=False)
        except Exception as e:
            st.error(f"Download failed: {e}")
            st.stop()
        if df.empty:
            st.error("No data returned – check the ticker symbol and date range.")
            st.stop()
        df.reset_index(inplace=True)
        data = df
        symbol_title = ticker.upper()
        st.success(f"Downloaded {len(data)} rows for {symbol_title} ✨")

if data is None:
    st.stop()  # nothing to work with

###############################################################################
# Prophet model fit & forecast                                                 #
###############################################################################

data["Date"] = pd.to_datetime(data["Date"])

data1 = (
    data[["Date", "Close"]]
    .rename(columns={"Date": "ds", "Close": "y"})
    .sort_values("ds")
    .reset_index(drop=True)
)

model = Prophet()
model.fit(data1)

future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)

###############################################################################
# Sidebar navigation                                                           #
###############################################################################

page = st.sidebar.radio(
    "Navigate",
    (
        "Home",
        "Historical Candlestick",
        "Forecast Line",
        "Actual vs Predicted",
        "Residuals by Date",
        "Error Histogram",
        "Components (Year / Week)",
        "Monthly Forecast (12 mo)",
        "Compare Price",
    ),
)

###############################################################################
# Page implementations                                                         #
###############################################################################

def page_home():
    url = "https://facebook.github.io/prophet/"
    st.markdown(
        f"""
        ### Predict stock **closing prices** with [Prophet]({url})

        **Data loaded for:** `{symbol_title}`  
        Rows: **{len(data)}**   Date range: **{data['Date'].min().date()} → {data['Date'].max().date()}**

        ---
        #### Data‑loading options
        * **Upload CSV** – any file with Yahoo Finance column layout.
        * **Download** – enter a ticker and date range in the sidebar; the app pulls data automatically with [`yfinance`](https://github.com/ranaroussi/yfinance).
        """
    )
    st.write("Preview:")
    st.dataframe(data.head())


def page_candlestick():
    st.subheader(f"Historical prices – {symbol_title}")
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=data["Date"],
                open=data["Open"],
                high=data["High"],
                low=data["Low"],
                close=data["Close"],
            )
        ]
    )
    fig.update_layout(title=f"{symbol_title} – Price history", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)


def page_forecast_line():
    st.subheader(f"12‑month forecast – {symbol_title}")
    fig = plot_plotly(model, forecast)
    st.plotly_chart(fig, use_container_width=True)


def page_actual_vs_pred():
    st.subheader("Actual vs predicted close price")
    merged = pd.merge(
        data1,
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
            fillcolor="rgba(0,100,80,0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.update_layout(showlegend=True)
    st.plotly_chart(fig, use_container_width=True)


def page_residuals():
    st.subheader("Prediction residuals (actual − predicted)")
    merged = pd.merge(data1, forecast[["ds", "yhat"]], on="ds", how="inner")
    merged["residual"] = merged["y"] - merged["yhat"]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=merged["ds"], y=merged["residual"], mode="lines+markers", name="Residual")
    )
    fig.update_layout(
        shapes=[
            {
                "type": "line",
                "x0": merged["ds"].min(),
                "y0": 0,
                "x1": merged["ds"].max(),
                "y1": 0,
                "line": {"dash": "dash"},
            }
        ]
    )
    st.plotly_chart(fig, use_container_width=True)


def page_histogram():
    st.subheader("Distribution of residuals")
    merged = pd.merge(data1, forecast[["ds", "yhat"]], on="ds", how="inner")
    merged["residual"] = merged["y"] - merged["yhat"]

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=merged["residual"], nbinsx=50))
    st.plotly_chart(fig, use_container_width=True)


def page_components():
    st.subheader("Yearly & weekly components")
    fig = plot_components_plotly(model, forecast)
    st.plotly_chart(fig, use_container_width=True)


def page_monthly():
    st.subheader("Monthly forecast (next 12 months)")
    m = Prophet(changepoint_prior_scale=0.01).fit(data1)
    fut = m.make_future_dataframe(periods=12, freq="M")
    fcst = m.predict(fut)
    fig = m.plot(fcst)
    plt.title("Monthly forecast – 12 months")
    st.pyplot(fig)


def page_compare():
    st.subheader("Compare actual vs predicted price on a specific date")
    compare_date = st.date_input("Pick a date to compare", value=data["Date"].iloc[-1].date())
    if compare_date:
        actual = data1.loc[data1["ds"] == pd.to_datetime(compare_date), "y"]
        predicted = forecast.loc[forecast["ds"] == pd.to_datetime(compare_date), "yhat"]
        st.write("Actual close:", float(actual) if not actual.empty else "N/A")
        st.write("Predicted close:", float(predicted) if not predicted.empty else "N/A")

###############################################################################
# Router                                                                       #
