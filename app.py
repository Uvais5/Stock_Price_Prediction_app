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
# File‑upload & common preprocessing                                           #
###############################################################################

st.title("📈 Simple Stock‑Price Prediction App (Prophet)")

uploaded_files = st.file_uploader(
    "Upload one or more CSV files exported from Yahoo Finance",
    accept_multiple_files=True,
    type="csv",
)

if not uploaded_files:
    st.info("⬆️ Upload a CSV file to get started.")
    st.stop()

# NOTE: we only keep the last uploaded dataset in global scope so that the
# sidebar pages all work with the same variables below. You can extend this to
# manage multiple symbols at once if needed.
for uploaded_file in uploaded_files:
    data = pd.read_csv(uploaded_file)

    # Prophet expects columns named ds (date) and y (value to forecast)
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

symbol_title = uploaded_file.name.split(".")[0]

###############################################################################
# Page implementations                                                         #
###############################################################################

def page_home():
    url = "https://facebook.github.io/prophet/"
    st.markdown(
        f"""
        ### Predict stock **closing prices** with [Prophet]({url})
        1. Download historical data (CSV) from *Yahoo Finance*.
        2. Upload it with the *Browse* button.
        3. Browse the interactive visualisations in the sidebar.
        """
    )
    st.image(Image.open("google.png"), caption="Search Yahoo Finance in Google")
    st.image(Image.open("gg.png"), caption="Download historical data")
    st.image(Image.open("aa.png"), caption="Upload CSV in the app")


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
    fig.update_layout(shapes=[{"type": "line", "x0": merged["ds"].min(), "y0": 0, "x1": merged["ds"].max(), "y1": 0, "line": {"dash": "dash"}}])
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
    st.write(
        "Data range:",
        data["Date"].iloc[0].strftime("%Y‑%m‑%d"),
        "→",
        data["Date"].iloc[-1].strftime("%Y‑%m‑%d"),
    )
    st.write("Forecast horizon ends:", forecast["ds"].iloc[-1].date())

    compare_date = st.date_input("Pick a date to compare")
    if compare_date:
        actual = data1.loc[data1["ds"] == pd.to_datetime(compare_date), "y"]
        predicted = forecast.loc[forecast["ds"] == pd.to_datetime(compare_date), "yhat"]
        st.write("Actual close price:", float(actual) if not actual.empty else "N/A")
        st.write("Predicted close price:", float(predicted) if not predicted.empty else "N/A")

###############################################################################
# Router                                                                       #
###############################################################################

if page == "Home":
    page_home()
elif page == "Historical Candlestick":
    page_candlestick()
elif page == "Forecast Line":
    page_forecast_line()
elif page == "Actual vs Predicted":
    page_actual_vs_pred()
elif page == "Residuals by Date":
    page_residuals()
elif page == "Error Histogram":
    page_histogram()
elif page == "Components (Year / Week)":
    page_components()
elif page == "Monthly Forecast (12 mo)":
    page_monthly()
elif page == "Compare Price":
    page_compare()

###############################################################################
# Footer                                                                       #
###############################################################################

# set_background("main1.jpg")  # uncomment if you want a background image
st.caption("📧 Contact: zaidsaifi523@gmail.com")
