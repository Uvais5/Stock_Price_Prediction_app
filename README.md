# 📈 Stock Price Prediction Web App (with Prophet)

![Stock Prediction](https://github.com/Uvais5/Stock_Price_Prediction_app/blob/main/stock.png)

> A sleek, interactive **Streamlit** application for predicting stock prices using **Facebook Prophet**. Simply upload your stock CSV file, and visualize future trends, monthly/weekly breakdowns, and compare real vs predicted prices. Ideal for learning, experimenting, and showcasing time series forecasting with a modern UI.

---

### 🚀 Live Demo

👉 **[Try the App Now](https://share.streamlit.io/uvais5/stock_price_prediction_app/main/app.py)**

---

### 🧠 Use Case & Goal

This app is built for:

- 🧪 **Experimentation with forecasting models** (Prophet).
- 📚 **Educational use** to understand stock prediction pipelines.
- 💡 **Learning time series visualization** using `Plotly`, `Streamlit`, and real-world data.
- 🔬 **Comparing predicted vs actual stock prices** on specific dates.

> ⚠️ This project is **for educational and demonstration purposes only**, not intended for financial advice or trading.

---

### 🛠 Tech Stack & Tools

| Tool        | Purpose                        |
|-------------|--------------------------------|
| 🐍 Python    | Core programming language      |
| 📈 Prophet   | Time series forecasting        |
| 📊 Plotly    | Rich interactive visualizations|
| 🎨 Streamlit | Web-based ML App UI            |
| 🖼 Pillow    | Displaying images              |
| 📂 Pandas    | Data manipulation              |

---

### 📂 Features

- Upload historical **CSV stock data** from Yahoo Finance.
- Build & train **Prophet** forecasting model.
- Interactive graphs for:
  - ✅ Historical candlestick chart
  - ✅ Future predictions
  - ✅ Monthly/yearly/weekly component analysis
  - ✅ Compare predicted price vs actual price on a selected date
- 📅 Predict 365 days into the future
- 🌄 Stylish background & onboarding guide

---

### 🔧 Installation

```bash
# 1. Clone the repo
git clone https://github.com/Uvais5/Stock_Price_Prediction_app.git
cd Stock_Price_Prediction_app

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate    # On Windows use: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
# If requirements.txt is missing, install manually:
# pip install streamlit fbprophet pandas plotly pillow
