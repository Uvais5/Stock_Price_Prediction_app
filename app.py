
from matplotlib import ticker
import numpy as np 
import pandas as pd 
import  matplotlib.pyplot as plt
import streamlit as st
from fbprophet import Prophet
from io import StringIO
import plotly.graph_objects as go 
from PIL import Image

st.title("Simple Stock Price Prediction App")

uploaded_files = st.file_uploader("Choose a CSV file", accept_multiple_files=True)
for uploaded_file in uploaded_files:
    bytes_data = uploaded_file.getvalue()
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    string_data = stringio.read()
    data = pd.read_csv(uploaded_file)

    data["Date"] = pd.to_datetime(data["Date"])
    data1 = data[["Date","Close"]]
    data1 = data1.rename(columns = {"Date" : "ds", "Close" : "y"})
    model = Prophet()
    model.fit(data1)

    Predict = model.make_future_dataframe(periods = 365)
    forcast = model.predict(Predict)


def main():
    page = st.sidebar.selectbox(
        "Select the mode",
        ("Homepage","Line_graph","graph","Predict Graph1","Monthly Prediction \n 1 year time frame","Predictive Graph2"),
    )

    if page == "Homepage":
        homepage()
    elif page == "Line_graph":
        line_graph()
    elif page == "graph":
        graph()
    elif page == "Predict Graph1":
        predict()
    elif page == "Monthly Prediction \n 1 year time frame":
        predict1()
    elif page == "Predictive Graph2":
        predict2()



def homepage():
    url = "https://facebook.github.io/prophet/"

    st.write("""
        ### Shown are the stock **closing price** and we can predict the stock price if we use this app 
        we use [Fbprophite](%s) to predict the stock price.
        how Facebook Prophite works just click the link and check how its works  
        #""" % url)
    st.write("""
    # How to use this app 
    """)
    st.write("1. Search  yahoo finance on google and open first website")
    image = Image.open('google.png')
    st.image(image, caption='IN chrome')
    st.write("2. Search any company stock you want to buy in yahoo finance search bar and click the company data.")
    st.write("Then click historical data and download the data")
    image = Image.open('gg.png')
    st.image(image, caption='')
    st.write("3. Then click the browse file and give the data ")
    image = Image.open("aa.png")
    st.image(image)



def line_graph():
    st.write("""
    # {} This graph is only show past data
    ## "y" mean close 
    """.format(uploaded_file.name))
    st.line_chart(data1.y)
    figure = go.Figure(data=[go.Candlestick(x=data["Date"],
                                         open=data["Open"],high=data["High"],
                                        low=data["Low"], close=data["Close"])])
    figure.update_layout(title = "{} Stock Price Analysis".format(uploaded_file.name), xaxis_rangeslider_visible=False)
    figure.show()
    st.plotly_chart(figure,use_container_width=True)

def graph():

    chart = pd.DataFrame(forcast,
    columns=["yhat","yhat_lower","yhat_upper"])
    date = forcast.ds
    yy = forcast.yhat
    st.line_chart(chart)
    
def predict():
   st.write("""
   # This graph is show predictive moves
   ## how {} company grow or fall in future
   """.format(uploaded_file.name))
   graph = model.plot(forcast)
   st.pyplot(graph)
   figure = go.Figure(data=[go.Candlestick(x=forcast["ds"],
                                         open=forcast["trend"],high=forcast["yhat_upper"],
                                        low=forcast["yhat_lower"], close=forcast["yhat"])])
   figure.update_layout(title = "{} Stock Price Prediction".format(uploaded_file.name), xaxis_rangeslider_visible=False)
   figure.show()
   st.plotly_chart(figure,use_container_width=True)


def predict1():
   st.write("""
   # This graph show how {} progress in future
   ## Either price is up or down
   """.format(uploaded_file.name))
   m = Prophet(changepoint_prior_scale=0.01).fit(data1)
   future = m.make_future_dataframe(periods=12, freq='M')
   fcst = m.predict(future)
   fig = m.plot(fcst)
   plt.title("Monthly Prediction \n 1 year time frame")
   st.pyplot(fig)

def predict2():
    fig2 = model.plot_components(forcast)
    st.pyplot(fig2)




if __name__=='__main__':
    main()
