import numpy as np 
import pandas as pd 
import  matplotlib.pyplot as plt
import streamlit as st
from fbprophet import Prophet
from io import StringIO
import plotly.graph_objects as go 
from PIL import Image
from fbprophet.plot import plot_plotly,plot_components_plotly
import base64

#streamlit title
st.title("Simple Stock Price Prediction App")
#upload function for csv uploader
uploaded_files = st.file_uploader("Choose a CSV file", accept_multiple_files=True)# only accept csv
for uploaded_file in uploaded_files:# when csv insert to our app
    data = pd.read_csv(uploaded_file)# now we read our file.

    data["Date"] = pd.to_datetime(data["Date"])#Then we use datetime function for to read date
    data1 = data[["Date","Close"]]# Then i chose only two columns 
    data1 = data1.rename(columns = {"Date" : "ds", "Close" : "y"})# i need to rename the column because ds and y is like parameter for fbprohit model
    model = Prophet()# Define our model
    model.fit(data1)

    Predict = model.make_future_dataframe(periods = 365)
    forcast = model.predict(Predict)


def main():
    page = st.sidebar.selectbox(
        "Select the option",
        ("Homepage","Normal_graph","Predict Graph1","Monthly Prediction \n 1 year time frame","Predictive_Graph_yearly and weakly","Compare_Price"),
    )

    if page == "Homepage":
        homepage()
    elif page == "Normal_graph":
        line_graph()
    elif page == "Predict Graph1":
        predict()
    elif page == "Predict Graph1":
        predict()
    elif page == "Monthly Prediction \n 1 year time frame":
        predict1()
    elif page == "Predictive_Graph_yearly and weakly":
        predict2()
    elif page == "Compare_Price":
        compare_Price()



def homepage():
    url = "https://facebook.github.io/prophet/"
    
    
    st.write("""
        ### You can predict the stock price (closing price)  using this app****  
        we use [Fbprophite](%s) to predict the stock price.
        how Facebook Prophite works just click on the link and you can see how its works  
        #""" % url)
  
    st.write("""
    # How to use this app 
    """)
    st.write("1. Search  yahoo finance on google and open the first link")
    image = Image.open('google.png')
    st.image(image, caption='IN chrome')
    st.write("2. In yahoo finance search bar, Search any company's stock you want to buy and click the company data.")
    st.write("Then click historical data and download the data")
    image = Image.open('gg.png')
    st.image(image, caption='')
    st.write("3. Then click the browse file and add the data ")
    image = Image.open("aa.png")
    st.image(image)



def line_graph():
    st.write("""
    # {} This graph is only show past data
    """.format(uploaded_file.name))
    
    figure = go.Figure(data=[go.Candlestick(x=data["Date"],
                                         open=data["Open"],high=data["High"],
                                        low=data["Low"], close=data["Close"])])
    figure.update_layout(title = "{} Stock Price Analysis".format(uploaded_file.name), xaxis_rangeslider_visible=False)
    figure.show()
    st.plotly_chart(figure,use_container_width=True)

    
def predict():
   st.write("""
   # This graph is show predictive moves
   ## how {} company rose or fall in future
   """.format(uploaded_file.name))
   p = plot_plotly(model,forcast)
   st.plotly_chart(p,use_container_width=True)
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
    st.write("## {} monthly and weakly data".format(uploaded_file.name))
    fig2 = plot_components_plotly(model,forcast)
    st.plotly_chart(fig2)
# now i crate functiom por compare stock price 
def compare_Price():
    st.title("________________________________")
    st.write("This is the starting date of this data",data["Date"].iloc[0],"and this is the last date of this data",data["Date"].iloc[-1])
    st.write("This is the last date of predition",forcast["ds"].iloc[-1])
    st.title("________________________________")
    input22 = st.date_input(label="Particular date for compare price")
    input = str(input22)
    normal = data1[data1.ds == input ]["y"]
    predictive = forcast[forcast.ds == input ]["yhat"]
    # st.write(input)
    st.write(normal,"y means actual price(close)")
    st.write(predictive,"yhat means predict price(close)")
    st.write("# y and ywhat is price (close)")


if __name__=='__main__':
    main()
     ############# function for bacgroud wallper instreamlit ################################
    def get_base64(bin_file):
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    def set_background(png_file):
        bin_str = get_base64(png_file)
        page_bg_img = '''
    <style>
    .stApp {
      background-image: url("data:image/png;base64,%s");
      background-size: cover;
    }
    </style>
        ''' % bin_str
        st.markdown(page_bg_img, unsafe_allow_html=True)

    set_background('main1.jpg')
    st.write(" # contact info : zaidsaifi523@gmail.com")
    
