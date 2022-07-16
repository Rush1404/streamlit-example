#open source framework that helps with GUI
import streamlit as st
#date api
from datetime import date
#finance api
import yfinance as yf
#prediction api
from fbprophet import Prophet

TODAY = date.datetime.now()
START = date.datetime(TODAY.year - 10, TODAY.month, TODAY.day)

st.title("Stockfest - Stock Prediction")

stocks = ("GOOG", "AAPL")
select_stock = st.selectbox("Select the stock for prediction", stocks)

t_years = st.slider("Days of prediction:", 1 , 365)

@st.cache
def generate_data(stock):
  data = yf.download(stock, START, TODAY)
  data.reset_index(inplace = True)
  return data

data_loading = st.text("Loading....")
data = generate_data(select_stock)
data_loading.text("....done!")

st.subheader("Stock History")
st.write(data.tail())

model = Prophet()
data = data.reset_index()
data[["ds", "y"]] = data[["Date", "Adj Close"]]
model.fit(data)

future = model.make_future_dataframe(periods=365)
prediction = model.predict(future)
model.plot(prediction)
