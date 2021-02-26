# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 12:54:00 2021

@author: khasy
"""

import streamlit as st
from datetime import date

import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Forecast App')

stocks = ('GRN.TO','CVE.TO','MTRX.V','REAL.TO','RNW.TO')
selected_stocks = st.selectbox('Select dataset for prediction', stocks)

num_years = st.slider('Years of prediction:', 1, 3)
period = num_years * 365

@st.cache
def load_data(ticker_sym):
    data = yf.download(ticker_sym, START, TODAY)
    data.reset_index(inplace = True)
    return data

data_load_state = st.text('Loading data...')
data = load_data(selected_stocks)
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(data.tail())

#Plot Raw Data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = data['Date'], y = data['Open'], name = 'stock_open'))
    fig.add_trace(go.Scatter(x = data['Date'], y = data['Close'], name = 'stock_close'))
    fig.layout.update(title_text = 'Time Series Data', xaxis_rangeslider_visible = True)
    st.plotly_chart(fig)

plot_raw_data()

#Forecasting with Prophet
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns = {'Date': 'ds', 'Close': 'y'})

model = Prophet()
model.fit(df_train)
future = model.make_future_dataframe(periods = period)
forecast = model.predict(future)

#Show and Plot Forecast
st.subheader('Forecase Data')
st.write(forecast.tail())

st.write(f'Forecase plot for {num_years} years')
fig1 = plot_plotly(model, forecast)
st.plotly_chart(fig1)

st.write('Forecase Components')
fig2 = model.plot_components(forecast)
st.write(fig2)

