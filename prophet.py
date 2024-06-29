import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
from functools import reduce

import warnings
warnings.filterwarnings('ignore')
 
plt.style.use('seaborn-deep')
pd.options.display.float_format = "{:,.2f}".format

stock_price =  pd.read_csv('^GSPC.csv',parse_dates=['Date'])
stock_price.info()
stock_price.describe()
stock_price = stock_price[['Date','Close']]
stock_price.columns = ['ds', 'y']
stock_price.head(10)
stock_price.set_index('ds').y.plot(figsize=(12,6), grid=True);
model = Prophet()
model.fit(stock_price)

future = model.make_future_dataframe(1095, freq='d')

future_boolean = future['ds'].map(lambda x : True if x.weekday() in range(0, 5) else False)
future = future[future_boolean] 

future.tail()

forecast = model.predict(future)
forecast.tail()

model.plot(forecast)
model.plot_components(forecast)
stock_price_forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
df = pd.merge(stock_price, stock_price_forecast, on='ds', how='right')
df.set_index('ds').plot(figsize=(16,8), color=['royalblue', "#34495e", "#e74c3c", "#e74c3c"], grid=True)