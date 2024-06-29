import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
#from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from ta.trend import MACD
import matplotlib.pyplot as plt
import yfinance as yf  


def get_historical_data(self, symbol, start_date, end_date):
    tsla_data = yf.download(symbol, start=start_date, end=end_date)
    return tsla_data

def calculate_rsi(self, prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def generate_signals(self, rsi_values):
    signals = []
    for rsi in rsi_values:
        if rsi > 70:
            signals.append('SELL')
        elif rsi < 30:
            signals.append('BUY')
        else:
            signals.append('HOLD')
    return signals

def plot_stock_with_rsi(self, symbol, start_date, end_date, rsi_period=14):
        # Fetch historical price data
        stock_data = self.get_historical_data(symbol, start_date, end_date)

        # Calculate RSI and generate signals
        rsi_values = self.calculate_rsi(stock_data['Close'], period=rsi_period)
        stock_data['RSI'] = rsi_values
        stock_data['Signal'] = self.generate_signals(rsi_values)

        # Update the predicted_values_area with RSI values and company name
        rsi_output = f"Selected Company: {symbol}\n\nDate\t\tRSI\tSignal\n"
        for date, rsi, signal in zip(stock_data.index, rsi_values, stock_data['Signal']):
            rsi_output += f"{date.strftime('%Y-%m-%d')}\t\t{rsi:.2f}\t{signal}\n"

        self.predicted_values_area.setPlainText(rsi_output)