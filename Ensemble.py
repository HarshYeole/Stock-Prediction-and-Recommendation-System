import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from ta.trend import MACD
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta

# Function to get historical stock data
def get_stock_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date)
    return df

# Function to calculate Bollinger Bands
def calculate_bollinger_bands(df, window=20, num_std=2):
    df['Rolling Mean'] = df['Close'].rolling(window=window).mean()
    df['Upper Band'] = df['Rolling Mean'] + (df['Close'].rolling(window=window).std() * num_std)
    df['Lower Band'] = df['Rolling Mean'] - (df['Close'].rolling(window=window).std() * num_std)
    return df

# Function to generate features
def generate_features(df):
    # MACD
    df['macd'] = MACD(df['Close']).macd()

    # Bollinger Bands
    df = calculate_bollinger_bands(df)

    # Linear Regression Coefficients
    df['LR_slope'], df['LR_intercept'] = linear_regression_features(df)

    return df

# Function to calculate linear regression coefficients
def linear_regression_features(df):
    X = np.arange(len(df)).reshape(-1, 1)
    y = df['Close'].values
    lr_model = LinearRegression().fit(X, y)
    return lr_model.coef_[0], lr_model.intercept_

# Main function
def main():
    ticker = 'HDFCBANK.NS'
    start_date = '2023-03-07'
    end_date = '2024-03-12'

    # Get historical stock data
    stock_data = get_stock_data(ticker, start_date, end_date)

    # Generate features
    stock_data = generate_features(stock_data)

    # Split data into features and labels
    features = stock_data.drop(['Close'], axis=1)
    labels = stock_data['Close']

    # Train HistGradientBoostingRegressor
    rf_regressor = HistGradientBoostingRegressor().fit(features, labels)

    # Predict for the next 30 days
    next_month_start = datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)
    next_month_end = next_month_start + timedelta(days=29)
    next_month_data = get_stock_data(ticker, next_month_start.strftime('%Y-%m-%d'), next_month_end.strftime('%Y-%m-%d'))
    next_month_features = generate_features(next_month_data)
    next_month_predicted_close = rf_regressor.predict(next_month_features.drop(['Close'], axis=1))

    # Repeat the predictions to fill 30 days
    repeated_predictions = np.repeat(next_month_predicted_close, 6)

    # Generate predicted dates for the next 30 days
    predicted_dates = pd.date_range(start=next_month_start, end=next_month_end, freq='D')

    # Print predicted price for the next day
    print("Predicted Closing Price for the Next 30 Days:")
    for i, pred_date in enumerate(predicted_dates):
        print(f"{pred_date.strftime('%Y-%m-%d')}: {repeated_predictions[i]:.2f}")

    # Plot actual values
    plt.figure(figsize=(12, 6))
    plt.plot(stock_data.index, stock_data['Close'], label='Actual Close Price', color='blue')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.grid(True)

    # Plot predicted values for the next 30 days
    plt.plot(predicted_dates, repeated_predictions, label='Predicted Close Price', color='green')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
