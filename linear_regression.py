def predict():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import matplotlib.pyplot as plt

    # Load data
    data = pd.read_csv('IBM1.csv', parse_dates=['Date'])

    # Prepare shuffled data
    X = data[['Open', 'High', 'Low', 'Volume']]
    y = data['Close']

    # Split shuffled data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Create and train the model (e.g., Linear Regression)
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on the shuffled test set
    y_pred = model.predict(X_test)

    # Calculate Mean Absolute Error (MAE)
    mae = mean_absolute_error(y_test, y_pred)

    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(y_test, y_pred)

    # Calculate Root Mean Squared Error (RMSE)
    rmse = mse ** 0.5

    # Calculate the mean of 'Close' values in the shuffled test set
    mean_close = y_test.mean()

    # Calculate accuracy in percentage using shuffled test set
    accuracy_percentage1 = 100 * (1 - (mae / mean_close))
    accuracy_percentage2 = 100 * (1 - (mse / mean_close))
    accuracy_percentage3 = 100 * (1 - (rmse / mean_close))

    print("mean absolute error: ", mae)
    print("mean squared error: ", mse)
    print("root mean squared error: ", rmse, "\n")

    print("Accuracy MAE(%):", accuracy_percentage1)
    print("Accuracy MSE(%):", accuracy_percentage2)
    print("Accuracy RMSE(%):", accuracy_percentage3)
    print(y_pred)

    # Plotting predicted closing price against the actual dates
    A = data['Date'].tail(len(y_pred))
    B = y_pred
    plt.plot(A, B)
    plt.xlabel('Date')
    plt.ylabel('Predicted Close Price')
    plt.title('Predicted Close Price against Date')
    plt.legend()
    plt.show()

predict()
