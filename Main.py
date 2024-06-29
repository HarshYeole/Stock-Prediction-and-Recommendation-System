
# Function to generate the RSI
    def select_RSI_button_clicked(self):
        # Get the selected company from the UI (replace this with your actual method to get the selected company)
        selected_company = self.get_selected_company()  # You need to implement this method

        # Print the selected company
        print(f"Selected Company: {selected_company}")

        # Call the plot_stock_with_rsi function with the selected company
        self.plot_stock_with_rsi(selected_company)

    # Function to plot stock with RSIs
    def plot_stock_with_rsi(self, symbol, rsi_period=14):
        # Fetch historical price data
        stock_data = self.get_historical_data(symbol)

        # Calculate RSI and generate signals
        rsi_values = self.calculate_rsi(stock_data['Close'], period=rsi_period)
        stock_data['RSI'] = rsi_values
        stock_data['Signal'] = self.generate_signals(rsi_values)
        print(stock_data)

        # Plotting (same as your existing code)
        plt.figure(figsize=(10, 6))
        plt.plot(stock_data.index, stock_data['Close'], label='Close Price', linewidth=2)
        plt.title(f'{symbol} Stock Price and RSI')
        plt.xlabel('Date')
        plt.ylabel('Closing Price')
        plt.legend(loc='upper left')

        ax2 = plt.gca().twinx()
        ax2.plot(stock_data.index, stock_data['RSI'], label='RSI', color='orange', linestyle='dashed')
        ax2.set_ylabel('RSI', color='orange')
        ax2.legend(loc='upper right')

        buy_signals = stock_data[stock_data['Signal'] == 'BUY']
        sell_signals = stock_data[stock_data['Signal'] == 'SELL']
        plt.scatter(buy_signals.index, buy_signals['Close'], marker='^', color='g', label='Buy Signal')
        plt.scatter(sell_signals.index, sell_signals['Close'], marker='v', color='r', label='Sell Signal')

        plt.legend()
        plt.show()

    # Helper method to get historical data
    def get_historical_data(self, symbol):
        start_date = '2023-01-01'
        end_date = '2024-01-1'
        comp_data = yf.download(symbol, start=start_date, end=end_date)
        return comp_data

    # Helper method to calculate RSI
    def calculate_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    # Helper method to generate trading signals
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
           
    start_date = '2023-01-01'
    end_date = '2024-01-1'
    plot_stock_with_rsi('GOOG', start_date, end_date)