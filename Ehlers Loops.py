import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Example Usage
tickers = ['AAPL', 'MSFT', 'NVDA', 'TSLA']  # List of financial instruments
ehlers = EhlersLoops(tickers=tickers, start_date='2024-09-01', end_date='2024-12-06', lp_period=10, hp_period=125)
ehlers.fetch_data()
ehlers.process_data()
ehlers.plot_2d_loops()

class EhlersLoops:
    def __init__(self, tickers, start_date, end_date, lp_period=10, hp_period=125):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.lp_period = lp_period
        self.hp_period = hp_period
        self.data = {}
        self.filtered_data = {}

    def fetch_data(self):
        """Fetch financial data for each ticker from Yahoo Finance."""
        for ticker in self.tickers:
            self.data[ticker] = yf.download(ticker, start=self.start_date, end=self.end_date)

    def roofing_filter(self, series, high_pass, low_pass):
        """Applies high-pass and low-pass filters to the data."""
        hp_alpha = np.exp(-np.sqrt(2) * np.pi / high_pass)
        high_passed = series.astype(float).copy()  # Ensure data type compatibility
        for i in range(2, len(series)):
            high_passed.iloc[i] = (
                (1 + hp_alpha * (1 - hp_alpha)) / 2
            ) * (series.iloc[i] - 2 * series.iloc[i - 1] + series.iloc[i - 2]) + 2 * hp_alpha * high_passed.iloc[i - 1] - (hp_alpha ** 2) * high_passed.iloc[i - 2]

        lp_alpha = np.exp(-np.sqrt(2) * np.pi / low_pass)
        low_passed = high_passed.copy()
        for i in range(2, len(high_passed)):
            low_passed.iloc[i] = (
                (1 - lp_alpha) ** 2
            ) * (high_passed.iloc[i] + high_passed.iloc[i - 1]) / 2 + 2 * lp_alpha * low_passed.iloc[i - 1] - (lp_alpha ** 2) * low_passed.iloc[i - 2]
        return low_passed

    def process_data(self):
        """Process data by applying filters and normalizing price and volume."""
        for ticker, df in self.data.items():
            df['Price'] = df['Close']
            df['Volume'] = df['Volume']

            # Apply roofing filter to price and volume
            df['Filtered_Price'] = self.roofing_filter(df['Price'], self.hp_period, self.lp_period)
            df['Filtered_Volume'] = self.roofing_filter(df['Volume'], self.hp_period, self.lp_period)

            # Normalize data
            df['Price_Scaled'] = (df['Filtered_Price'] - df['Filtered_Price'].mean()) / df['Filtered_Price'].std()
            df['Volume_Scaled'] = (df['Filtered_Volume'] - df['Filtered_Volume'].mean()) / df['Filtered_Volume'].std()

            # Store processed data
            self.filtered_data[ticker] = df.dropna()

    def plot_2d_loops(self):
        """Plot 2D Ehlers Loops for all tickers on a single graph."""
        plt.figure(figsize=(12, 10))

        # Calculate global limits for consistent scaling
        all_x = []
        all_y = []
        for ticker, df in self.filtered_data.items():
            all_x.extend(df['Volume_Scaled'].values)
            all_y.extend(df['Price_Scaled'].values)

        max_abs_x = max(abs(min(all_x)), abs(max(all_x)))
        max_abs_y = max(abs(min(all_y)), abs(max(all_y)))
        limit = max(max_abs_x, max_abs_y)

        for ticker, df in self.filtered_data.items():
            x = df['Volume_Scaled']  # Volume is now the X-axis
            y = df['Price_Scaled']   # Price is now the Y-axis

            # Plot arrows to indicate movement direction for each ticker
            for i in range(1, len(x)):
                plt.arrow(x.iloc[i - 1], y.iloc[i - 1], x.iloc[i] - x.iloc[i - 1], y.iloc[i] - y.iloc[i - 1],
                          color=plt.cm.tab10(self.tickers.index(ticker) % 10), alpha=0.5,
                          head_width=0.05, head_length=0.1, label=None)

            # Add label for the ticker
            plt.plot([], [], label=ticker, color=plt.cm.tab10(self.tickers.index(ticker) % 10))

        # Customize plot
        plt.axhline(0, color='red', linestyle='--', linewidth=1)  # Y-axis
        plt.axvline(0, color='red', linestyle='--', linewidth=1)  # X-axis
        plt.xlim(-limit, limit)
        plt.ylim(-limit, limit)
        plt.title("Ehlers Loops for All Tickers (Volume on X-axis, Price on Y-axis)")
        plt.xlabel("Scaled Volume")
        plt.ylabel("Scaled Price")
        plt.legend()
        plt.grid()
        plt.show()


