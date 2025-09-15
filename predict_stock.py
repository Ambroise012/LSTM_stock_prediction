import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import os
import time
import random
import requests
import pandas as pd
import yfinance as yf
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s: %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

# Parameters # set in config file
ticker = sys.argv[1] # get ticker from arg 
look_back = 30
epochs = 100
batch_size = 32
future_days = 4

########################################
##  extract data and data preprocess  ##
########################################

# class MarketstackProvider:
#     def __init__(self):
#         self.api_key = os.getenv("MARKETSTACK_API_KEY")
#         if not self.api_key:
#             raise ValueError("MARKETSTACK_API_KEY environment variable not set")
#         self.base_url = "http://api.marketstack.com/v1/"
#         self.max_retries = 3
#         self.min_delay = 1
#         self.max_delay = 5

#     def _make_request(self, endpoint, params=None):
#         params = params or {}
#         params.update({"access_key": self.api_key})

#         for attempt in range(self.max_retries):
#             try:
#                 response = requests.get(
#                     f"{self.base_url}{endpoint}",
#                     params=params,
#                     timeout=10
#                 )
#                 response.raise_for_status()
#                 return response.json()
#             except Exception as e:
#                 if attempt < self.max_retries - 1:
#                     delay = random.uniform(self.min_delay, self.max_delay)
#                     logger.warning(f"Attempt {attempt + 1} failed. Retrying in {delay:.1f}s...")
#                     time.sleep(delay)
#                 else:
#                     raise Exception(f"API request failed: {str(e)}")

#     def get_stock_data(self, ticker, start_date=None, end_date=None):
#         params = {
#             "symbols": ticker,
#             "limit": 1000,
#             "sort": "ASC"
#         }
#         if start_date:
#             params["date_from"] = start_date
#         if end_date:
#             params["date_to"] = end_date

#         data = self._make_request("eod", params)

#         if not data.get("data"):
#             return None

#         df = pd.DataFrame(data["data"])
#         df["date"] = pd.to_datetime(df["date"])
#         df.set_index("date", inplace=True)
#         df = df.sort_index()

#         df = df.rename(columns={
#             "open": "Open",
#             "high": "High",
#             "low": "Low",
#             "close": "Close",
#             "volume": "Volume"
#         })

#         return df[["Open", "High", "Low", "Close", "Volume"]]


# def fetch_and_save_stock_data(tickers, output_dir="stock_data"):
#     os.makedirs(output_dir, exist_ok=True)
#     results = {}
#     marketstack = MarketstackProvider()

#     for ticker in tickers:
#         logger.info(f"Fetching Yahoo Finance data for {ticker}...")
#         df_yf = yf.download(ticker, period="max", interval="1d")

#         if df_yf.empty:
#             logger.warning(f"No Yahoo Finance data for {ticker}")
#             results[ticker] = None
#             continue

#         df_yf = df_yf[["Open", "High", "Low", "Close", "Volume"]]
#         last_date = df_yf.index[-1].date()
#         today = datetime.today().date()

#         if last_date < today:
#             logger.info(f"Yahoo data stops at {last_date}, fetching Marketstack from {last_date+timedelta(days=1)} to {today}...")
#             try:
#                 df_ms = marketstack.get_stock_data(
#                     ticker,
#                     start_date=(last_date + timedelta(days=1)).strftime("%Y-%m-%d"),
#                     end_date=today.strftime("%Y-%m-%d")
#                 )
#                 if df_ms is not None and not df_ms.empty:
#                     df_yf = pd.concat([df_yf, df_ms])
#                     df_yf = df_yf[~df_yf.index.duplicated(keep="last")]  # éviter doublons
#             except Exception as e:
#                 logger.error(f"Failed to fetch Marketstack data for {ticker}: {e}")

#         # Save to CSV
#         csv_file = os.path.join(output_dir, f"{ticker.replace('/', '_')}.csv")
#         df_yf.to_csv(csv_file)
#         logger.info(f"Saved {len(df_yf)} rows for {ticker} in {csv_file}")

#         results[ticker] = df_yf

#     return results



# tickers = ["MSFT", "AI.PA", "SAN.PA", "AAPL", "ENGI.PA", "IBM", "ALO.PA", "GE"]
# stock_data = fetch_and_save_stock_data(tickers)

# for ticker, data in stock_data.items():
#     print(f"\nData for {ticker}:")
#     if data is not None:
#         print(f"Rows: {len(data)}, last date: {data.index[-1].date()}")
#         print(data.tail())
#     else:
#         print("No data available.")



# =========================
# Load and preprocess data
# =========================
df = pd.read_csv(f"stock_data/{ticker}.csv")
close_values = df[['Close']].values
scaler = MinMaxScaler()
scaled_values = scaler.fit_transform(close_values)

# =========================
# Create sequences
# =========================
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:(i + look_back), 0])
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

X, Y = create_dataset(scaled_values, look_back)

# =========================
# Train-test split
# =========================
trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.2, shuffle=False)
trainX = trainX.reshape((trainX.shape[0], trainX.shape[1], 1))
testX = testX.reshape((testX.shape[0], testX.shape[1], 1))

# =========================
# Build LSTM model
# =========================
model = Sequential([
    LSTM(50, input_shape=(look_back, 1)),
    Dense(1)
])
model.compile(loss='mean_squared_error', optimizer='adam')

# =========================
# Train model
# =========================
model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=1, validation_data=(testX, testY))

# =========================
# Plot next 5 days prediction
# =========================
# Ensure Date column is datetime
df["Date"] = pd.to_datetime(df["Date"])

# Last 30 days for plotting
df_plot = df.tail(30).copy()

# Forecast future days starting after the last row
last_sequence = scaled_values[-look_back:].reshape(1, look_back, 1)
future_predictions = []

for _ in range(future_days):
    next_pred = model.predict(last_sequence)
    future_predictions.append(next_pred[0,0])
    next_pred_reshaped = next_pred.reshape(1,1,1)
    last_sequence = np.append(last_sequence[:,1:,:], next_pred_reshaped, axis=1)

future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1,1))
future_dates = [df["Date"].iloc[-1] + pd.Timedelta(days=i+1) for i in range(future_days)]

# Plot last 30 days + forecast
# Ensure the 'predict' folder exists
os.makedirs("predict", exist_ok=True)

# Plot last 30 days + forecast
plt.figure(figsize=(12,6))
plt.plot(df_plot["Date"], df_plot["Close"], label="Last 30 Days Actual", linewidth=2)
plt.plot(future_dates, future_predictions, label="Future Forecast", linestyle="--", color="red")
plt.title(f"{ticker} Stock Price - Last 30 Days + {future_days} Days Forecast")
plt.xlabel("Date")
plt.ylabel("Close Price ($)")
plt.grid(True)
plt.legend()

# Save the figure in the 'predict' folder
plt.savefig(f"predict/{ticker}_forecast.png", bbox_inches="tight")

# Optionally close the plot to free memory
plt.close()

