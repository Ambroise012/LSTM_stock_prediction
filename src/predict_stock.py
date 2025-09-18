import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 
import sys
import time
import json
import random
import logging
from datetime import datetime, timedelta

import requests
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import Input
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

from src.config import config
from src.predict_utils import get_company_name, fetch_stock_data, create_dataset

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s: %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

# Parameters # set in config file
ticker = sys.argv[1] # get ticker from arg 

# Run fetch
df_ticker = fetch_stock_data(ticker)

if isinstance(df_ticker, dict):
    df_ticker = df_ticker.get(ticker)

if df_ticker is None or df_ticker.empty:
    logger.error(f"No data avaible for {ticker}.")
    sys.exit(1)

df_plot = df_ticker.tail(30).copy()
future_dates = [df_ticker.index[-1] + pd.Timedelta(days=i+1) for i in range(config.predict.future_days)]

# =========================
# Load and preprocess data
# =========================

close_values = df_ticker[['Close']].values
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

X, Y = create_dataset(scaled_values, config.predict.look_back)

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
    Input(shape=(config.predict.look_back, 1)),
    LSTM(50),
    Dense(1)
])
model.compile(loss='mean_squared_error', optimizer='adam')

# =========================
# Train model
# =========================
model.fit(trainX, trainY, epochs=config.LSTM.epochs, batch_size=config.LSTM.batch_size, verbose=1, validation_data=(testX, testY))

# =========================
# Plot next days prediction
# =========================
last_sequence = scaled_values[-config.predict.look_back:].reshape(1, config.predict.look_back, 1)
future_predictions = []
for _ in range(config.predict.future_days):
    next_pred = model.predict(last_sequence)
    future_predictions.append(next_pred[0, 0])
    next_pred_reshaped = next_pred.reshape(1, 1, 1)
    last_sequence = np.append(last_sequence[:, 1:, :], next_pred_reshaped, axis=1)

future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
future_dates = [df_ticker.index[-1] + pd.Timedelta(days=i+1) for i in range(config.predict.future_days)]

# Plot
os.makedirs("predict", exist_ok=True)
plt.figure(figsize=(12, 6))
plt.plot(df_ticker["Close"].tail(30).index, df_ticker["Close"].tail(30).values, label="Last 30 Days Actual", linewidth=2)
plt.plot(future_dates, future_predictions, label="Future Forecast", linestyle="--", color="red")
plt.title(f"{ticker} Stock Price - Last 30 Days + {config.predict.future_days} Days Forecast")
plt.xlabel("Date")
plt.ylabel("Close Price ($)")
plt.grid(True)
plt.legend()
plt.savefig(f"predict/{ticker}_forecast.png", bbox_inches="tight")
plt.close()

# Save results
last_close = df_ticker["Close"].iloc[-1]
predicted_change_pct = ((future_predictions[-1] - last_close) / last_close) * 100

results = {
    "ticker": ticker,
    "company_name": get_company_name(ticker),
    "last_close": float(last_close),
    "predicted_change_pct": float(predicted_change_pct),
    "future_predictions": future_predictions.flatten().tolist(),
}
with open(f"predict/{ticker}_results.json", "w") as f:
    json.dump(results, f, indent=4)

np.save(f"predict/{ticker}_forecast.npy", future_predictions)

logger.info(f"Results saved: predict/{ticker}_forecast.png and predict/{ticker}_results.json")