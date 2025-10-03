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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout
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

# =========================
# Fetch data
# =========================
df_ticker = fetch_stock_data(ticker)

if isinstance(df_ticker, dict):
    df_ticker = df_ticker.get(ticker)

if df_ticker is None or df_ticker.empty:
    logger.error(f"No data available for {ticker}.")
    sys.exit(1)

# =========================
# Feature Engineering
# =========================

# Percentage change features
df_ticker["Return"] = df_ticker["Close"].pct_change()
df_ticker["Volume_Change"] = df_ticker["Volume"].pct_change()

# Moving Averages
df_ticker["SMA_5"] = df_ticker["Close"].rolling(window=5).mean()
df_ticker["SMA_20"] = df_ticker["Close"].rolling(window=20).mean()

# Exponential Moving Averages
df_ticker["EMA_12"] = df_ticker["Close"].ewm(span=12, adjust=False).mean()
df_ticker["EMA_26"] = df_ticker["Close"].ewm(span=26, adjust=False).mean()

# MACD
df_ticker["MACD"] = df_ticker["EMA_12"] - df_ticker["EMA_26"]

# RSI
delta = df_ticker["Close"].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df_ticker["RSI"] = 100 - (100 / (1 + rs))

# Volatility (rolling std of returns)
df_ticker["Volatility"] = df_ticker["Return"].rolling(window=10).std()

# Drop missing values
df_ticker.dropna(inplace=True)

df_ticker.replace([np.inf, -np.inf], np.nan, inplace=True)
df_ticker.dropna(inplace=True)
# =========================
# Preprocess: use percentage change instead of raw close
# =========================
feature_cols = ["Return", "Volume_Change", "SMA_5", "SMA_20",
                "EMA_12", "EMA_26", "MACD", "RSI", "Volatility"]

values = df_ticker[feature_cols].values
# Scale all features
scaler_all = MinMaxScaler()
scaled_values = scaler_all.fit_transform(df_ticker[feature_cols].values)

# Scale only the target ("Return") separately
scaler_target = MinMaxScaler()
scaled_target = scaler_target.fit_transform(df_ticker[["Return"]].values)


# =========================
# Create dataset
# =========================
def create_dataset(dataset_all, dataset_target, look_back=1, horizon=1):
    X, Y = [], []
    for i in range(len(dataset_all) - look_back - horizon + 1):
        X.append(dataset_all[i:(i + look_back), :])  # all features
        Y.append(dataset_target[i + look_back : i + look_back + horizon, 0])  # target only
    return np.array(X), np.array(Y)

X, Y = create_dataset(scaled_values, scaled_target, config.predict.look_back, config.predict.future_days)

# Train-test split
split_index = int(len(X) * 0.8)
trainX, testX = X[:split_index], X[split_index:]
trainY, testY = Y[:split_index], Y[split_index:]

trainX = trainX.reshape((trainX.shape[0], trainX.shape[1], len(feature_cols)))
testX = testX.reshape((testX.shape[0], testX.shape[1], len(feature_cols)))

# =========================
# Build LSTM model
# =========================
n_features = len(feature_cols)

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(config.predict.look_back, n_features)),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(config.predict.future_days)
])
model.compile(loss='huber', optimizer='adam')

# =========================
# Train model with EarlyStopping
# =========================
early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

history = model.fit(
    trainX, trainY,
    epochs=config.LSTM.epochs,
    batch_size=config.LSTM.batch_size,
    validation_data=(testX, testY),
    callbacks=[early_stop],
    verbose=1
)

# =========================
# Model evaluation
# =========================
test_pred = model.predict(testX)
test_pred_rescaled = scaler_target.inverse_transform(test_pred)
testY_rescaled = scaler_target.inverse_transform(testY)

mse = mean_squared_error(testY_rescaled, test_pred_rescaled)
mae = mean_absolute_error(testY_rescaled, test_pred_rescaled)
r2 = r2_score(testY_rescaled, test_pred_rescaled)

eval_metrics = {
    "mse": float(mse),
    "mae": float(mae),
    "r2": float(r2)
}

# =========================
# Future prediction
# =========================
last_sequence = scaled_values[-config.predict.look_back:].reshape(1, config.predict.look_back, n_features)

# Predict future % changes (scaled)
future_pct_changes = model.predict(last_sequence)

# Inverse transform with the target scaler, not the full feature scaler
future_pct_changes = scaler_target.inverse_transform(future_pct_changes).flatten()

# Convert % changes to future prices
last_close = df_ticker["Close"].iloc[-1]
future_prices = [last_close]
for pct in future_pct_changes:
    future_prices.append(future_prices[-1] * (1 + pct / 100))
future_prices = future_prices[1:]

# =========================
# Plot results
# =========================
os.makedirs("predict", exist_ok=True)
future_dates = [df_ticker.index[-1] + pd.Timedelta(days=i+1) for i in range(config.predict.future_days)]

plt.figure(figsize=(12, 6))
plt.plot(df_ticker["Close"].tail(30).index, df_ticker["Close"].tail(30).values, label="Last 30 Days Actual", linewidth=2)
plt.plot(future_dates, future_prices, label="Future Forecast (Reconstructed)", linestyle="--", color="red")
plt.title(f"{ticker} Stock - Last 30 Days + {config.predict.future_days} Days Forecast")
plt.xlabel("Date")
plt.ylabel("Close Price ($)")
plt.grid(True)
plt.legend()
plt.savefig(f"predict/{ticker}_forecast.png", bbox_inches="tight")
plt.close()

# =========================
# Save results
# =========================
predicted_change_pct = np.sum(future_pct_changes)  # cumulative % change

results = {
    "ticker": ticker,
    "company_name": get_company_name(ticker),
    "last_close": float(last_close),
    "predicted_change_pct": float(predicted_change_pct),
    "future_pct_changes": [float(x) for x in future_pct_changes],
    "future_prices": [float(x) for x in future_prices],
    "evaluation": {k: float(v) for k, v in eval_metrics.items()}
}
with open(f"predict/{ticker}_results.json", "w") as f:
    json.dump(results, f, indent=4)

np.save(f"predict/{ticker}_forecast.npy", future_prices)

logger.info(f"Results saved: predict/{ticker}_forecast.png and predict/{ticker}_results.json")
