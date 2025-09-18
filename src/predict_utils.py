import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 
import sys
import time
import random
import json
import logging
from datetime import datetime, timedelta

import requests
import yfinance as yf
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s: %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

class MarketstackProvider:
    def __init__(self):
        self.api_key = os.getenv("MARKETSTACK_API_KEY")
        if not self.api_key:
            raise ValueError("MARKETSTACK_API_KEY environment variable not set")
        self.base_url = "http://api.marketstack.com/v1/"
        self.max_retries = 3
        self.min_delay = 1
        self.max_delay = 5

    def _make_request(self, endpoint, params=None):
        params = params or {}
        params.update({"access_key": self.api_key})

        for attempt in range(self.max_retries):
            try:
                response = requests.get(
                    f"{self.base_url}{endpoint}",
                    params=params,
                    timeout=10
                )
                response.raise_for_status()
                return response.json()
            except Exception as e:
                if attempt < self.max_retries - 1:
                    delay = random.uniform(self.min_delay, self.max_delay)
                    logger.warning(f"Attempt {attempt + 1} failed. Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                else:
                    raise Exception(f"API request failed: {str(e)}")

    def get_stock_data(self, ticker, start_date=None, end_date=None):
        params = {
            "symbols": ticker,
            "limit": 1000,
            "sort": "ASC"
        }
        if start_date:
            params["date_from"] = start_date
        if end_date:
            params["date_to"] = end_date

        data = self._make_request("eod", params)

        if not data.get("data"):
            return None

        df = pd.DataFrame(data["data"])
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        df = df.sort_index()

        df = df.rename(columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume"
        })

        # Log the available dates
        self._log_available_dates(df, source="Marketstack", ticker=ticker)

        return df[["Open", "High", "Low", "Close", "Volume"]]

    def resolve_ticker(self, yahoo_ticker: str):
        """
        Try to resolve a Yahoo Finance ticker to Marketstack's symbol.
        Example: AI.PA (Yahoo) -> AIR (Marketstack, Euronext Paris)
        """
        base_symbol = yahoo_ticker.split(".")[0]
        data = self._make_request("tickers", params={"search": base_symbol})
        candidates = data.get("data", [])

        if not candidates:
            logger.warning(f"No Marketstack match found for {yahoo_ticker}")
            return yahoo_ticker  # fallback

        if "." in yahoo_ticker:
            suffix = yahoo_ticker.split(".")[1]
            exchange_map = {
                "PA": "XPAR",  # Euronext Paris
                "DE": "XETR",  # Xetra (Germany)
                "L": "XLON",   # London Stock Exchange
                "MI": "XMIL",  # Milan
                "AS": "XAMS",  # Amsterdam
            }
            target_exchange = exchange_map.get(suffix)
            for c in candidates:
                if c.get("stock_exchange", {}).get("acronym") == target_exchange:
                    return c["symbol"]

        return candidates[0]["symbol"]

    def _log_available_dates(self, df: pd.DataFrame, source: str, ticker: str):
        """Helper to log the date range of a given DataFrame."""
        if df is None or df.empty:
            logger.info(f"{source}: No data available for {ticker}")
            return

        first_date = df.index.min().date()
        last_date = df.index.max().date()
        count = len(df)

        logger.info(
            f"{source}: Extracted {count} rows for {ticker} "
            f"from {first_date} to {last_date}"
        )

def get_company_name(ticker: str) -> str:
    try:
        info = yf.Ticker(ticker).info
        return info.get("longName", ticker)
    except Exception:
        return ticker


def fetch_stock_data(ticker):
    marketstack = MarketstackProvider()
    company_name = get_company_name(ticker)
    logger.info(f"Fetching data for {company_name} ({ticker})...")

    df_yf = yf.download(ticker, period="max", interval="1d", auto_adjust=False)
    if df_yf.empty:
        logger.warning(f"No Yahoo Finance data for {ticker}, trying Marketstack...")
        try:
            ms_symbol = marketstack.resolve_ticker(ticker)
            df_ms = marketstack.get_stock_data(ms_symbol)
            if df_ms is not None and not df_ms.empty:
                return df_ms
        except Exception as e:
            logger.error(f"Failed to fetch Marketstack data for {ticker}: {e}")
        return None


    df_yf = df_yf[["Open", "High", "Low", "Close", "Volume"]]
    last_date = df_yf.index[-1].date()
    today = datetime.today().date()

    # If Yahoo data is incomplete, supplement with Marketstack
    if last_date < today:
        try:
            ms_symbol = marketstack.resolve_ticker(ticker)
            df_ms = marketstack.get_stock_data(
                ms_symbol,
                start_date=(last_date + timedelta(days=1)).strftime("%Y-%m-%d"),
                end_date=today.strftime("%Y-%m-%d"),
            )
            if df_ms is not None and not df_ms.empty:
                df_yf = pd.concat([df_yf, df_ms])
                df_yf = df_yf[~df_yf.index.duplicated(keep="last")]
        except Exception as e:
            logger.warning(f"Marketstack fetch failed for {ticker}: {e}")

    return df_yf


def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:(i + look_back), 0])
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)