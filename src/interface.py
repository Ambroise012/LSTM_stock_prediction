import json
import os
import sys
import subprocess
from PIL import Image
import streamlit as st
import pandas as pd
import datetime

st.set_page_config(page_title="Stock Prediction", layout="wide")
st.title("📈 Stock Price Prediction")

ticker = st.text_input("Enter stock ticker (e.g., AAPL, AI.PA, MSFT):")

if st.button("Predict"):
    if not ticker:
        st.warning("Please enter a ticker symbol.")
    else:
        with st.spinner("Running prediction model... This may take a while ⏳"):
            proc = subprocess.run([sys.executable, "-m", "src.predict_stock", ticker],
                                    capture_output=True, text=True)


        if proc.returncode != 0:
            st.error("❌ Prediction script failed")
            st.text(proc.stderr)
            st.stop()

        # Load JSON results
        results_file = f"predict/{ticker}_results.json"
        if os.path.exists(results_file):
            with open(results_file) as f:
                results = json.load(f)

            st.subheader(f"{results['company_name']} ({results['ticker']})")

            st.metric(
                label="Predicted Change (%)",
                value=f"{results['predicted_change_pct']:.2f}%",
                delta=f"Last Close: ${results['last_close']:.2f}"
            )

            # Show prediction chart
            img_path = f"predict/{ticker}_forecast.png"
            if os.path.exists(img_path):
                st.image(img_path, caption=f"Forecast for {ticker}", width=700)
            else:
                st.error("Prediction figure not found.")

            # Show table of predictions
            preds = pd.DataFrame({
                "Day": [+1, +2, +3, +4],
                "Predicted Price": results["future_predictions"]
            })
            st.dataframe(preds, use_container_width=True)
        else:
            st.error("Prediction results not found. Make sure predict_stock.py ran successfully.")
