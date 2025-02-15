## AI-Powered Technical Analysis Dashboard (Gemini 2.0)

# Libraries - Importing necessary Python libraries
import streamlit as st  # Streamlit for creating the web application interface
import google.generativeai as genai
import yfinance as yf  # yfinance to download historical stock market data from Yahoo Finance
import pandas as pd  # pandas for data manipulation and analysis (DataFrames)
import plotly.graph_objects as go  # plotly for creating interactive charts, specifically candlestick charts
import tempfile  # tempfile for creating temporary files (used to save chart images)
import os  # os for interacting with the operating system (e.g., deleting temporary files)
import json  # json for working with JSON data (expected response format from Gemini)
from datetime import datetime, timedelta  # datetime and timedelta for date and time calculations

# Configure the API key - Use Streamlit secrets or environment variables for security
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# Select the Gemini model - using 'gemini-2.0-flash' as a general-purpose model
MODEL_NAME = 'gemini-2.0-flash'
gen_model = genai.GenerativeModel(MODEL_NAME)

# Set up Streamlit app
st.set_page_config(layout="wide")
st.title("AI-Powered Technical Stock Analysis Dashboard")
st.sidebar.header("Configuration")

# Input for multiple stock tickers (comma-separated)
tickers_input = st.sidebar.text_input("Enter Stock Tickers (comma-separated):", "AAPL,MSFT,GOOG")
tickers = [ticker.strip().upper() for ticker in tickers_input.split(",") if ticker.strip()]

# Set the date range: default is one year before today to today
end_date_default = datetime.today()
start_date_default = end_date_default - timedelta(days=365)
start_date = st.sidebar.date_input("Start Date", value=start_date_default)
end_date = st.sidebar.date_input("End Date", value=end_date_default)

# --- New: Time Frame Selection ---
time_frame_options = ["5min", "10min", "15min", "1hour", "4hours", "day", "week"]
selected_time_frame = st.sidebar.selectbox("Select Time Frame", time_frame_options)

# Map the user-friendly time frame to yfinance intervals.
# Note: yfinance supports only these intervals for intraday data: 1m,2m,5m,15m,30m,60m,90m.
# Therefore, we map unsupported options (10min and 4hours) to the closest supported values.
interval_map = {
    "5min": "5m",
    "15min": "15m",
    "1hour": "60m",
    "day": "1d",
    "week": "1wk"
}
selected_interval = interval_map[selected_time_frame]
# --- End Time Frame Selection ---

# Technical indicators selection (applied to every ticker)
st.sidebar.subheader("Technical Indicators")
indicators = st.sidebar.multiselect(
    "Select Indicators:",
    ["20-Day SMA", "20-Day EMA", "20-Day Bollinger Bands", "VWAP"],
    default=["20-Day SMA"]
)

# Button to fetch data for all tickers
if st.sidebar.button("Fetch Data"):
    stock_data = {}
    for ticker in tickers:
        # Download data for each ticker using yfinance with the selected interval.
        data = yf.download(ticker, start=start_date, end=end_date, interval=selected_interval)
        if not data.empty:
            stock_data[ticker] = data
        else:
            st.warning(f"No data found for {ticker}.")
    st.session_state["stock_data"] = stock_data
    st.success("Stock data loaded successfully for: " + ", ".join(stock_data.keys()))

# Ensure we have data to analyze
if "stock_data" in st.session_state and st.session_state["stock_data"]:
    # Define a function to build chart, call the Gemini API and return structured result
    def analyze_ticker(ticker, data):
        # Re-fetch data with the selected interval (if needed)
        data = yf.download(ticker, start=start_date, end=end_date, interval=selected_interval)

        # --- Debugging: Print the raw DataFrame ---
        st.write("### Raw Data from yfinance:")
        st.dataframe(data)
        # --- End Debugging ---
        if data.empty:
            st.warning(f"No data found for {ticker}.")
            return None, {"action": "Error", "justification": "No data fetched from yfinance"}
        # Build candlestick chart for the given ticker's data
        fig = go.Figure(data=[
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name="Candlestick"
            )
        ])

        # Add selected technical indicators
        def add_indicator(indicator):
            if indicator == "20-Day SMA":
                sma = data['Close'].rolling(window=20).mean()
                fig.add_trace(go.Scatter(x=data.index, y=sma, mode='lines', name='SMA (20)'))
            elif indicator == "20-Day EMA":
                ema = data['Close'].ewm(span=20).mean()
                fig.add_trace(go.Scatter(x=data.index, y=ema, mode='lines', name='EMA (20)'))
            elif indicator == "20-Day Bollinger Bands":
                sma = data['Close'].rolling(window=20).mean()
                std = data['Close'].rolling(window=20).std()
                bb_upper = sma + 2 * std
                bb_lower = sma - 2 * std
                fig.add_trace(go.Scatter(x=data.index, y=bb_upper, mode='lines', name='BB Upper'))
                fig.add_trace(go.Scatter(x=data.index, y=bb_lower, mode='lines', name='BB Lower'))
            elif indicator == "VWAP":
                data['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
                fig.add_trace(go.Scatter(x=data.index, y=data['VWAP'], mode='lines', name='VWAP'))

        for ind in indicators:
            add_indicator(ind)
        fig.update_layout(xaxis_rangeslider_visible=False)

        # Save chart as temporary PNG file and read image bytes
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
            fig.write_image(tmpfile.name)
            tmpfile_path = tmpfile.name
        with open(tmpfile_path, "rb") as f:
            image_bytes = f.read()
        os.remove(tmpfile_path)

        image_part = {
            "data": image_bytes,
            "mime_type": "image/png"
        }

        analysis_prompt = (
            f"You are a Stock Trader specializing in Technical Analysis at a top financial institution. "
            f"Analyze the stock chart for {ticker} based on its candlestick chart and the displayed technical indicators. "
            f"Provide a detailed justification of your analysis, explaining what patterns, signals, and trends you observe. "
            f"Then, based solely on the chart, provide a recommendation from the following options: "
            f"'Strong Buy', 'Buy', 'Weak Buy', 'Hold', 'Weak Sell', 'Sell', or 'Strong Sell'. "
            f"Return your output as a JSON object with two keys: 'action' and 'justification'."
        )

        contents = [
            {"role": "user", "parts": [analysis_prompt]},
            {"role": "user", "parts": [image_part]}
        ]

        response = gen_model.generate_content(contents=contents)

        try:
            result_text = response.text
            json_start_index = result_text.find('{')
            json_end_index = result_text.rfind('}') + 1
            if json_start_index != -1 and json_end_index > json_start_index:
                json_string = result_text[json_start_index:json_end_index]
                result = json.loads(json_string)
            else:
                raise ValueError("No valid JSON object found in the response")
        except json.JSONDecodeError as e:
            result = {"action": "Error", "justification": f"JSON Parsing error: {e}. Raw response text: {response.text}"}
        except ValueError as ve:
            result = {"action": "Error", "justification": f"Value Error: {ve}. Raw response text: {response.text}"}
        except Exception as e:
            result = {"action": "Error", "justification": f"General Error: {e}. Raw response text: {response.text}"}

        return fig, result

    # Create tabs: first tab for overall summary, subsequent tabs per ticker
    tab_names = ["Overall Summary"] + list(st.session_state["stock_data"].keys())
    tabs = st.tabs(tab_names)

    overall_results = []

    for i, ticker in enumerate(st.session_state["stock_data"]):
        data = st.session_state["stock_data"][ticker]
        fig, result = analyze_ticker(ticker, data)
        overall_results.append({"Stock": ticker, "Recommendation": result.get("action", "N/A")})
        with tabs[i + 1]:
            st.subheader(f"Analysis for {ticker}")
            st.plotly_chart(fig)
            st.write("**Detailed Justification:**")
            st.write(result.get("justification", "No justification provided."))

    with tabs[0]:
        st.subheader("Overall Structured Recommendations")
        df_summary = pd.DataFrame(overall_results)
        st.table(df_summary)
else:
    st.info("Please fetch stock data using the sidebar.")

# docker run -p 8501:8501 -e GOOGLE_API_KEY="your_actual_api_key" streamlit-stock-analysis
