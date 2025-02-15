import streamlit as st
import google.generativeai as genai
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import tempfile
import os
import json
from datetime import datetime, timedelta

# ... (API Key and Model Configuration - No changes needed) ...

# Set up Streamlit app
st.set_page_config(layout="wide")
st.title("AI-Powered Technical Stock Analysis Dashboard")
st.sidebar.header("Configuration")

# ... (Ticker Input - No changes needed) ...

# Set the date range: default is one year before today to today
end_date_default = datetime.today()
start_date_default = end_date_default - timedelta(days=365)
start_date = st.sidebar.date_input("Start Date", value=start_date_default)
end_date = st.sidebar.date_input("End Date", value=end_date_default)

# --- Time Frame Selection ---
time_frame_options = {
    "5min": {"interval": "5m", "days": 3, "max_points": 1000},
    "15min": {"interval": "15m", "days": 5, "max_points": 1000},
    "1hour": {"interval": "60m", "days": 30, "max_points": 2000},
    "day": {"interval": "1d", "days": 365, "max_points": None},
    "week": {"interval": "1wk", "days": 365, "max_points": None}
}

def show_timeframe_warning(selected_timeframe):
    if selected_timeframe in ["5min", "15min"]:
        st.sidebar.warning(f"""
        ⚠️ Important Note for {selected_timeframe} timeframe:
        - Limited to last {time_frame_options[selected_timeframe]['days']} days only
        - Data might be delayed or limited by Yahoo Finance
        - May not work reliably outside market hours
        """)

selected_time_frame = st.sidebar.selectbox(
    "Select Time Frame",
    list(time_frame_options.keys())
)
show_timeframe_warning(selected_time_frame) # Call the warning function

# ... (Technical Indicators Selection - No changes needed) ...

if st.sidebar.button("Fetch Data"):
    stock_data = {}
    for ticker in tickers:
        try:
            # Calculate appropriate start date based on selected timeframe
            max_days = time_frame_options[selected_time_frame]["days"]
            interval = time_frame_options[selected_time_frame]["interval"]

            # Adjust start date based on timeframe
            adjusted_start_date = datetime.today() - timedelta(days=max_days)
            if start_date > adjusted_start_date:
                adjusted_start_date = start_date

            # **DEBUG PRINTING - Important for troubleshooting**
            st.write(f"Fetching {ticker} data:")
            st.write(f"  Interval: {interval}")
            st.write(f"  Start Date: {adjusted_start_date.strftime('%Y-%m-%d %H:%M:%S')}") # Format for clarity
            st.write(f"  End Date: {end_date.strftime('%Y-%m-%d %H:%M:%S')}")

            data = yf.download(
                ticker,
                start=adjusted_start_date,
                end=end_date,
                interval=interval,
                prepost=True
            )

            if data is not None and not data.empty: # **Robust check for None and empty**
                stock_data[ticker] = data
                st.success(f"Successfully fetched {len(data)} rows for {ticker} ({interval} interval)")
                # **DEBUG PRINTING - Check data shape**
                st.write(f"  Data shape: {data.shape}")
            else:
                st.warning(f"No data found for {ticker} with {interval} interval.  Data might be limited for this timeframe or date range.")

        except Exception as e:
            st.error(f"Error fetching {ticker}: {str(e)}")
            st.error(f"Error details: {e}") # Print full error for debugging

    if stock_data:
        st.session_state["stock_data"] = stock_data
        st.success("Stock data loaded successfully for: " + ", ".join(stock_data.keys()))
    else:
        st.error("No data was loaded for any ticker")


if "stock_data" in st.session_state and st.session_state["stock_data"]:

    def downsample_data(data, max_points):
        if max_points and len(data) > max_points:
            sample_interval = len(data) // max_points
            return data.iloc[::sample_interval]
        return data

    def analyze_ticker(ticker, data): # **Removed redundant data fetching here**
        try:
            timeframe_info = time_frame_options[selected_time_frame]
            max_points = timeframe_info["max_points"]
            interval = timeframe_info["interval"]

            if data.empty: # **Check if data is already empty (from outer scope)**
                st.warning(f"No data available for {ticker} (already checked in data fetching).")
                empty_fig = go.Figure()
                empty_fig.add_annotation(
                    text="No data available from yfinance",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )
                return empty_fig, {"action": "Error", "justification": "No data fetched from yfinance (already checked)."}


            data = data.dropna()
            if max_points:
                original_length = len(data)
                data = downsample_data(data, max_points)
                if len(data) < original_length:
                    st.info(f"Data downsampled from {original_length} to {len(data)} points for performance.")

            # Build candlestick chart
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

            # ... (Add Indicators - No changes needed) ...

            fig.update_layout(
                title=f"{ticker} Stock Price",
                yaxis_title="Price",
                xaxis_title="Date",
                xaxis_rangeslider_visible=False
            )

            for ind in indicators:
                add_indicator(ind)

            # ... (Save Chart as Image and Gemini Analysis - No changes needed for now) ...
            # ... (Keep Gemini part for later testing, but focus on chart display first) ...
            # ... (For now, let's just return a placeholder result to avoid Gemini errors while debugging charting) ...
            result = {"action": "Hold", "justification": "Placeholder analysis - Charting debug in progress."} # Placeholder

            return fig, result

        except Exception as e:
            st.error(f"Error analyzing {ticker}: {str(e)}")
            st.error(f"Analysis error details: {e}") # Print full error for analysis errors
            empty_fig = go.Figure()
            empty_fig.add_annotation(
                text=f"Error during analysis: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return empty_fig, {"action": "Error", "justification": f"Analysis error: {str(e)}"}

    # Create tabs
    tab_names = ["Overall Summary"] + list(st.session_state["stock_data"].keys())
    tabs = st.tabs(tab_names)

    overall_results = []

    for i, ticker in enumerate(st.session_state["stock_data"]):
        data = st.session_state["stock_data"][ticker]
        fig, result = analyze_ticker(ticker, data)
        overall_results.append({"Stock": ticker, "Recommendation": result.get("action", "N/A")})
        with tabs[i + 1]:
            st.subheader(f"Analysis for {ticker}")
            st.plotly_chart(fig) # **Display the chart**
            st.write("**Detailed Justification (Placeholder):**") # Indicate placeholder
            st.write(result.get("justification", "No justification provided."))

    with tabs[0]:
        st.subheader("Overall Structured Recommendations (Placeholder)") # Indicate placeholder
        df_summary = pd.DataFrame(overall_results)
        st.table(df_summary)
else:
    st.info("Please fetch stock data using the sidebar.")

# docker run -p 8501:8501 -e GOOGLE_API_KEY="your_actual_api_key" streamlit-stock-analysis