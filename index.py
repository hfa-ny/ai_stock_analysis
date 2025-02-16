## Gemini AI-Powered Stocks Technical Analysis 

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

# Update the page config and title section
st.set_page_config(layout="wide", initial_sidebar_state="expanded")

# Add custom CSS to fix header spacing
st.markdown("""
    <style>
        .block-container {
            padding-top: 3rem !important;
            padding-bottom: 1rem !important;
        }
        header {
            margin-bottom: 2rem !important;
        }
        .main > div {
            padding-left: 2rem !important;
            padding-right: 2rem !important;
        }
        .financial-metrics {
            font-size: 0.8rem !important;
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            justify-content: space-between;
        }
        .metric-item {
            flex: 1 1 auto;
            min-width: 120px;
            padding: 0.3rem;
        }
        .metric-label {
            color: #666;
            font-size: 0.7rem !important;
        }
        .metric-value {
            font-weight: bold;
            font-size: 0.8rem !important;
        }
        #MainMenu {visibility: visible;}
        header {visibility: visible;}
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

st.title("HFA AI-Powered Technical Stock Analysis")
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
# Update the time frame options with only supported intervals
time_frame_options = {
    "5min": {"interval": "5m", "days": 10, "max_points": None},
    "15min": {"interval": "15m", "days": 14, "max_points": None},
    "1hour": {"interval": "60m", "days": 45, "max_points": None},
    "day": {"interval": "1d", "days": 730, "max_points": None},
    "week": {"interval": "1wk", "days": 730, "max_points": None}
}

# Add warning for intraday data limitations
def show_timeframe_warning(selected_timeframe):
    if selected_timeframe in ["5min", "15min"]:
        st.sidebar.warning(f"""
        ⚠️ Important Note for {selected_timeframe} timeframe:
        - Limited to last {time_frame_options[selected_timeframe]['days']} days only
        - Data might be delayed or limited
        - May not work outside market hours
        """)

# Update the sidebar selection
selected_time_frame = st.sidebar.selectbox(
    "Select Time Frame",
    list(time_frame_options.keys())
)
show_timeframe_warning(selected_time_frame)

# Technical indicators selection (applied to every ticker)
st.sidebar.subheader("Technical Indicators")
# Update the indicators selection list
indicators = st.sidebar.multiselect(
    "Select Indicators:",
    ["20-Day SMA", "20-Day EMA", "20-Day Bollinger Bands", "VWAP", "RSI"],  # Added RSI
    default=["20-Day SMA"]
)

# Modify the data fetching part
if st.sidebar.button("Fetch Data"):
    stock_data = {}
    for ticker in tickers:
        try:
            # Calculate appropriate start date based on selected timeframe
            max_days = time_frame_options[selected_time_frame]["days"]
            interval = time_frame_options[selected_time_frame]["interval"]

            # Adjust start date based on timeframe
            adjusted_start_date = datetime.today() - timedelta(days=max_days)
            if start_date > adjusted_start_date.date(): # Corrected line: compare dates
                adjusted_start_date = start_date

            # Debug information - Removed for cleaner UI, keep for debugging if needed
            # st.write(f"Fetching {ticker} data:")
            # st.write(f"Interval: {interval}")
            # st.write(f"Start Date: {adjusted_start_date.strftime('%Y-%m-%d %H:%M:%S')}")
            # st.write(f"End Date: {end_date.strftime('%Y-%m-%d %H:%M:%S')}")

            # Download data with proper parameters
            data = yf.download(
                ticker,
                start=adjusted_start_date,
                end=end_date,
                interval=interval,
                prepost=False  # Changed to False to exclude pre/post market hours
            )

            if data is not None and not data.empty:
                stock_data[ticker] = data
               # st.success(f"Successfully fetched {len(data)} rows for {ticker} ({interval} interval)")
                # **DEBUG PRINTING - Check data shape - Removed for cleaner UI, keep for debugging if needed
                # st.write(f"  Data shape: {data.shape}")
            else:
                st.warning(f"No data found for {ticker} with {interval} interval.  Data might be limited for this timeframe or date range.")

        except Exception as e:
            st.error(f"Error fetching {ticker}: {str(e)}")
            st.error(f"Error details: {e}")

    if stock_data:
        st.session_state["stock_data"] = stock_data
        st.success("Stock data loaded successfully for: " + ", ".join(stock_data.keys()))
    else:
        st.error("No data was loaded for any ticker")

# Ensure we have data to analyze
if "stock_data" in st.session_state and st.session_state["stock_data"]:
    # Add this function before the analyze_ticker function
    def downsample_data(data, max_points):
        """Downsample data if it exceeds max_points"""
        if max_points and len(data) > max_points:
            # Calculate sampling interval
            sample_interval = len(data) // max_points
            return data.iloc[::sample_interval]
        return data

    # Add this function before analyze_ticker
    def get_financial_metrics(ticker_symbol):
        """Get key financial metrics for a stock"""
        try:
            ticker = yf.Ticker(ticker_symbol)
            info = ticker.info
            
            # Key financial metrics to display
            metrics = {
                "Market Cap": info.get('marketCap', 'N/A'),
                "P/E Ratio": info.get('trailingPE', 'N/A'),
                "Forward P/E": info.get('forwardPE', 'N/A'),
                "PEG Ratio": info.get('pegRatio', 'N/A'),
                "Price/Book": info.get('priceToBook', 'N/A'),
                "Dividend Yield": info.get('dividendYield', 'N/A'),
                "52 Week High": info.get('fiftyTwoWeekHigh', 'N/A'),
                "52 Week Low": info.get('fiftyTwoWeekLow', 'N/A'),
                "50 Day MA": info.get('fiftyDayAverage', 'N/A'),
                "200 Day MA": info.get('twoHundredDayAverage', 'N/A'),
            }
            
            # Format numerical values
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    if key == "Market Cap":
                        metrics[key] = f"${value:,.0f}"
                    elif key == "Dividend Yield" and value is not None:
                        metrics[key] = f"{value:.2%}"
                    else:
                        metrics[key] = f"{value:.2f}"
                        
            return metrics
        except Exception as e:
            st.warning(f"Error fetching financial data for {ticker_symbol}: {str(e)}")
            return {}

    # Define a function to build chart, call the Gemini API and return structured result
    def analyze_ticker(ticker, data, indicators): # **Pass 'indicators' as argument**
        try:
            # Get the max points limit and interval for the selected timeframe
            timeframe_info = time_frame_options[selected_time_frame]
            max_points = timeframe_info["max_points"]
            interval = timeframe_info["interval"]

            if data.empty:
                st.warning(f"No data available for {ticker} with {interval} interval")
                empty_fig = go.Figure()
                empty_fig.add_annotation(
                    text="No data available for selected timeframe",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )
                return empty_fig, {"action": "Error", "justification": "No data available for selected timeframe"}

            # Clean and downsample the data if needed
            data = data.dropna()
            if max_points:
                original_length = len(data)
                data = downsample_data(data, max_points)
                if len(data) < original_length:
                    st.info(f"Data has been downsampled from {original_length} to {len(data)} points for better performance")

            # Debug information - Removed for cleaner UI, keep for debugging if needed
            # st.write(f"### Data Information for {ticker}:")
            # st.write(f"Total rows after processing: {len(data)}")

            # Check if data is empty or invalid
            if data.empty:
                st.warning(f"No data found for {ticker}.")
                # Return empty figure and error message
                empty_fig = go.Figure()
                empty_fig.add_annotation(
                    text="No data available",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )
                return empty_fig, {"action": "Error", "justification": "No data fetched from yfinance"}

            # Add this before creating the chart
            if selected_time_frame in ["5min", "15min", "1hour"]:
                # Remove data points with no volume
                data = data[data['Volume'] > 0]
                
                # Remove outliers
                Q1 = data['Close'].quantile(0.25)
                Q3 = data['Close'].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                # Filter out extreme values
                data = data[
                    (data['Close'] >= lower_bound) & 
                    (data['Close'] <= upper_bound)
                ]

                # Forward fill small gaps (only for remaining data)
                data = data.fillna(method='ffill', limit=3)

                # Remove any remaining NaN values
                data = data.dropna()

            # Build candlestick chart with logarithmic scale
            fig = go.Figure(data=[
                go.Candlestick(
                    x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name="Price",
                    increasing_line_color='#26A69A',
                    decreasing_line_color='#EF5350'
                )
            ])

            # Update layout with logarithmic scale for small timeframes
            fig.update_layout(
                title=f"{ticker} Stock Price",
                yaxis_title="Price (log scale)",
                xaxis_title="Date",
                yaxis_type="log" if selected_time_frame in ["5min", "15min", "1hour"] else "linear",
                xaxis_rangeslider_visible=False,
                template="plotly_white",
                height=600,
                yaxis=dict(
                    gridcolor="rgba(128, 128, 128, 0.2)",
                    zerolinecolor="rgba(128, 128, 128, 0.2)",
                    tickformat=".2f",  # Show 2 decimal places
                    showgrid=True,
                    showline=True
                ),
                xaxis=dict(
                    gridcolor="rgba(128, 128, 128, 0.2)",
                    rangeslider=dict(visible=False),
                    showgrid=True,
                    showline=True,
                    type='date'
                ),
                plot_bgcolor='white',
                paper_bgcolor='white',
                margin=dict(l=50, r=50, t=50, b=50),
                showlegend=True
            )

            # For intraday data, add specific configurations
            if selected_time_frame in ["5min", "15min", "1hour"]:
                fig.update_xaxes(
                    rangebreaks=[
                        dict(bounds=["sat", "mon"]),  # Hide weekends
                        dict(bounds=[20, 4], pattern="hour"),  # Hide non-trading hours
                    ]
                )

            # Add selected technical indicators
            def add_indicator(indicator):
                try:
                    if indicator == "20-Day SMA":
                        sma = data['Close'].rolling(window=20).mean()
                        fig.add_trace(go.Scatter(x=data.index, y=sma, mode='lines', name='SMA (20)'))
                    elif indicator == "20-Day EMA":
                        ema = data['Close'].ewm(span=20, adjust=False).mean()
                        fig.add_trace(go.Scatter(x=data.index, y=ema, mode='lines', name='EMA (20)'))
                    elif indicator == "20-Day Bollinger Bands":
                        sma = data['Close'].rolling(window=20).mean()
                        std = data['Close'].rolling(window=20).std()
                        bb_upper = sma + 2 * std
                        bb_lower = sma - 2 * std
                        fig.add_trace(go.Scatter(x=data.index, y=bb_upper, mode='lines', name='BB Upper'))
                        fig.add_trace(go.Scatter(x=data.index, y=bb_lower, mode='lines', name='BB Lower'))
                    elif indicator == "VWAP":
                        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
                        vwap = (typical_price * data['Volume']).cumsum() / data['Volume'].cumsum()
                        fig.add_trace(go.Scatter(x=data.index, y=vwap, mode='lines', name='VWAP'))
                    elif indicator == "RSI":
                        # Calculate RSI
                        delta = data['Close'].diff()
                        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                        rs = gain / loss
                        rsi = 100 - (100 / (1 + rs))
                        
                        # Add RSI subplot
                        fig.add_trace(
                            go.Scatter(
                                x=data.index,
                                y=rsi,
                                name='RSI',
                                yaxis="y2"
                            )
                        )
                        
                        # Update layout to include RSI subplot
                        fig.update_layout(
                            yaxis2=dict(
                                title="RSI",
                                overlaying="y",
                                side="right",
                                range=[0, 100],
                                showgrid=False
                            ),
                            # Add RSI level lines
                            shapes=[
                                # Overbought line (70)
                                dict(
                                    type="line",
                                    xref="paper",
                                    x0=0,
                                    x1=1,
                                    y0=70,
                                    y1=70,
                                    yref="y2",
                                    line=dict(
                                        color="red",
                                        width=1,
                                        dash="dash"
                                    )
                                ),
                                # Oversold line (30)
                                dict(
                                    type="line",
                                    xref="paper",
                                    x0=0,
                                    x1=1,
                                    y0=30,
                                    y1=30,
                                    yref="y2",
                                    line=dict(
                                        color="green",
                                        width=1,
                                        dash="dash"
                                    )
                                )
                            ]
                        )
                        
                except Exception as e:
                    st.warning(f"Error adding indicator {indicator}: {str(e)}")

            # Update layout
            fig.update_layout(
                title=f"{ticker} Stock Price",
                yaxis_title="Price",
                xaxis_title="Date",
                xaxis_rangeslider_visible=False,
                xaxis_rangeselector=dict(  # Add range selector for better navigation
                    buttons=list([
                        dict(count=1, label="1D", step="day", stepmode="backward"),
                        dict(count=5, label="5D", step="day", stepmode="backward"),
                        dict(count=1, label="1M", step="month", stepmode="backward"),
                        dict(count=3, label="3M", step="month", stepmode="backward"),
                        dict(count=6, label="6M", step="month", stepmode="backward"),
                        dict(count=1, label="1Y", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                )
            )

            # Add indicators
            for ind in indicators: # **'indicators' is now passed as argument**
                add_indicator(ind)

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

        except Exception as e:
            st.error(f"Error analyzing {ticker}: {str(e)}")
            st.error(f"Analysis error details: {e}")
            empty_fig = go.Figure()
            empty_fig.add_annotation(
                text=f"Error during analysis: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return empty_fig, {"action": "Error", "justification": f"Analysis error: {str(e)}"}

    # First, collect all results
    overall_results = []
    fig_results = {}

    # First loop to collect all analyses
    for ticker in st.session_state["stock_data"]:
        data = st.session_state["stock_data"][ticker]
        fig, result = analyze_ticker(ticker, data, indicators)
        overall_results.append({"Stock": ticker, "Recommendation": result.get("action", "N/A")})
        fig_results[ticker] = (fig, result)

    # Now create tabs and display results
    tab_names = ["Overall Summary"] + list(st.session_state["stock_data"].keys())
    tabs = st.tabs(tab_names)

    # Display Overall Summary tab
    with tabs[0]:
        st.subheader("Overall Structured Recommendations")
        df_summary = pd.DataFrame(overall_results)
        st.table(df_summary)
        
        # Display all stocks' data in Overall Summary
        st.subheader("Raw Data for All Stocks")
        for stock, stock_data in st.session_state["stock_data"].items():
            with st.expander(f"Raw Data for {stock} ({selected_time_frame})"):
                st.dataframe(stock_data)

    # Display individual stock tabs
    for i, ticker in enumerate(st.session_state["stock_data"]):
        with tabs[i + 1]:
            fig, result = fig_results[ticker]
            
            # First row: Title and Recommendation
            col1, col2 = st.columns([3, 2])
            with col1:
                st.subheader(f"Analysis for {ticker}")
            with col2:
                recommendation = result.get("action", "N/A")
                # Color-code the recommendation
                color = {
                    "Strong Buy": "green",
                    "Buy": "lightgreen",
                    "Weak Buy": "palegreen",
                    "Hold": "yellow",
                    "Weak Sell": "pink",
                    "Sell": "lightcoral",
                    "Strong Sell": "red"
                }.get(recommendation, "gray")
                
                st.markdown(f"""
                    <div style='
                        background-color: {color};
                        padding: 10px;
                        border-radius: 5px;
                        text-align: center;
                        margin-top: 20px;
                    '>
                        <strong>Recommendation: {recommendation}</strong>
                    </div>
                    """, unsafe_allow_html=True)

            # Second row: Financial Metrics
            metrics = get_financial_metrics(ticker)
            if metrics:
                st.markdown("""
                    <style>
                        .metrics-grid {
                            display: grid;
                            grid-auto-flow: column;
                            grid-auto-columns: minmax(120px, 1fr);
                            gap: 8px;
                            overflow-x: auto;
                            padding: 8px 0;
                            margin: 10px 0;
                        }
                        .metric-box {
                            background-color: #f8f9fa;
                            border-radius: 4px;
                            padding: 8px;
                            text-align: center;
                        }
                        .metric-label {
                            color: #666;
                            font-size: 0.7rem;
                            margin-bottom: 4px;
                            white-space: nowrap;
                        }
                        .metric-value {
                            font-weight: bold;
                            font-size: 0.8rem;
                            white-space: nowrap;
                        }
                    </style>
                    <div class="metrics-grid">
                """, unsafe_allow_html=True)
                
                # Add metrics in a single row
                metric_html = "".join([f"""
                    <div class="metric-box">
                        <div class="metric-label">{key}</div>
                        <div class="metric-value">{value}</div>
                    </div>
                """ for key, value in metrics.items()])
                
                st.markdown(f'{metric_html}</div>', unsafe_allow_html=True)
            
            # Third row: Chart and Analysis
            st.plotly_chart(fig, key=f"plotly_chart_{ticker}")
            st.write("**Detailed Justification:**")
            st.write(result.get("justification", "No justification provided."))
            
            # Show only current stock's data in its tab
            with st.expander(f"Raw Data for {ticker} ({selected_time_frame})"):
                st.dataframe(data)

else:
    st.info("Please fetch stock data using the sidebar.")

# docker run -p 8501:8501 -e GOOGLE_API_KEY="your_actual_api_key" streamlit-stock-analysis