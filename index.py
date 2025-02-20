# Configure Streamlit page - must be the first Streamlit command
import streamlit as st
st.set_page_config(layout="wide", initial_sidebar_state="expanded")

# Rest of the imports
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import tempfile
import os
import json
from datetime import datetime, timedelta
import time
from concurrent.futures import ThreadPoolExecutor
import pandas_market_calendars as mcal
import requests.exceptions
import requests
import google.generativeai as genai

# Add this helper function at the top of the file, after imports
def safe_format_price(value):
    """Safely format price values from Series"""
    try:
        if pd.isna(value):
            return "N/A"
        return f"${float(value):.2f}"
    except (ValueError, TypeError):
        return "N/A"

# After imports, add debug logging
def log_debug(title, data):
    """Debug logging function (currently disabled)"""
    # Commented out debug statements
    # st.sidebar.markdown(f"**Debug: {title}**")
    # if isinstance(data, pd.DataFrame):
    #     st.sidebar.write(f"Shape: {data.shape}")
    #     st.sidebar.write("First few rows:")
    #     st.sidebar.write(data.head())
    # else:
    #     st.sidebar.write(data)
    pass

# Add this function after the imports
def test_yfinance_connection():
    """Basic test of yfinance functionality"""
    try:
        # Test with a simple, direct yfinance call
        ticker = yf.Ticker("AAPL")
        end = datetime.now()
        start = end - timedelta(days=5)
        
        # Try to get just daily data for 5 days
        hist = ticker.history(
            start=start,
            end=end,
            interval='1d',
            auto_adjust=False,
            repair=True
        )
        
        if (hist.empty):
            return False, "No data returned from yfinance"
        
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in hist.columns for col in required_cols):
            return False, f"Missing columns. Found: {list(hist.columns)}"
            
        return True, hist
    except Exception as e:
        return False, f"Error: {str(e)}"

# Add at the top with other imports
def test_yahoo_connection():
    """Test direct connection to Yahoo Finance"""
    url = "https://query1.finance.yahoo.com/v8/finance/chart/AAPL?range=1d&interval=1d"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return True, "Connection successful"
        return False, f"Connection failed with status code: {response.status_code}"
    except Exception as e:
        return False, f"Connection error: {str(e)}"

# Configure the API key - Use Streamlit secrets or environment variables for security
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# Select the Gemini model - using 'gemini-2.0-flash' as a general-purpose model
MODEL_NAME = 'gemini-2.0-flash'
gen_model = genai.GenerativeModel(MODEL_NAME)

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
tickers_input = st.sidebar.text_input("Enter Stock Tickers (comma-separated):", "AAPL,^GSPC")
tickers = [ticker.strip().upper() for ticker in tickers_input.split(",") if ticker.strip()]

# Set the date range: default is one year before today to today
end_date_default = datetime.today()
start_date_default = end_date_default - timedelta(days=365)
start_date = st.sidebar.date_input("Start Date", value=start_date_default)
end_date = st.sidebar.date_input("End Date", value=end_date_default)

# --- New: Time Frame Selection ---
# Update the time frame options with more conservative settings
time_frame_options = {
    "1day": {"interval": "1d", "days": 365, "max_points": None},
    "5day": {"interval": "5d", "days": 365, "max_points": None},
    "1week": {"interval": "1wk", "days": 730, "max_points": None},
    "1month": {"interval": "1mo", "days": 1825, "max_points": None},
    "3month": {"interval": "3mo", "days": 1825, "max_points": None}
}

# Add warning for intraday data limitations
def show_timeframe_warning(selected_timeframe):
    if (selected_timeframe in ["5min", "15min"]):
        st.sidebar.warning(f"""
        ⚠️ Important Note for {selected_timeframe} timeframe:
        - Limited to last {time_frame_options[selected_timeframe]['days']} days only
        - Data might be delayed or limited
        - May not work outside market hours
        """)

# Update the sidebar selection with new timeframes
selected_time_frame = st.sidebar.selectbox(
    "Select Time Frame",
    list(time_frame_options.keys()),
    index=list(time_frame_options.keys()).index("1day") 
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

def fetch_yahoo_finance_data(ticker, start_date, end_date):
    """Backup method to fetch data directly from Yahoo Finance API"""
    try:
        # Convert dates to UNIX timestamps
        start_timestamp = int(pd.Timestamp(start_date).timestamp())
        end_timestamp = int(pd.Timestamp(end_date).timestamp())
        
        # Construct the URL
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?period1={start_timestamp}&period2={end_timestamp}&interval=1d"
        
        # Add headers to mimic a browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Make the request
        response = requests.get(url, headers=headers)
        data = response.json()
        
        # Extract price data
        timestamps = data['chart']['result'][0]['timestamp']
        quote = data['chart']['result'][0]['indicators']['quote'][0]
        
        # Create DataFrame
        df = pd.DataFrame({
            'Open': quote['open'],
            'High': quote['high'],
            'Low': quote['low'],
            'Close': quote['close'],
            'Volume': quote['volume']
        }, index=pd.to_datetime(timestamps, unit='s'))
        
        return df
    except Exception as e:
        st.sidebar.error(f"Backup data fetch failed: {str(e)}")
        return pd.DataFrame()

# Update the fetch_with_retry function
def fetch_with_retry(ticker, start_date, end_date, interval, max_retries=3, delay=2):
    """Fetch stock data with retry logic"""
    for attempt in range(max_retries):
        try:
            # Try yfinance download first
            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                interval=interval,
                prepost=False,
                progress=False,
                repair=True,
                ignore_tz=True,
                timeout=20
            )
            
            if not data.empty:
                return data
                
            # If first method fails, try Ticker object
            stock = yf.Ticker(ticker)
            data = stock.history(
                start=start_date,
                end=end_date,
                interval=interval,
                repair=True
            )
            
            if not data.empty:
                return data
                
            # If both methods fail, try the backup method
            backup_data = fetch_yahoo_finance_data(ticker, start_date, end_date)
            if not backup_data.empty:
                return backup_data
            
            if attempt < max_retries - 1:
                # st.sidebar.warning(f"Attempt {attempt + 1} failed for {ticker}, retrying...")  # Commented out debug message
                time.sleep(delay * (attempt + 1))
            
        except Exception as e:
            if attempt == max_retries - 1:
                # st.sidebar.error(f"Failed to fetch {ticker} after all attempts: {str(e)}")  # Commented out debug message
                # Try one last time with minimal parameters
                try:
                    data = yf.download(ticker, period="1mo", progress=False)
                    if not data.empty:
                        return data
                except:
                    pass
                raise Exception(f"All fetch attempts failed for {ticker}")
            # st.sidebar.warning(f"Attempt {attempt + 1} failed: {str(e)}, retrying...")  # Commented out debug message
            time.sleep(delay * (attempt + 1))
    
    return pd.DataFrame()

# Add at the top after imports and before page config
def initialize_session_state():
    """Initialize or reset session state variables"""
    if 'debug_mode' not in st.session_state:
        st.session_state.debug_mode = False
    if 'last_fetch_time' not in st.session_state:
        st.session_state.last_fetch_time = None
    if 'fetch_attempts' not in st.session_state:
        st.session_state.fetch_attempts = {}

# Comment out debug options in sidebar
# st.sidebar.header("Debug Options")
# st.session_state.debug_mode = st.sidebar.checkbox("Enable Debug Mode", value=False)

# if st.session_state.debug_mode:
#     st.sidebar.markdown("### Session State Debug")
#     st.sidebar.write("Last fetch time:", st.session_state.get('last_fetch_time'))
#     st.sidebar.write("Fetch attempts:", st.session_state.get('fetch_attempts'))
#     if 'stock_data' in st.session_state:
#         for ticker, data in st.session_state['stock_data'].items():
#             st.sidebar.markdown(f"**{ticker} Data Shape:** {data.shape}")

# In the "Fetch Data" button click handler, add logging
if st.sidebar.button("Fetch Data"):
    stock_data = {}
    failed_tickers = []
    
    for ticker in tickers:
        try:
            # Calculate appropriate start date based on selected timeframe
            max_days = time_frame_options[selected_time_frame]["days"]
            interval = time_frame_options[selected_time_frame]["interval"]

            # Commented out debug logging
            # log_debug("Fetch Parameters", {
            #     "ticker": ticker,
            #     "interval": interval,
            #     "max_days": max_days,
            #     "start_date": start_date,
            #     "end_date": end_date
            # })

            # Adjust start date based on timeframe
            adjusted_start_date = datetime.today() - timedelta(days=max_days)
            if start_date > adjusted_start_date.date():
                adjusted_start_date = datetime.combine(start_date, datetime.min.time())

            # Use the retry function
            data = fetch_with_retry(
                ticker,
                start_date=adjusted_start_date,
                end_date=end_date,
                interval=interval
            )
            
            # log_debug(f"Fetched Data for {ticker}", data)  # Commented out debug logging

            if not data.empty:
                stock_data[ticker] = data
                st.success(f"Successfully fetched {len(data)} rows for {ticker}")
            else:
                failed_tickers.append(ticker)
                st.warning(f"No data found for {ticker}. The symbol might be incorrect or delisted.")

        except Exception as e:
            failed_tickers.append(ticker)
            st.error(f"Error fetching {ticker}: {str(e)}")

    if stock_data:
        st.session_state["stock_data"] = stock_data
        # log_debug("Session State Data", stock_data)  # Commented out debug logging
        st.success(f"Stock data loaded successfully for {len(stock_data)} ticker(s)")
        if failed_tickers:
            st.warning(f"Failed to fetch data for: {', '.join(failed_tickers)}")
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
            # Skip detailed metrics for index symbols
            if ticker_symbol.startswith('^'):
                return {
                    "Type": "Market Index",
                    "Note": "Detailed metrics not available for indices"
                }
            
            # Add delay between API calls
            time.sleep(2)  # Wait 2 seconds between requests
            
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
            # st.warning(f"Error fetching financial data for {ticker_symbol}: {str(e)}")  # Commented out debug message
            return {}

    # Define a function to build chart, call the Gemini API and return structured result
    def analyze_ticker(ticker, data, indicators):
        try:
            # Initial data validation
            if data.isna().any().any():
                data = data.dropna()
            
            if data.empty:
                raise ValueError(f"No data available for {ticker}")
            
            # Ensure numeric data types
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in numeric_columns:
                if col in data.columns:
                    if not pd.api.types.is_numeric_dtype(data[col]):
                        data[col] = pd.to_numeric(data[col].astype(str), errors='coerce')
            
            # Drop any NaN values after conversion
            data = data.dropna(subset=numeric_columns)
                    
            # Convert index to datetime if not already
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index)
            
            # Sort data by date and remove duplicates
            data = data.sort_index().loc[~data.index.duplicated(keep='first')]
            
            if data.empty:
                raise ValueError(f"No valid numeric data available for {ticker} after cleaning")

            # Create the candlestick chart
            candlestick = go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name="Price",
                increasing_line_color='#26A69A',
                decreasing_line_color='#EF5350'
            )
            
            fig = go.Figure(data=[candlestick])
            
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
                        
                        fig.add_trace(
                            go.Scatter(x=data.index, y=rsi, name='RSI', yaxis="y2")
                        )
                        
                        fig.update_layout(
                            yaxis2=dict(
                                title="RSI",
                                overlaying="y",
                                side="right",
                                range=[0, 100],
                                showgrid=False
                            )
                        )
                except Exception as e:
                    pass  # Commented out: st.warning(f"Error adding indicator {indicator}: {str(e)}")

            # Add indicators
            for ind in indicators:
                add_indicator(ind)

            # Update layout
            fig.update_layout(
                title=f"{ticker} Stock Price",
                yaxis_title="Price",
                xaxis_title="Date",
                xaxis_rangeslider_visible=False,
                template="plotly_white",
                height=600
            )

            # Rest of analyze_ticker function remains the same...

            # Update the chart title and layout in analyze_ticker function
            # Get latest prices and use safe formatting
            latest_data = data.iloc[-1]
            latest_open = safe_format_price(latest_data['Open'])
            latest_close = safe_format_price(latest_data['Close'])
            
            # Update layout with safely formatted values
            fig.update_layout(
                title=dict(
                    text=f"{ticker} Stock Price (Open: {latest_open} Close: {latest_close})",
                    y=0.95,
                    x=0.5,
                    xanchor='center',
                    yanchor='top',
                    pad=dict(b=20),
                    font=dict(size=12)
                ),
                margin=dict(t=80, b=50, l=50, r=50)
            )

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
                f"Identify any potential patterns, signals, or trends that you observe. "
                F"Identify potential support and resistance levels, and any other key insights. "
                f"Provide a detailed justification of your analysis, explaining what patterns, signals, and trends you observe. "
                f"Then, based solely on the chart, patterns, signals, and significant levels provide a recommendation from the following options: "
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
                if (json_start_index != -1 and json_end_index > json_start_index):
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
            # st.error(f"Error analyzing {ticker}: {str(e)}")  # Commented out error message
            # st.error(f"Analysis error details: {str(e)}")    # Commented out error details
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

    # In the chart display section, add debug logging and ensure data is passed correctly
    # First loop to collect all analyses
    for ticker in st.session_state["stock_data"]:
        data = st.session_state["stock_data"][ticker]
        # log_debug(f"Pre-analysis data for {ticker}", data)  # Commented out debug logging
        
        # Create a copy of the data to avoid modifying the original
        data = data.copy()
        
        # Convert data to numeric if needed
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col].astype(str), errors='coerce')
        
        # Verify data after conversion
        # log_debug(f"Post-conversion data for {ticker}", data)  # Commented out debug logging
        
        if data.empty:
            st.error(f"No valid data for {ticker} after conversion")
            continue
            
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
        
        # Create styled HTML table with colored recommendations using the same colors
        recommendation_colors = {
            "Strong Buy": "rgba(0, 128, 0, 0.7)",      # Semi-transparent green
            "Buy": "rgba(144, 238, 144, 0.7)",         # Semi-transparent lightgreen
            "Weak Buy": "rgba(152, 251, 152, 0.7)",    # Semi-transparent palegreen
            "Hold": "rgba(255, 255, 0, 0.7)",          # Semi-transparent yellow
            "Weak Sell": "rgba(255, 192, 203, 0.7)",   # Semi-transparent pink
            "Sell": "rgba(240, 128, 128, 0.7)",        # Semi-transparent lightcoral
            "Strong Sell": "rgba(255, 0, 0, 0.7)",     # Semi-transparent red
            "Error": "rgba(128, 128, 128, 0.7)",       # Semi-transparent gray
            "N/A": "rgba(128, 128, 128, 0.7)"          # Semi-transparent gray
        }
        
        # Create HTML table with enhanced styling
        html_table = '''
        <style>
        .summary-table {
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            margin: 10px 0;
            font-size: 0.9em;
        }
        .summary-table th, .summary-table td {
            padding: 12px;
            text-align: left;
            border: 1px solid #ddd;
        }
        .summary-table th {
            background-color: #f8f9fa;
            font-weight: bold;
        }
        .recommendation-cell {
            padding: 5px 10px;
            border-radius: 4px;
            color: black;
            text-align: center;
            font-weight: 500;
            border: 1px solid rgba(0,0,0,0.1);
            box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        }
        </style>
        <table class="summary-table">
        <tr><th>Stock</th><th>Recommendation</th></tr>
        '''
        
        # Add rows with enhanced styling
        for _, row in df_summary.iterrows():
            color = recommendation_colors.get(row['Recommendation'], "rgba(128, 128, 128, 0.7)")
            html_table += f'''
            <tr>
                <td>{row["Stock"]}</td>
                <td><div class="recommendation-cell" style="background-color:{color}">{row["Recommendation"]}</div></td>
            </tr>
            '''
        
        html_table += '</table>'
        st.markdown(html_table, unsafe_allow_html=True)

        # Display all stocks' data in Overall Summary
        st.subheader("Raw Data for All Stocks")
        for stock, stock_data in st.session_state["stock_data"].items():
            with st.expander(f"Raw Data for {stock} ({selected_time_frame})"):
                st.dataframe(stock_data)

    # Display individual stock tabs section
    for i, ticker in enumerate(st.session_state["stock_data"]):
        with tabs[i + 1]:
            # Get the data and ensure it's properly converted
            data = st.session_state["stock_data"][ticker].copy()
            
            # Ensure numeric types before displaying
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in numeric_columns:
                if col in data.columns:
                    try:
                        data[col] = pd.to_numeric(data[col].astype(str), errors='coerce')
                    except Exception as e:
                        pass  # Commented out: st.warning(f"Warning: Could not convert {col} column to numeric: {str(e)})
            
            fig, result = fig_results[ticker]
            
            # First row: Title and Recommendation
            col1, col2 = st.columns([3, 2])
            with col1:
                try:
                    latest_data = data.iloc[-1]
                    open_price = safe_format_price(latest_data.get('Open'))
                    close_price = safe_format_price(latest_data.get('Close'))
                    
                    # Get the recommendation and its color with improved styling
                    recommendation = result.get("action", "N/A")
                    recommendation_colors = {
                        "Strong Buy": "rgba(0, 128, 0, 0.7)",  # Semi-transparent green
                        "Buy": "rgba(144, 238, 144, 0.7)",     # Semi-transparent lightgreen
                        "Weak Buy": "rgba(152, 251, 152, 0.7)", # Semi-transparent palegreen
                        "Hold": "rgba(255, 255, 0, 0.7)",      # Semi-transparent yellow
                        "Weak Sell": "rgba(255, 192, 203, 0.7)", # Semi-transparent pink
                        "Sell": "rgba(240, 128, 128, 0.7)",    # Semi-transparent lightcoral
                        "Strong Sell": "rgba(255, 0, 0, 0.7)", # Semi-transparent red
                        "Error": "rgba(128, 128, 128, 0.7)",   # Semi-transparent gray
                        "N/A": "rgba(128, 128, 128, 0.7)"      # Semi-transparent gray
                    }
                    rec_color = recommendation_colors.get(recommendation, "rgba(128, 128, 128, 0.7)")
                    
                    st.markdown(f"""
                        <h3 style='margin-bottom: 0px;'>
                            Analysis for {ticker}
                            <span style='font-size: 0.8em; font-weight: normal; color: #666;'>
                                (Open: {open_price} Close: {close_price})
                            </span>
                            <span style='font-size: 0.8em; margin-left: 10px; padding: 2px 8px; border-radius: 4px; 
                                background-color: {rec_color}; color: black; border: 1px solid rgba(0,0,0,0.1); 
                                font-weight: 500; box-shadow: 0 1px 2px rgba(0,0,0,0.1);'>
                                {recommendation}
                            </span>
                        </h3>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    pass
                    
            # ... rest of the tab display code remains the same ...

            # Second row: Financial Metrics
            metrics = get_financial_metrics(ticker)
            if metrics:
                st.markdown("""
                    <style>
                        .metrics-grid {
                            display: flex;
                            flex-direction: row;
                            flex-wrap: nowrap;
                            gap: 8px;
                            overflow-x: auto;
                            padding: 8px 0;
                            margin: 10px 0;
                            width: 100%;
                        }
                        .metric-box {
                            flex: 0 0 auto;
                            background-color: #f8f9fa;
                            border-radius: 4px;
                            padding: 8px;
                            text-align: center;
                            min-width: 120px;
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
                """, unsafe_allow_html=True)
                
                # Create the metrics HTML without extra newlines
                metrics_html = '<div class="metrics-grid">'
                for key, value in metrics.items():
                    metrics_html += f'<div class="metric-box"><div class="metric-label">{key}</div><div class="metric-value">{value}</div></div>'
                metrics_html += '</div>'
                
                st.markdown(metrics_html, unsafe_allow_html=True)
            
            # Third row: Chart and Analysis
            st.plotly_chart(fig, key=f"plotly_chart_{ticker}")
            st.write("**Detailed Justification:**")
            st.write(result.get("justification", "No justification provided."))
            
            # Show only current stock's data in its tab
            with st.expander(f"Raw Data for {ticker} ({selected_time_frame})"):
                st.dataframe(data)

else:
    st.info("Please fetch stock data using the sidebar.")

# Comment out debug testing buttons at the bottom of the file
# if st.sidebar.button("Test yfinance API"):
#     test_tickers = ["AAPL", "MSFT", "GOOG"]  # Test multiple reliable tickers
#     for test_ticker in test_tickers:
#         try:
#             # Test with shorter timeframe first
#             test_data = fetch_with_retry(
#                 test_ticker,
#                 start_date=datetime.today() - timedelta(days=5),
#                 end_date=datetime.today(),
#                 interval="1d"
#             )
            
#             if not test_data.empty:
#                 st.sidebar.success(f"✅ {test_ticker}: Successfully fetched {len(test_data)} rows")
#                 log_debug(f"Test Data for {test_ticker}", test_data)
                
#                 # Verify data quality
#                 missing_data = test_data.isnull().sum()
#                 if missing_data.sum() > 0:
#                     st.sidebar.warning(f"⚠️ {test_ticker}: Contains some missing values:\n{missing_data}")
#             else:
#                 st.sidebar.error(f"❌ {test_ticker}: No data received")
                
#         except Exception as e:
#             st.sidebar.error(f"❌ {test_ticker}: API test failed: {str(e)}")
    
#     st.sidebar.info("💡 If tests fail, try another ticker or wait a few minutes before retrying.")

# if st.sidebar.button("Basic API Test"):
#     success, result = test_yfinance_connection()
#     if success:
#         st.sidebar.success("✅ Basic API test successful!")
#         st.sidebar.dataframe(result.head())
#     else:
#         st.sidebar.error(f"❌ API test failed: {result}")

# if st.sidebar.button("Test Yahoo Connection"):
#     success, message = test_yahoo_connection()
#     if success:
#         st.sidebar.success("✅ Direct Yahoo Finance connection test successful")
#     else:
#         st.sidebar.error(f"❌ Yahoo Finance connection test failed: {message}")