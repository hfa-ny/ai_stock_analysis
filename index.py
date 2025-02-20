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
import time
from concurrent.futures import ThreadPoolExecutor
import pandas_market_calendars as mcal
import requests.exceptions
import requests

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
    st.sidebar.markdown(f"**Debug: {title}**")
    if isinstance(data, pd.DataFrame):
        st.sidebar.write(f"Shape: {data.shape}")
        st.sidebar.write("First few rows:")
        st.sidebar.write(data.head())
    else:
        st.sidebar.write(data)

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
        
        if hist.empty:
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

def fetch_with_retry(ticker, start_date, end_date, interval, max_retries=3, delay=2):
    """Fetch stock data with retry logic"""
    for attempt in range(max_retries):
        try:
            # Simple, direct approach using just the period parameter
            data = yf.download(
                ticker,
                period="1y",  # Use fixed period first
                interval=interval,
                progress=False,
                show_errors=False
            )
            
            if not data.empty:
                # Filter to the requested date range after successful fetch
                if isinstance(start_date, datetime):
                    start_ts = start_date
                else:
                    start_ts = pd.Timestamp(start_date)
                
                if isinstance(end_date, datetime):
                    end_ts = end_date
                else:
                    end_ts = pd.Timestamp(end_date)
                
                mask = (data.index >= start_ts) & (data.index <= end_ts)
                data = data[mask]
                
                if not data.empty:
                    return data
            
            # If no data received, try the backup method
            backup_data = fetch_yahoo_finance_data(ticker, start_date, end_date)
            if not backup_data.empty:
                return backup_data
            
            time.sleep(delay * (attempt + 1))
            
        except Exception as e:
            if attempt == max_retries - 1:
                # Final attempt - try the most basic fetch possible
                try:
                    data = yf.download(ticker, period="1mo", progress=False)
                    if not data.empty:
                        return data
                except:
                    pass
                raise Exception(f"All fetch attempts failed for {ticker}: {str(e)}")
            time.sleep(delay * (attempt + 1))
    
    return pd.DataFrame()

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

# In the "Fetch Data" button click handler, add logging
if st.sidebar.button("Fetch Data"):
    stock_data = {}
    failed_tickers = []
    
    for ticker in tickers:
        try:
            # Calculate appropriate start date based on selected timeframe
            max_days = time_frame_options[selected_time_frame]["days"]
            interval = time_frame_options[selected_time_frame]["interval"]

            log_debug("Fetch Parameters", {
                "ticker": ticker,
                "interval": interval,
                "max_days": max_days,
                "start_date": start_date,
                "end_date": end_date
            })

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
            
            log_debug(f"Fetched Data for {ticker}", data)

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
        log_debug("Session State Data", stock_data)
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
    def analyze_ticker(ticker, data, indicators):
        try:
            # Initial data validation
            if data is None or data.empty:
                raise ValueError(f"No data available for {ticker}")
            
            # Ensure numeric data types
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in numeric_columns:
                if col in data.columns:
                    if not pd.api.types.is_numeric_dtype(data[col]):
                        # Convert to string first to handle any data type
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
                
            # Build candlestick chart with validation
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
            
            # Update layout with error handling
            try:
                latest_data = data.iloc[-1]
                latest_open = safe_format_price(latest_data['Open'])
                latest_close = safe_format_price(latest_data['Close'])
                
                title_text = f"{ticker} Stock Price"
                if latest_open != "N/A" and latest_close != "N/A":
                    title_text += f" (Open: {latest_open} Close: {latest_close})"
                
                fig.update_layout(
                    title=dict(
                        text=title_text,
                        y=0.95,
                        x=0.5,
                        xanchor='center',
                        yanchor='top'
                    ),
                    yaxis_title="Price",
                    xaxis_title="Date",
                    xaxis_rangeslider_visible=False,
                    template="plotly_white",
                    height=600
                )
                
            except Exception as layout_error:
                st.warning(f"Warning: Could not update chart layout with latest prices: {layout_error}")
                fig.update_layout(title=f"{ticker} Stock Price")

            # Debug data
            st.sidebar.write(f"Debug - Data shape for {ticker}: {data.shape}")
            st.sidebar.write(f"Debug - First few rows of data:")
            st.sidebar.write(data.head())
            
            # Make sure we have the required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in data.columns for col in required_columns):
                missing_cols = [col for col in required_columns if col not in data.columns]
                raise ValueError(f"Missing required columns: {missing_cols}")

            # Ensure data is properly sorted by date
            data = data.sort_index()
            
            # Get the max points limit and interval for the selected timeframe
            timeframe_info = time_frame_options[selected_time_frame]
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

            # Clean data - handle NaN values
            data = data.copy()  # Create a copy to avoid SettingWithCopyWarning
            data = data.dropna()
            
            # Build candlestick chart
            fig = go.Figure(data=[
                go.Candlestick(
                    x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name="Price"
                )
            ])

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

    # In the chart display section, add debug logging and ensure data is passed correctly
    # First loop to collect all analyses
    for ticker in st.session_state["stock_data"]:
        data = st.session_state["stock_data"][ticker]
        log_debug(f"Pre-analysis data for {ticker}", data)
        
        # Create a copy of the data to avoid modifying the original
        data = data.copy()
        
        # Convert data to numeric if needed
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col].astype(str), errors='coerce')
        
        # Verify data after conversion
        log_debug(f"Post-conversion data for {ticker}", data)
        
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
        
        # Create styled HTML table with colored recommendations
        recommendation_colors = {
            "Strong Buy": "green",
            "Buy": "lightgreen",
            "Weak Buy": "palegreen",
            "Hold": "yellow",
            "Weak Sell": "pink",
            "Sell": "lightcoral",
            "Strong Sell": "red"
        }
        
        # Create compact HTML without extra whitespace or newlines
        html_table = '<style>.summary-table{width:100%;border-collapse:collapse;margin:10px 0;font-size:0.9em;}.summary-table th,.summary-table td{padding:8px;text-align:left;border:1px solid #ddd;}.summary-table th{background-color:#f8f9fa;font-weight:bold;}.recommendation-cell{padding:5px 10px;border-radius:4px;color:black;text-align:center;}</style><table class="summary-table"><tr><th>Stock</th><th>Recommendation</th></tr>'
        
        # Add rows without extra formatting
        for _, row in df_summary.iterrows():
            color = recommendation_colors.get(row['Recommendation'], 'gray')
            html_table += f'<tr><td>{row["Stock"]}</td><td><div class="recommendation-cell" style="background-color:{color}">{row["Recommendation"]}</div></td></tr>'
        
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
                        st.warning(f"Warning: Could not convert {col} column to numeric: {str(e)}")
            
            fig, result = fig_results[ticker]
            
            # First row: Title and Recommendation
            col1, col2 = st.columns([3, 2])
            with col1:
                try:
                    latest_data = data.iloc[-1]
                    open_price = safe_format_price(latest_data.get('Open'))
                    close_price = safe_format_price(latest_data.get('Close'))
                    
                    st.markdown(f"""
                        <h3 style='margin-bottom: 0px;'>
                            Analysis for {ticker}
                            <span style='font-size: 0.8em; font-weight: normal; color: #666;'>
                                (Open: {open_price} Close: {close_price})
                            </span>
                        </h3>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error displaying price data: {str(e)}")
                    
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

# docker run -p 8501:8501 -e GOOGLE_API_KEY="your_actual_api_key" streamlit-stock-analysis

if st.sidebar.button("Test yfinance API"):
    test_tickers = ["AAPL", "MSFT", "GOOG"]  # Test multiple reliable tickers
    for test_ticker in test_tickers:
        try:
            # Test with shorter timeframe first
            test_data = fetch_with_retry(
                test_ticker,
                start_date=datetime.today() - timedelta(days=5),
                end_date=datetime.today(),
                interval="1d"
            )
            
            if not test_data.empty:
                st.sidebar.success(f"✅ {test_ticker}: Successfully fetched {len(test_data)} rows")
                log_debug(f"Test Data for {test_ticker}", test_data)
                
                # Verify data quality
                missing_data = test_data.isnull().sum()
                if missing_data.sum() > 0:
                    st.sidebar.warning(f"⚠️ {test_ticker}: Contains some missing values:\n{missing_data}")
            else:
                st.sidebar.error(f"❌ {test_ticker}: No data received")
                
        except Exception as e:
            st.sidebar.error(f"❌ {test_ticker}: API test failed: {str(e)}")
    
    st.sidebar.info("💡 If tests fail, try another ticker or wait a few minutes before retrying.")

# Add test button in sidebar
if st.sidebar.button("Basic API Test"):
    success, result = test_yfinance_connection()
    if success:
        st.sidebar.success("✅ Basic API test successful!")
        st.sidebar.dataframe(result.head())
    else:
        st.sidebar.error(f"❌ API test failed: {result}")

# Add this to the sidebar section
if st.sidebar.button("Test Yahoo Connection"):
    success, message = test_yahoo_connection()
    if success:
        st.sidebar.success("✅ Direct Yahoo Finance connection test successful")
    else:
        st.sidebar.error(f"❌ Yahoo Finance connection test failed: {message}")