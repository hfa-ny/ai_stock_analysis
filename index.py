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

# Helper to safely format prices
def safe_format_price(value):
    """Safely format price values from Series"""
    try:
        if pd.isna(value):
            return "N/A"
        return f"${float(value):.2f}"
    except (ValueError, TypeError):
        return "N/A"

# Debug logging (currently disabled)
def log_debug(title, data):
    # st.sidebar.markdown(f"**Debug: {title}**")
    # if isinstance(data, pd.DataFrame):
    #     st.sidebar.write(f"Shape: {data.shape}")
    #     st.sidebar.write("First few rows:")
    #     st.sidebar.write(data.head())
    # else:
    #     st.sidebar.write(data)
    pass

# Test yfinance connection
def test_yfinance_connection():
    try:
        ticker = yf.Ticker("AAPL")
        end = datetime.now()
        start = end - timedelta(days=5)
        hist = ticker.history(start=start, end=end, interval='1d', auto_adjust=False, repair=True)
        if hist.empty:
            return False, "No data returned from yfinance"
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in hist.columns for col in required_cols):
            return False, f"Missing columns. Found: {list(hist.columns)}"
        return True, hist
    except Exception as e:
        return False, f"Error: {str(e)}"

# Test direct connection to Yahoo Finance
def test_yahoo_connection():
    url = "https://query1.finance.yahoo.com/v8/finance/chart/AAPL?range=1d&interval=1d"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return True, "Connection successful"
        return False, f"Connection failed with status code: {response.status_code}"
    except Exception as e:
        return False, f"Connection error: {str(e)}"

# Get real-time quote with caching.
@st.cache_data(ttl=60)  # Cache for 60 seconds
def get_real_time_quote(ticker, max_retries=3):
    """Get real-time quote data using yfinance.fast_info.
    Note: fast_info returns a dictionary, so we check keys instead of attributes.
    """
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                time.sleep(2 * attempt)
            yf_ticker = yf.Ticker(ticker)
            info = yf_ticker.fast_info  # fast_info is a dict now
            if 'last_price' in info and 'previous_close' in info:
                return {
                    'current_price': info['last_price'],
                    'previous_close': info['previous_close'],
                    'change_percent': (info['last_price'] / info['previous_close'] - 1)
                }
        except Exception as e:
            if attempt == max_retries - 1:
                return None
            continue
    return None

# Configure the API key (use Streamlit secrets or environment variables)
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# Select the Gemini model
MODEL_NAME = 'gemini-2.0-flash'
gen_model = genai.GenerativeModel(MODEL_NAME)

# Custom CSS to fix header spacing
st.markdown("""
    <style>
        .block-container { padding-top: 3rem !important; padding-bottom: 1rem !important; }
        header { margin-bottom: 2rem !important; }
        .main > div { padding-left: 2rem !important; padding-right: 2rem !important; }
        .financial-metrics { font-size: 0.8rem !important; display: flex; flex-wrap: wrap; gap: 0.5rem; justify-content: space-between; }
        .metric-item { flex: 1 1 auto; min-width: 120px; padding: 0.3rem; }
        .metric-label { color: #666; font-size: 0.7rem !important; }
        .metric-value { font-weight: bold; font-size: 0.8rem !important; }
        #MainMenu {visibility: visible;}
        header {visibility: visible;}
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Color mapping for recommendations
RECOMMENDATION_COLORS = {
    "Strong Buy": "rgba(0, 128, 0, 0.7)",
    "Buy": "rgba(144, 238, 144, 0.7)",
    "Weak Buy": "rgba(152, 251, 152, 0.7)",
    "Hold": "rgba(255, 255, 0, 0.7)",
    "Weak Sell": "rgba(255, 192, 203, 0.7)",
    "Sell": "rgba(240, 128, 128, 0.7)",
    "Strong Sell": "rgba(255, 0, 0, 0.7)",
    "Error": "rgba(128, 128, 128, 0.7)",
    "N/A": "rgba(128, 128, 128, 0.7)"
}

st.title("HFA AI-Powered Technical Stock Analysis")
st.sidebar.header("Configuration")

# Input for multiple tickers
tickers_input = st.sidebar.text_input("Enter Stock Tickers (comma-separated):", "AAPL,^GSPC")
tickers = [ticker.strip().upper() for ticker in tickers_input.split(",") if ticker.strip()]

# Date range selection
end_date_default = datetime.today() + timedelta(days=1)  # Add one day to include today's data
start_date_default = end_date_default - timedelta(days=365)
start_date = st.sidebar.date_input("Start Date", value=start_date_default)
end_date = st.sidebar.date_input("End Date", value=end_date_default)

# Time Frame Options
time_frame_options = {
    "1day": {"interval": "1d", "days": 365, "max_points": None},
    "5day": {"interval": "5d", "days": 365, "max_points": None},
    "1week": {"interval": "1wk", "days": 730, "max_points": None},
    "1month": {"interval": "1mo", "days": 1825, "max_points": None},
    "3month": {"interval": "3mo", "days": 1825, "max_points": None},
    "1min": {"interval": "1m", "days": 100, "max_points": None},  
    "5min": {"interval": "5m", "days": 150, "max_points": None},  
    "15min": {"interval": "15m", "days": 150, "max_points": None},
    "30min": {"interval": "30m", "days": 150, "max_points": None},
    "1hour": {"interval": "1h", "days": 150, "max_points": None},
    "90min": {"interval": "90m", "days": 150, "max_points": None},
    "2hour": {"interval": "2h", "days": 120, "max_points": None},
    "4hour": {"interval": "4h", "days": 180, "max_points": None},
}

def show_timeframe_warning(selected_timeframe):
    intraday_frames = ["1min", "5min","15min", "30min", "1hour", "90min", "2hour"]
    if selected_timeframe in intraday_frames:
        st.sidebar.warning(f"""
        ⚠️ Important Note for {selected_timeframe} timeframe:
        - Limited to last {time_frame_options[selected_timeframe]['days']} days only
        - Data might be delayed or limited
        - May not work outside market hours
        - Premium API limits may apply
        """)

selected_time_frame = st.sidebar.selectbox(
    "Select Time Frame",
    list(time_frame_options.keys()),
    index=list(time_frame_options.keys()).index("1day")
)
show_timeframe_warning(selected_time_frame)

# Technical indicators selection
st.sidebar.subheader("Technical Indicators")
indicators = st.sidebar.multiselect(
    "Select Indicators:",
    ["20-Day SMA", "20-Day EMA", "20-Day Bollinger Bands", "VWAP", "RSI"],
    default=["20-Day SMA"]
)

def fetch_yahoo_finance_data(ticker, start_date, end_date):
    """Backup method to fetch data directly from Yahoo Finance API"""
    try:
        start_timestamp = int(pd.Timestamp(start_date).timestamp())
        end_timestamp = int(pd.Timestamp(end_date).timestamp())
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?period1={start_timestamp}&period2={end_timestamp}&interval=1d"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers)
        data = response.json()
        timestamps = data['chart']['result'][0]['timestamp']
        quote = data['chart']['result'][0]['indicators']['quote'][0]
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
            stock = yf.Ticker(ticker)
            data = stock.history(start=start_date, end=end_date, interval=interval, repair=True)
            if not data.empty:
                return data
            backup_data = fetch_yahoo_finance_data(ticker, start_date, end_date)
            if not backup_data.empty:
                return backup_data
            if attempt < max_retries - 1:
                time.sleep(delay * (attempt + 1))
        except Exception as e:
            if attempt == max_retries - 1:
                try:
                    data = yf.download(ticker, period="1mo", progress=False)
                    if not data.empty:
                        return data
                except:
                    pass
                raise Exception(f"All fetch attempts failed for {ticker}")
            time.sleep(delay * (attempt + 1))
    return pd.DataFrame()

if st.sidebar.button("Fetch Data"):
    stock_data = {}
    failed_tickers = []
    for ticker in tickers:
        try:
            max_days = time_frame_options[selected_time_frame]["days"]
            interval = time_frame_options[selected_time_frame]["interval"]
            adjusted_start_date = datetime.today() - timedelta(days=max_days)
            if start_date > adjusted_start_date.date():
                adjusted_start_date = datetime.combine(start_date, datetime.min.time())
            data = fetch_with_retry(ticker, start_date=adjusted_start_date, end_date=end_date, interval=interval)
            if not data.empty:
                stock_data[ticker] = data
            else:
                failed_tickers.append(ticker)
                st.warning(f"No data found for {ticker}. The symbol might be incorrect or delisted.")
        except Exception as e:
            failed_tickers.append(ticker)
            st.error(f"Error fetching {ticker}: {str(e)}")
    if stock_data:
        st.session_state["stock_data"] = stock_data
        if failed_tickers:
            st.warning(f"Failed to fetch data for: {', '.join(failed_tickers)}")
    else:
        st.error("No data was loaded for any ticker")

if "stock_data" in st.session_state and st.session_state["stock_data"]:
    def downsample_data(data, max_points):
        if max_points and len(data) > max_points:
            sample_interval = len(data) // max_points
            return data.iloc[::sample_interval]
        return data

    def get_financial_metrics(ticker_symbol):
        try:
            if ticker_symbol.startswith('^'):
                return {"Type": "Market Index", "Note": "Detailed metrics not available for indices"}
            time.sleep(2)
            ticker = yf.Ticker(ticker_symbol)
            info = ticker.info
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
            return {}

    def analyze_ticker(ticker, data, indicators, timeframe):
        """Add timeframe parameter to adjust chart display"""
        try:
            if data.isna().any().any():
                data = data.dropna()
            if data.empty:
                raise ValueError(f"No data available for {ticker}")
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in numeric_columns:
                if col in data.columns:
                    if not pd.api.types.is_numeric_dtype(data[col]):
                        data[col] = pd.to_numeric(data[col].astype(str), errors='coerce')
            data = data.dropna(subset=numeric_columns)
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index)
            data = data.sort_index().loc[~data.index.duplicated(keep='first')]
            if data.empty:
                raise ValueError(f"No valid numeric data available for {ticker} after cleaning")
            candlestick = go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name=f"Price ({timeframe})",  # Add timeframe to name
                increasing_line_color='#26A69A',
                decreasing_line_color='#EF5350'
            )
            fig = go.Figure(data=[candlestick])
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
                        delta = data['Close'].diff()
                        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                        rs = gain / loss
                        rsi = 100 - (100 / (1 + rs))
                        fig.add_trace(go.Scatter(x=data.index, y=rsi, name='RSI', yaxis="y2"))
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
                    pass
            for ind in indicators:
                add_indicator(ind)
            fig.update_layout(
                title=f"{ticker} Stock Price ({timeframe})",
                yaxis_title="Price",
                xaxis_title="Date",
                template="plotly_white",
                height=600
            )
            if timeframe in ["1min", "5min", "15min", "30min"]:
                fig.update_xaxes(
                    rangeslider_visible=False,
                    rangebreaks=[
                        dict(bounds=["sat", "mon"]),  # hide weekends
                        dict(bounds=[20, 4], pattern="hour"),  # hide non-trading hours
                    ]
                )
            elif timeframe in ["1hour", "2hour", "4hour"]:
                fig.update_xaxes(
                    rangeslider_visible=False,
                    rangebreaks=[
                        dict(bounds=["sat", "mon"]),  # hide weekends
                    ]
                )
            latest_data = data.iloc[-1]
            latest_open = safe_format_price(latest_data['Open'])
            latest_close = safe_format_price(latest_data['Close'])
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
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                fig.write_image(tmpfile.name)
                tmpfile_path = tmpfile.name
            with open(tmpfile_path, "rb") as f:
                image_bytes = f.read()
            os.remove(tmpfile_path)
            image_part = {"data": image_bytes, "mime_type": "image/png"}
            analysis_prompt = (
                f"You are a Stock Trader specializing in Technical Analysis at a top financial institution. "
                f"Analyze the stock chart for {ticker} based on its candlestick chart and the displayed technical indicators. "
                f"Identify any potential patterns, signals, or trends that you observe. "
                f"Identify potential support and resistance levels, and any other key insights. "
                f"Always try to give your best estimates on buy and sell levels."
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
            empty_fig = go.Figure()
            empty_fig.add_annotation(
                text=f"Error during analysis: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return empty_fig, {"action": "Error", "justification": f"Analysis error: {str(e)}"}

    overall_results = []
    fig_results = {}
    for ticker in st.session_state["stock_data"]:
        data = st.session_state["stock_data"][ticker].copy()
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col].astype(str), errors='coerce')
        if data.empty:
            st.error(f"No valid data for {ticker} after conversion")
            continue
        fig, result = analyze_ticker(ticker, data, indicators, selected_time_frame)
        overall_results.append({"Stock": ticker, "Recommendation": result.get("action", "N/A")})
        fig_results[ticker] = (fig, result)
    tab_names = ["Overall Summary"] + list(st.session_state["stock_data"].keys())
    tabs = st.tabs(tab_names)
    with tabs[0]:
        st.markdown("## Market Overview")
        st.markdown("""
            <style>
            .modern-summary {
                display: flex;
                flex-wrap: wrap;
                gap: 1rem;
                padding: 1rem 0;
                margin-bottom: 2rem;
            }
            .stock-card {
                flex: 1 1 300px;
                background: white;
                border-radius: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                padding: 1.5rem;
                transition: transform 0.2s;
            }
            .stock-card:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }
            .stock-symbol {
                font-size: 1.2rem;
                font-weight: 600;
                color: #1a1a1a;
                margin-bottom: 0.5rem;
            }
            .stock-prices {
                font-size: 0.9rem;
                color: #666;
                margin-bottom: 0.8rem;
            }
            .stock-recommendation {
                display: inline-block;
                padding: 0.4rem 1rem;
                border-radius: 20px;
                font-weight: 500;
                font-size: 0.9rem;
                text-align: center;
                box-shadow: 0 1px 2px rgba(0,0,0,0.1);
            }
            </style>
            <div class="modern-summary">
        """, unsafe_allow_html=True)
        @st.cache_resource(ttl=300)
        def get_cached_ticker(ticker):
            return yf.Ticker(ticker)
        for ticker in st.session_state["stock_data"]:
            data = st.session_state["stock_data"][ticker]
            latest_data = data.iloc[-1]
            real_time = get_real_time_quote(ticker)
            recommendation = next((item["Recommendation"] for item in overall_results if item["Stock"] == ticker), "N/A")
            color = RECOMMENDATION_COLORS.get(recommendation, "rgba(128, 128, 128, 0.7)")
            change_color = "green"
            change_symbol = "↑"
            if real_time and real_time['change_percent']:
                if real_time['change_percent'] < 0:
                    change_color = "red"
                    change_symbol = "↓"
            current_price = f"${real_time['current_price']:.2f}" if real_time and real_time['current_price'] else "N/A"
            change_pct = f"{real_time['change_percent']*100:.2f}%" if real_time and real_time['change_percent'] else "N/A"
            st.markdown(f"""
                <div class="stock-card">
                    <div class="stock-symbol">{ticker}</div>
                    <div class="stock-prices">
                        Current: {current_price}
                        <span style="color: {change_color}">
                            {change_symbol} {change_pct}
                        </span>
                    </div>
                    <div class="stock-details">
                        Previous Close: {safe_format_price(latest_data['Close'])}
                    </div>
                    <div class="stock-recommendation" style="background-color: {color}">
                        {recommendation}
                    </div>
                </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)
        st.markdown("""
            <h2 class="historical-header">Historical Price Data</h2>
            <p class="historical-desc">
                Expand each symbol below to view detailed historical data and performance metrics.
                Data includes full price history and calculated statistics for the selected time period.
            </p>
        """, unsafe_allow_html=True)
        for ticker in st.session_state["stock_data"]:
            data = st.session_state["stock_data"][ticker]
            with st.expander(f"Historical Data - {ticker}"):
                stats = pd.DataFrame({
                    'Open': [data['Open'].min(), data['Open'].max(), data['Open'].mean()],
                    'High': [data['High'].min(), data['High'].max(), data['High'].mean()],
                    'Low': [data['Low'].min(), data['Low'].max(), data['Low'].mean()],
                    'Close': [data['Close'].min(), data['Close'].max(), data['Close'].mean()],
                    'Volume': [data['Volume'].min(), data['Volume'].max(), data['Volume'].mean()]
                }, index=['Min', 'Max', 'Average'])
                for col in stats.columns:
                    if col != 'Volume':
                        stats[col] = stats[col].apply(lambda x: f"${x:.2f}")
                    else:
                        stats[col] = stats[col].apply(lambda x: f"{x:,.0f}")
                st.markdown("### Key Statistics")
                st.dataframe(stats, use_container_width=True)
                st.markdown("### Historical Data")
                formatted_data = data.copy()
                for col in ['Open', 'High', 'Low', 'Close']:
                    formatted_data[col] = formatted_data[col].apply(lambda x: f"${x:.2f}")
                formatted_data['Volume'] = formatted_data['Volume'].apply(lambda x: f"{x:,.0f}")
                st.dataframe(formatted_data, use_container_width=True)
    for i, ticker in enumerate(st.session_state["stock_data"]):
        with tabs[i + 1]:
            data = st.session_state["stock_data"][ticker].copy()
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in numeric_columns:
                if col in data.columns:
                    try:
                        data[col] = pd.to_numeric(data[col].astype(str), errors='coerce')
                    except Exception as e:
                        pass
            fig, result = fig_results[ticker]
            col1, col2 = st.columns([3, 2])
            with col1:
                try:
                    latest_data = data.iloc[-1]
                    open_price = safe_format_price(latest_data.get('Open'))
                    close_price = safe_format_price(latest_data.get('Close'))
                    recommendation = result.get("action", "N/A")
                    rec_color = RECOMMENDATION_COLORS.get(recommendation, "rgba(128, 128, 128, 0.7)")
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
                metrics_html = '<div class="metrics-grid">'
                for key, value in metrics.items():
                    metrics_html += f'<div class="metric-box"><div class="metric-label">{key}</div><div class="metric-value">{value}</div></div>'
                metrics_html += '</div>'
                st.markdown(metrics_html, unsafe_allow_html=True)
            st.plotly_chart(fig, key=f"plotly_chart_{ticker}")
            st.write("**Detailed Justification:**")
            st.write(result.get("justification", "No justification provided."))
            with st.expander(f"Raw Data for {ticker} ({selected_time_frame})"):
                st.dataframe(data)
else:
    st.info("Please fetch stock data using the sidebar.")

st.markdown("""
    <style>
    .market-overview {
        display: flex;
        flex-flow: row nowrap;
        gap: 1rem;
        padding: 1rem 0;
        margin-bottom: 2rem;
        width: 100%;
        overflow-x: auto;
    }
    .stock-card {
        flex: 0 0 300px;
        background: white;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        padding: 1.5rem;
        transition: all 0.3s ease;
        position: relative;
        border: 1px solid rgba(0,0,0,0.05);
    }
    .stock-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    </style>
    <div class="market-overview">
""", unsafe_allow_html=True)
