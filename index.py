## AI-Powered Technical Analysis Dashboard (Gemini 2.0)

# Libraries - Importing necessary Python libraries
import streamlit as st  # Streamlit for creating the web application interface
import google.generativeai as genai
import yfinance as yf  # yfinance to download historical stock market data from Yahoo Finance
import pandas as pd  # pandas for data manipulation and analysis (DataFrames)
import plotly.graph_objects as go  # plotly for creating interactive charts, specifically candlestick charts
import google.generativeai as genai  # google-generativeai to interact with Google's Gemini AI models
import tempfile  # tempfile for creating temporary files (used to save chart images)
import os  # os for interacting with the operating system (e.g., deleting temporary files)
import json  # json for working with JSON data (expected response format from Gemini)
from datetime import datetime, timedelta  # datetime and timedelta for date and time calculations

# Configure the API key - IMPORTANT: Use Streamlit secrets or environment variables for security
# For now, using hardcoded API key - REPLACE WITH YOUR ACTUAL API KEY SECURELY
# In a production environment, it's highly recommended to use Streamlit secrets or environment variables
# to avoid hardcoding sensitive information directly in the code.
# GOOGLE_API_KEY = "your api key goes here"  # Replace "your api key goes here" with your actual Google API key
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])  # Configure the Gemini API with your API key

# Select the Gemini model - using 'gemini-2.0-flash' as a general-purpose model
MODEL_NAME = 'gemini-2.0-flash'  # Specifies the Gemini model to be used. You can choose other models if available.
gen_model = genai.GenerativeModel(MODEL_NAME)  # Initializes a GenerativeModel object to interact with the chosen Gemini model.

# Set up Streamlit app - Configuring the Streamlit web application
st.set_page_config(layout="wide")  # Sets the page layout to 'wide' to utilize more screen width
st.title("AI-Powered Technical Stock Analysis Dashboard")  # Sets the title of the Streamlit application displayed at the top
st.sidebar.header("Configuration")  # Creates a header "Configuration" in the Streamlit sidebar for user input elements

# Input for multiple stock tickers (comma-separated) - User input for stock symbols
tickers_input = st.sidebar.text_input("Enter Stock Tickers (comma-separated):", "AAPL,MSFT,GOOG")  # Creates a text input box in the sidebar
# Label: "Enter Stock Tickers (comma-separated):"
# Default value: "AAPL,MSFT,GOOG"
tickers = [ticker.strip().upper() for ticker in tickers_input.split(",") if ticker.strip()]  # Parses the input string into a list of stock tickers
# 1. tickers_input.split(","): Splits the input string by commas into a list of strings.
# 2. for ticker in ...: Iterates through each string in the list.
# 3. ticker.strip(): Removes leading/trailing whitespace from each ticker string.
# 4. ticker.upper(): Converts each ticker string to uppercase for consistency.
# 5. if ticker.strip(): Filters out empty strings that might result from extra commas or spaces.

# Set the date range: start date = one year before today, end date = today - Date input for historical data range
end_date_default = datetime.today()  # Sets the default end date to today's date
start_date_default = end_date_default - timedelta(days=365)  # Sets the default start date to one year before today
start_date = st.sidebar.date_input("Start Date", value=start_date_default)  # Creates a date input widget in the sidebar for the start date
# Label: "Start Date"
# Default value: start_date_default (one year ago)
end_date = st.sidebar.date_input("End Date", value=end_date_default)  # Creates a date input widget in the sidebar for the end date
# Label: "End Date"
# Default value: end_date_default (today)

# Technical indicators selection (applied to every ticker) - Multiselect for technical indicators
st.sidebar.subheader("Technical Indicators")  # Creates a subheader "Technical Indicators" in the sidebar
indicators = st.sidebar.multiselect(  # Creates a multiselect widget in the sidebar allowing users to choose multiple options
    "Select Indicators:",  # Label for the multiselect widget
    ["20-Day SMA", "20-Day EMA", "20-Day Bollinger Bands", "VWAP"],  # List of available technical indicators to select from
    default=["20-Day SMA"]  # Default selected indicator(s) - in this case, "20-Day SMA"
)

# Button to fetch data for all tickers - Button to trigger data fetching and analysis
if st.sidebar.button("Fetch Data"):
    stock_data = {}
    with st.spinner("Fetching stock data..."):
        for ticker in tickers:
            try:
                # Fetch data with explicit interval
                data = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    interval="1d",  # Explicitly set daily interval
                    progress=False
                )
                
                # Clean and validate data
                if not data.empty:
                    # Remove any NaN values
                    data = data.dropna()
                    if len(data) > 0:
                        stock_data[ticker] = data
                        st.success(f"Successfully loaded data for {ticker}")
                    else:
                        st.warning(f"No valid data points found for {ticker} after cleaning")
                else:
                    st.warning(f"No data found for {ticker}")
                    
            except Exception as e:
                st.error(f"Error fetching data for {ticker}: {str(e)}")
                
    if stock_data:
        st.session_state["stock_data"] = stock_data
        st.success("Stock data loaded successfully for: " + ", ".join(stock_data.keys()))
    else:
        st.error("No valid data was loaded for any ticker")

# Ensure we have data to analyze - Conditional execution of analysis and charting only if stock data is loaded
if "stock_data" in st.session_state and st.session_state["stock_data"]:  # Checks if 'stock_data' exists in session state and is not empty
    # This ensures that the following code runs only after stock data has been successfully fetched.

    # Define a function to build chart, call the Gemini API and return structured result
    def analyze_ticker(ticker, data):  # Defines a function to analyze a single stock ticker
        # First, clean the data by removing NaN values
        data = data.dropna()

        # Check if data is empty after cleaning
        if data.empty:
            st.warning(f"No valid data found for {ticker} after cleaning.")
            return None, {"action": "Error", "justification": "No valid data after cleaning NaN values"}

        # Add debug information
        st.write(f"### Data Shape for {ticker}: {data.shape}")
        st.write("### First few rows of cleaned data:")
        st.dataframe(data.head())

        # Check if required columns exist
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required_columns):
            missing_cols = [col for col in required_columns if col not in data.columns]
            st.error(f"Missing required columns for {ticker}: {missing_cols}")
            return None, {"action": "Error", "justification": f"Missing required columns: {missing_cols}"}

        # Build candlestick chart for the given ticker's data
        fig = go.Figure(data=[  # Creates a Plotly Figure object to hold the chart
            go.Candlestick(  # Creates a Candlestick trace for the stock price data
                x=data.index,  # x-axis values are the dates from the DataFrame index
                open=data['Open'],  # Open prices for the candlestick chart
                high=data['High'],  # High prices for the candlestick chart
                low=data['Low'],  # Low prices for the candlestick chart
                close=data['Close'],  # Close prices for the candlestick chart
                name="Candlestick"  # Name of the trace in the legend
            )
        ])

        # Add selected technical indicators - Nested function to add indicators to the chart
        def add_indicator(indicator):
            try:
                if indicator == "20-Day SMA":
                    sma = data['Close'].rolling(window=20).mean()
                    if len(sma) > 0:  # Simple length check
                        fig.add_trace(go.Scatter(
                            x=data.index,
                            y=sma,
                            mode='lines',
                            name='SMA (20)',
                            line=dict(color='blue')
                        ))
                        
                elif indicator == "20-Day EMA":
                    ema = data['Close'].ewm(span=20, adjust=False).mean()
                    if len(ema) > 0:  # Simple length check
                        fig.add_trace(go.Scatter(
                            x=data.index,
                            y=ema,
                            mode='lines',
                            name='EMA (20)',
                            line=dict(color='orange')
                        ))
                        
                elif indicator == "20-Day Bollinger Bands":
                    bb_sma = data['Close'].rolling(window=20).mean()
                    bb_std = data['Close'].rolling(window=20).std()
                    bb_upper = bb_sma + (bb_std * 2)
                    bb_lower = bb_sma - (bb_std * 2)
                    
                    if len(bb_sma) > 0:  # Simple length check
                        fig.add_trace(go.Scatter(
                            x=data.index,
                            y=bb_upper,
                            mode='lines',
                            name='BB Upper',
                            line=dict(color='gray', dash='dash')
                        ))
                        fig.add_trace(go.Scatter(
                            x=data.index,
                            y=bb_lower,
                            mode='lines',
                            name='BB Lower',
                            line=dict(color='gray', dash='dash')
                         ))
                        fig.add_trace(go.Scatter(
                            x=data.index,
                            y=bb_sma,
                            mode='lines',
                            name='BB Middle',
                            line=dict(color='gray')
                        ))
                        
                elif indicator == "VWAP":
                    # Calculate VWAP
                    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
                    cumulative_tp_vol = (typical_price * data['Volume']).cumsum()
                    cumulative_vol = data['Volume'].cumsum()
                    vwap = cumulative_tp_vol / cumulative_vol
                    
                    if len(vwap) > 0:  # Simple length check
                        fig.add_trace(go.Scatter(
                            x=data.index,
                            y=vwap,
                            mode='lines',
                            name='VWAP',
                            line=dict(color='purple')
                        ))
                        
            except Exception as e:
                st.warning(f"Error adding indicator {indicator}: {str(e)}")

        for ind in indicators:  # Loops through the list of selected indicators
            add_indicator(ind)  # Calls the add_indicator function for each selected indicator to add it to the chart
        fig.update_layout(
            title=f"{ticker} Stock Price",
            yaxis_title="Price",
            xaxis_title="Date",
            xaxis_rangeslider_visible=False
        )

        # Save chart as temporary PNG file and read image bytes - Prepare chart image for Gemini API
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:  # Creates a temporary file to save the chart image
            # suffix=".png": specifies the file extension as .png
            # delete=False: prevents automatic deletion of the file when closed, we need to delete it manually later
            fig.write_image(tmpfile.name)  # Saves the Plotly chart figure as a PNG image to the temporary file
            tmpfile_path = tmpfile.name  # Stores the path to the temporary file
        with open(tmpfile_path, "rb") as f:  # Opens the temporary image file in binary read mode ('rb')
            image_bytes = f.read()  # Reads the entire content of the image file as bytes
        os.remove(tmpfile_path)  # Deletes the temporary image file after reading the bytes

        # Create an image Part - Structure the image data for Gemini API request
        image_part = {  # Creates a dictionary to represent the image part for Gemini API
            "data": image_bytes,  # Image data in bytes format
            "mime_type": "image/png"  # Specifies the MIME type of the image (PNG image)
        }

        # Updated prompt asking for a detailed justification of technical analysis and a recommendation. - Define the prompt for Gemini API
        analysis_prompt = (  # Defines a multi-line string for the prompt to be sent to Gemini
            f"You are a Stock Trader specializing in Technical Analysis at a top financial institution. "  # Sets the persona for Gemini
            f"Analyze the stock chart for {ticker} based on its candlestick chart and the displayed technical indicators. "  # Instructs Gemini to analyze the chart
            f"Provide a detailed justification of your analysis, explaining what patterns, signals, and trends you observe. "  # Asks for detailed justification
            f"Then, based solely on the chart, provide a recommendation from the following options: "  # Asks for a recommendation from a list
            f"'Strong Buy', 'Buy', 'Weak Buy', 'Hold', 'Weak Sell', 'Sell', or 'Strong Sell'. "  # List of recommendation options
            f"Return your output as a JSON object with two keys: 'action' and 'justification'."  # Specifies the desired output format as JSON
        )

        # Call the Gemini API with text and image input - Roles added: "user" for both text and image - Prepare and send request to Gemini API
        contents = [  # Creates a list of content parts for the Gemini API request
            {"role": "user", "parts": [analysis_prompt]},  # Text prompt part with role "user" - the analysis instructions
            {"role": "user", "parts": [image_part]}  # Image part with role "user" - the chart image
        ]

        response = gen_model.generate_content(  # Sends the content to the Gemini model to generate a response
            contents=contents  # Passes the structured content (text and image) to the Gemini API
        )

        try:  # Error handling block for JSON parsing and other potential errors
            # Attempt to parse JSON from the response text
            result_text = response.text  # Gets the text response from Gemini
            # Find the start and end of the JSON object within the text (if Gemini includes extra text)
            json_start_index = result_text.find('{')  # Finds the index of the first '{' character (start of JSON)
            json_end_index = result_text.rfind('}') + 1  # Finds the index of the last '}' character + 1 (end of JSON)
            if json_start_index != -1 and json_end_index > json_start_index:  # Checks if both '{' and '}' were found and in the correct order
                json_string = result_text[json_start_index:json_end_index]  # Extracts the JSON string from the response text
                result = json.loads(json_string)  # Parses the JSON string into a Python dictionary
            else:
                raise ValueError("No valid JSON object found in the response")  # Raises a ValueError if no valid JSON is found

        except json.JSONDecodeError as e:  # Catches JSON parsing errors
            result = {"action": "Error", "justification": f"JSON Parsing error: {e}. Raw response text: {response.text}"}  # Creates an error result dictionary
        except ValueError as ve:  # Catches ValueErrors (e.g., no JSON found)
            result = {"action": "Error", "justification": f"Value Error: {ve}. Raw response text: {response.text}"}  # Creates an error result dictionary
        except Exception as e:  # Catches any other general exceptions
            result = {"action": "Error", "justification": f"General Error: {e}. Raw response text: {response.text}"}  # Creates an error result dictionary

        return fig, result  # Returns the Plotly chart figure and the analysis result (dictionary)

    # Create tabs: first tab for overall summary, subsequent tabs per ticker - Streamlit tab structure for output display
    tab_names = ["Overall Summary"] + list(st.session_state["stock_data"].keys())  # Creates a list of tab names
    # First tab: "Overall Summary", subsequent tabs for each stock ticker
    tabs = st.tabs(tab_names)  # Creates Streamlit tabs using the generated tab names

    # List to store overall results - List to collect analysis results for all tickers
    overall_results = []  # Initializes an empty list to store overall analysis results

    # Process each ticker and populate results - Loop through tickers to analyze and display results
    for i, ticker in enumerate(st.session_state["stock_data"]):  # Loops through each ticker in the 'stock_data' dictionary
        # enumerate provides both index (i) and ticker
        data = st.session_state["stock_data"][ticker]  # Retrieves the stock data DataFrame for the current ticker from session state
        # Analyze ticker: get chart figure and structured output result
        fig, result = analyze_ticker(ticker, data)  # Calls the analyze_ticker function to get chart and analysis result for the current ticker
        overall_results.append({"Stock": ticker, "Recommendation": result.get("action", "N/A")})  # Appends the ticker and its recommendation to the overall_results list
        # result.get("action", "N/A"): safely retrieves 'action' from result, defaults to "N/A" if not found
        # In each ticker-specific tab, display the chart and detailed justification
        with tabs[i + 1]:  # Opens a Streamlit tab for the current ticker (tabs are 0-indexed, "Overall Summary" is tab 0)
            st.subheader(f"Analysis for {ticker}")  # Displays a subheader with the stock ticker in the tab
            st.plotly_chart(fig)  # Displays the Plotly chart in the tab
            st.write("**Detailed Justification:**")  # Writes a bold subheading "Detailed Justification:"
            st.write(result.get("justification", "No justification provided."))  # Displays the justification from the Gemini analysis

    # In the Overall Summary tab, display a table of all results - Display summary table in the first tab
    with tabs[0]:  # Opens the first tab - "Overall Summary" (index 0)
        st.subheader("Overall Structured Recommendations")  # Displays a subheader for the overall summary table
        df_summary = pd.DataFrame(overall_results)  # Creates a pandas DataFrame from the overall_results list for tabular display
        st.table(df_summary)  # Displays the DataFrame as a table in the "Overall Summary" tab
else:  # Executed if 'stock_data' is not in session state or is empty (no data fetched yet)
    st.info("Please fetch stock data using the sidebar.")  # Displays an information message prompting the user to fetch stock data


# docker run -p 8501:8501 -e GOOGLE_API_KEY="your_actual_api_key" streamlit-stock-analysis