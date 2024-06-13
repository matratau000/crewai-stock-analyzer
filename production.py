import os
from datetime import datetime
import yfinance as yf
import matplotlib.pyplot as plt
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# Prompt user to choose LLM
llm_choice = input("Choose LLM (groq/openai): ").strip().lower()
if llm_choice == "groq":
    llm = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")
elif llm_choice == "openai":
    llm = ChatOpenAI(model='gpt-4o', api_key=openai_api_key)  # Loading GPT-4o
else:
    raise ValueError("Invalid LLM choice. Please choose 'groq' or 'openai'.")

# Initialize the SerperDevTool
serper_tool = SerperDevTool(api_key=os.getenv("SERPER_API_KEY"))

stock_to_analyze = input("Enter the stock to analyze: ")
stock_ticker = input("Enter the stock ticker symbol: ")
current_date = input("Enter the current date (YYYY-MM-DD): ")

# Define the tasks and use SerperDevTool for data collection
def collect_stock_data(stock, date):
    query = f"latest stock sentiment data for {stock} after {date}"
    result = serper_tool.run(query=query)
    print("Data Collector Output:", result)  # Debugging: Print the raw result

    # Parsing the result manually
    lines = result.split('\n')
    search_results = []
    current_result = {}
    
    for line in lines:
        if line.startswith("Title: "):
            if current_result:
                search_results.append(current_result)
                current_result = {}
            current_result['Title'] = line.replace("Title: ", "")
        elif line.startswith("Link: "):
            current_result['Link'] = line.replace("Link: ", "")
        elif line.startswith("Snippet: "):
            current_result['Snippet'] = line.replace("Snippet: ", "")
    
    if current_result:
        search_results.append(current_result)
    
    return search_results

def summarize_content(content):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a content summarizer."),
        ("human", f"Summarize the following content: {content}")
    ])
    summary_output = (prompt | llm).invoke({"text": f"Summarize the content"})
    return summary_output.content

def scrape_and_summarize_link(link):
    scrape_tool = ScrapeWebsiteTool(website_url=link)
    try:
        content = scrape_tool.run()
        summary = summarize_content(content)
        return summary
    except Exception as e:
        logging.error(f"Error scraping {link}: {e}")
        return ""

# Collect the stock data
data_summary = collect_stock_data(stock_to_analyze, current_date)
links = [result['Link'] for result in data_summary]

summaries = []
for link in links:
    summary = scrape_and_summarize_link(link)
    if summary:
        summaries.append(summary)
        print(f"Summary for {link}: {summary}")

combined_summary = " ".join(summaries)
final_summary = summarize_content(combined_summary)

# Create task functions directly instead of delegating between agents unnecessarily
def analyze_stock_data(stock, data_summary):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a financial data analyst."),
        ("human", f"Analyze the collected stock sentiment data for {stock}: {data_summary} and predict next week's sentiment.")
    ])
    analysis_output = (prompt | llm).invoke({"text": f"Analyze the stock sentiment data for {stock}"})
    print("Analyze Data Task Output:", analysis_output.content)
    return analysis_output.content

def review_analysis(stock, analysis_output):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a financial data reviewer."),
        ("human", f"Review the analysis of {stock}: {analysis_output} and provide feedback for improvement.")
    ])
    review_output = (prompt | llm).invoke({"text": f"Review the analysis of {stock}"})
    print("Review Analysis Task Output:", review_output.content)
    return review_output.content

def improve_analysis(stock, review_output, analysis_output):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a financial analyst."),
        ("human", f"Based on the review feedback: {review_output}, improve the analysis of {stock}: {analysis_output}.")
    ])
    improved_output = (prompt | llm).invoke({"text": f"Improve the analysis of {stock}"})
    print("Improved Analysis Output:", improved_output.content)
    return improved_output.content

def expert_analysis(stock, price_analysis, sentiment_analysis):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert financial investment banker."),
        ("human", f"Analyze the following price and sentiment data for {stock} and provide insights: \n\nPrice Analysis: {price_analysis}\n\nSentiment Analysis: {sentiment_analysis}")
    ])
    expert_output = (prompt | llm).invoke({"text": f"Provide expert analysis for {stock}"})
    print("Expert Analysis Output:", expert_output.content)
    return expert_output.content

def generate_trade_signal(stock, improved_output, price_analysis):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a financial advisor."),
        ("human", f"Based on the following analysis of {stock}, determine if it is a buy, sell, or hold signal. Sentiment Analysis: {improved_output}. Price Analysis: {price_analysis}")
    ])
    signal_output = (prompt | llm).invoke({"text": f"Generate a trade signal for {stock}"})
    print("Trade Signal Output:", signal_output.content)
    return signal_output.content

def get_weekly_stock_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date, interval='1wk')
        if data.empty:
            raise ValueError(f"No data found for {ticker}. The stock may be delisted or the symbol may be incorrect.")
        return data
    except Exception as e:
        logging.error(f"Failed to download data for {ticker}: {e}")
        return None

def perform_numerical_analysis(data, stock):
    plt.figure(figsize=(14, 7))
    plt.plot(data.index, data['Close'], label='Close Price')
    plt.title(f'Weekly Stock Prices for {stock}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    chart_filename = f'{stock}_weekly_chart.png'
    plt.savefig(chart_filename)
    plt.show()
    return chart_filename

def calculate_macd(data):
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

def calculate_bollinger_bands(data, window=20):
    rolling_mean = data['Close'].rolling(window).mean()
    rolling_std = data['Close'].rolling(window).std()
    upper_band = rolling_mean + (rolling_std * 2)
    lower_band = rolling_mean - (rolling_std * 2)
    return rolling_mean, upper_band, lower_band

def analyze_price_data(data, stock):
    if len(data) < 52:
        logging.warning(f"Not enough data to calculate Year-over-Year (YOY) return for {stock}. Data length: {len(data)}")
        yoy_return = "Not enough data"
    else:
        yoy_return = (data['Close'].iloc[-1] / data['Close'].iloc[-52] - 1) * 100

    avg_price = data['Close'].mean()
    std_dev_price = data['Close'].std()
    latest_price = data['Close'].iloc[-1]

    if len(data) >= 50:
        data['50_week_MA'] = data['Close'].rolling(window=50).mean()
        fifty_week_ma = f"${data['50_week_MA'].iloc[-1]:.2f}"
    else:
        fifty_week_ma = "Not enough data"

    if len(data) >= 200:
        data['200_week_MA'] = data['Close'].rolling(window=200).mean()
        two_hundred_week_ma = f"${data['200_week_MA'].iloc[-1]:.2f}"
    else:
        two_hundred_week_ma = "Not enough data"

    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    log_returns = np.log(data['Close'] / data['Close'].shift(1))
    volatility = log_returns.std() * np.sqrt(52)

    ytd_return = (data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) * 100

    macd, signal = calculate_macd(data)
    macd_latest = macd.iloc[-1]
    signal_latest = signal.iloc[-1]

    rolling_mean, upper_band, lower_band = calculate_bollinger_bands(data)
    upper_band_latest = upper_band.iloc[-1]
    lower_band_latest = lower_band.iloc[-1]
    price_analysis = (
        f"The average weekly closing price for {stock} over the past year is ${avg_price:.2f} "
        f"with a standard deviation of ${std_dev_price:.2f}. The latest closing price is ${latest_price:.2f}.\n\n"
        f"50-week moving average: {fifty_week_ma}\n"
        f"200-week moving average: {two_hundred_week_ma}\n"
        f"RSI (14-week): {rsi.iloc[-1]:.2f}\n"
        f"Annualized Volatility: {volatility:.2%}\n"
        f"Year-to-date (YTD) return: {ytd_return:.2f}%\n"
        f"Year-over-year (YOY) return: {yoy_return}\n"
        f"MACD: {macd_latest:.2f}\n"
        f"Signal Line: {signal_latest:.2f}\n"
        f"Bollinger Bands: Upper Band: ${upper_band_latest:.2f}, Lower Band: ${lower_band_latest:.2f}\n"
    )
    return price_analysis

# Summarize the collected content
summarized_content = summarize_content(combined_summary)

# Analyze the collected stock data
analysis_output = analyze_stock_data(stock_to_analyze, summarized_content)

# Review the analysis
review_output = review_analysis(stock_to_analyze, analysis_output)

# Improve the analysis based on the review feedback
improved_output = improve_analysis(stock_to_analyze, review_output, analysis_output)

# Perform numerical analysis
start_date = f"{int(current_date[:4])-5}-01-01"
end_date = current_date
weekly_data = get_weekly_stock_data(stock_ticker, start_date, end_date)

if weekly_data is not None:
    chart_filename = perform_numerical_analysis(weekly_data, stock_to_analyze)
    price_analysis = analyze_price_data(weekly_data, stock_to_analyze)

    # Generate a trade signal based on the improved analysis, price analysis, and expert analysis
    expert_output = expert_analysis(stock_to_analyze, price_analysis, improved_output)
    trade_signal = generate_trade_signal(stock_to_analyze, improved_output, price_analysis)

    # Compile the final report
    final_report = f"""
    # Financial Report for {stock_to_analyze} - Week of {current_date}

    ## Collected Data:

    ## Initial Analysis Output:

    {analysis_output}

    ## Review Output:

    {review_output}

    ## Improved Analysis Output:

    {improved_output}

    ## Price Analysis:

    {price_analysis}

    ## Expert Analysis:

    {expert_output}

    ## Trade Signal:

    {trade_signal}

    ## Weekly Numerical Analysis:

    A weekly chart has been generated to analyze the price trends of {stock_to_analyze} over the past year.

    ![Weekly Chart](./{chart_filename})

    ## Sources:
    {data_summary}
    """

    # Save the complete result to a .txt file
    output_txt_filename = f"Financial_Report_{stock_to_analyze}_Week.txt"
    with open(output_txt_filename, "w") as file:
        file.write(final_report)

    # Save the complete result to a .md file
    output_md_filename = f"Financial_Report_{stock_to_analyze}_Week.md"
    with open(output_md_filename, "w") as file:
        file.write(final_report)

    print(f"Analysis complete. Results saved to {output_txt_filename} and {output_md_filename}")
else:
    print(f"Failed to retrieve stock data for {stock_ticker}. The report cannot be generated.")
