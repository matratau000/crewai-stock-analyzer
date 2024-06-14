# crewai-stock-analyzer

# Financial Stock Analysis Tool

This Python script provides a comprehensive financial analysis of a specified stock. It leverages various APIs and libraries to collect, analyze, and summarize stock sentiment and price data. The tool generates detailed financial reports, including sentiment analysis, price analysis, expert insights, and trade signals.

## Features

- Collects and summarizes stock sentiment data from various sources.
- Analyzes historical stock price data using technical indicators.
- Provides a detailed financial report with sentiment analysis, price analysis, and trade signals.
- Supports both OpenAI and Groq language models for natural language processing tasks.

## Requirements

- Python 3.7+
- Libraries:
  - `os`
  - `datetime`
  - `yfinance`
  - `matplotlib`
  - `crewai_tools`
  - `langchain_groq`
  - `langchain_openai`
  - `langchain_core`
  - `dotenv`
  - `logging`
  - `numpy`
  - `concurrent.futures`

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   - Create a `.env` file in the root directory.
   - Add your API keys for OpenAI, Groq, and SerperDevTool:
     ```env
     OPENAI_API_KEY=your_openai_api_key
     GROQ_API_KEY=your_groq_api_key
     SERPER_API_KEY=your_serper_api_key
     ```

## Usage

1. Run the script:
   ```bash
   python financial_stock_analysis.py
   ```

2. Follow the prompts to choose a language model and enter the stock details:
   - Choose LLM (groq/openai)
   - Enter the stock to analyze
   - Enter the stock ticker symbol
   - Enter the current date (YYYY-MM-DD)

3. The script will perform the following tasks:
   - Collect and summarize stock sentiment data.
   - Analyze historical stock price data.
   - Generate a detailed financial report.
   - Save the report to a `.txt` and `.md` file.

## Output

The script generates a comprehensive financial report saved in both `.txt` and `.md` formats. The report includes:

- Initial sentiment analysis output
- Review and improved analysis
- Detailed price analysis with technical indicators
- Expert analysis insights
- Trade signal recommendations
- Weekly numerical analysis with charts

## Example Report

Here is a sample of the generated report structure:

```
# Financial Report for [Stock] - Week of [Date]

## Collected Data:

## Initial Analysis Output:

[Analysis Output]

## Review Output:

[Review Output]

## Improved Analysis Output:

[Improved Output]

## Price Analysis:

[Price Analysis]

## Expert Analysis:

[Expert Analysis]

## Trade Signal:

[Trade Signal]

## Weekly Numerical Analysis:

A weekly chart has been generated to analyze the price trends of [Stock] over the past year.

![Weekly Chart](./[chart_filename])

## Sources:
[Data Summary]
```

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss any changes.

## License

This project is licensed under the MIT License.

## Additional Files

### markdown_to_pdf.py

This script converts markdown files to PDF format. It includes the following function:
- `extract_sources(md_content)`: Extracts sources from markdown content.

### production.py

This script contains various functions for stock data analysis and summarization. It includes the following functions:
- `collect_stock_data(stock, date)`: Collects stock data for a given stock and date.
- `summarize_content(content)`: Summarizes the given content.
- `scrape_and_summarize_link(link)`: Scrapes and summarizes content from a given link.
- `analyze_stock_data(stock, data_summary)`: Analyzes stock data based on the data summary.
- `review_analysis(stock, analysis_output)`: Reviews the analysis output for a given stock.
- `improve_analysis(stock, review_output, analysis_output)`: Improves the analysis based on the review output.
- `expert_analysis(stock, price_analysis, sentiment_analysis)`: Provides expert analysis based on price and sentiment analysis.
- `generate_trade_signal(stock, improved_output, price_analysis)`: Generates trade signals based on the improved output and price analysis.
- `get_weekly_stock_data(ticker, start_date, end_date)`: Retrieves weekly stock data for a given ticker and date range.
- `perform_numerical_analysis(data, stock)`: Performs numerical analysis on the given data and stock.
- `calculate_macd(data)`: Calculates the MACD (Moving Average Convergence Divergence) for the given data.
- `calculate_bollinger_bands(data, window=20)`: Calculates the Bollinger Bands for the given data and window.
- `analyze_price_data(data, stock)`: Analyzes the price data for a given stock.
