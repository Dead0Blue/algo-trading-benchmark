import yfinance as yf
import pandas as pd
import os

# Create data directory if it doesn't exist
os.makedirs('../data', exist_ok=True)

TICKERS = ['AAPL', 'MSFT', 'TSLA']
START_DATE = '2019-01-01'
END_DATE = '2024-01-01'

def fetch_data():
    for ticker in TICKERS:
        print(f'Fetching data for {ticker}...')
        data = yf.download(ticker, start=START_DATE, end=END_DATE)
        
        # Ensure we have data
        if data.empty:
            print(f'Failed to fetch data for {ticker}')
            continue
            
        # Clean columns if yf returns multi-index
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel('Ticker')
            data.columns = [col.lower() for col in data.columns]
        else:
            data.columns = [col.lower() for col in data.columns]
        
        # Make sure we have the required columns
        expected_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in expected_cols):
            print(f'Missing columns for {ticker}. Expected {expected_cols}, got {list(data.columns)}')
            continue
            
        # Save to csv
        filepath = f'../data/{ticker}.csv'
        data.to_csv(filepath)
        print(f'Saved {ticker} to {filepath}')

if __name__ == '__main__':
    fetch_data()
