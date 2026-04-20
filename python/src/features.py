import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange
import os

DATA_DIR = '../data'
TICKERS = ['AAPL', 'MSFT', 'TSLA']

def compute_features(df):
    """Computes technical indicators and target label for a given DataFrame."""
    df = df.copy()
    
    # Needs to be sorted by date if 'Date' is index, our CSVs should have date from yf
    
    # RSI
    rsi_indicator = RSIIndicator(close=df['close'], window=14)
    df['rsi_14'] = rsi_indicator.rsi()
    
    # MACD
    macd_indicator = MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
    df['macd'] = macd_indicator.macd()
    df['macd_signal'] = macd_indicator.macd_signal()
    df['macd_diff'] = macd_indicator.macd_diff()
    
    # Bollinger Bands
    bb_indicator = BollingerBands(close=df['close'], window=20, window_dev=2)
    df['bb_bbm'] = bb_indicator.bollinger_mavg()
    df['bb_bbh'] = bb_indicator.bollinger_hband()
    df['bb_bbl'] = bb_indicator.bollinger_lband()
    df['bb_bbhi'] = bb_indicator.bollinger_hband_indicator()
    df['bb_bbli'] = bb_indicator.bollinger_lband_indicator()
    
    # EMA
    ema_10 = EMAIndicator(close=df['close'], window=10)
    ema_50 = EMAIndicator(close=df['close'], window=50)
    df['ema_10'] = ema_10.ema_indicator()
    df['ema_50'] = ema_50.ema_indicator()
    
    # ATR
    atr_indicator = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14)
    df['atr'] = atr_indicator.average_true_range()
    
    # Volume Change %
    df['volume_change_pct'] = df['volume'].pct_change()
    
    # Target label: 1 if next-day close > today's close, else 0
    df['next_close'] = df['close'].shift(-1)
    df['target'] = (df['next_close'] > df['close']).astype(int)
    
    # Drop rows with NaN values (due to shifting/indicators)
    df.dropna(inplace=True)
    
    return df

def main():
    for ticker in TICKERS:
        filepath = os.path.join(DATA_DIR, f'{ticker}.csv')
        if not os.path.exists(filepath):
            print(f'File {filepath} not found. Did you run data_fetcher.py?')
            continue
            
        print(f'Computing features for {ticker}...')
        # `yfinance` saves 'Date' as an index column name, usually
        df = pd.read_csv(filepath, index_col='Date', parse_dates=True)
        
        df_featured = compute_features(df)
        
        out_filepath = os.path.join(DATA_DIR, f'{ticker}_features.csv')
        df_featured.to_csv(out_filepath)
        print(f'Saved features to {out_filepath}')

if __name__ == '__main__':
    main()
