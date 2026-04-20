import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json

DATA_DIR = '../data'
RESULTS_DIR = '../results'
TICKERS = ['AAPL', 'MSFT', 'TSLA']
INITIAL_CAPITAL = 10000.0

def calculate_drawdown(equity_curve):
    peak = equity_curve.expanding(min_periods=1).max()
    drawdown = (equity_curve / peak) - 1.0
    return drawdown.min()

def backtest_strategy(df, pred_col, ticker, model_name):
    # Assume we buy at open tomorrow if prediction is 1, and sell at close tomorrow.
    # A simpler assumption: Daily returns are close-to-close. 
    # If pred==1 (today), we hold the asset for tomorrow. Meaning we get the return from today's close to tomorrow's close.
    # Let's compute daily return
    df['daily_return'] = df['close'].pct_change()
    
    # We shift returns by -1 because if we predict 1 today, we earn tomorrow's return
    df['next_day_return'] = df['daily_return'].shift(-1)
    
    # Strategy returns
    # We hold if pred = 1, cash if pred = 0
    df['strategy_return'] = df[pred_col] * df['next_day_return']
    
    # Drop NAs
    df_clean = df.dropna(subset=['next_day_return', pred_col]).copy()
    
    if len(df_clean) == 0:
        return None
        
    # Equity curve
    df_clean['strategy_equity'] = INITIAL_CAPITAL * (1 + df_clean['strategy_return']).cumprod()
    df_clean['buy_hold_equity'] = INITIAL_CAPITAL * (1 + df_clean['next_day_return']).cumprod()
    
    # Metrics
    total_return = (df_clean['strategy_equity'].iloc[-1] / INITIAL_CAPITAL) - 1.0
    bh_total_return = (df_clean['buy_hold_equity'].iloc[-1] / INITIAL_CAPITAL) - 1.0
    
    # Annualized Sharpe (assuming 252 trading days)
    # Risk free rate = 0 for simplicity
    if df_clean['strategy_return'].std() != 0:
        sharpe = np.sqrt(252) * df_clean['strategy_return'].mean() / df_clean['strategy_return'].std()
    else:
        sharpe = 0.0
        
    max_dd = calculate_drawdown(df_clean['strategy_equity'])
    
    # Trades and Win Rate
    # A trade occurs when we are invested (pred=1)
    # Win rate: % of days where strategy_return > 0 given pred=1
    trades = df_clean[df_clean[pred_col] == 1]
    num_trades = len(trades)
    win_rate = len(trades[trades['strategy_return'] > 0]) / num_trades if num_trades > 0 else 0
    
    metrics = {
        'total_return_pct': total_return * 100,
        'buy_hold_return_pct': bh_total_return * 100,
        'sharpe_ratio': sharpe,
        'max_drawdown_pct': max_dd * 100,
        'win_rate_pct': win_rate * 100,
        'num_trades': num_trades
    }
    
    print(f"--- Backtest {model_name} on {ticker} ---")
    print(f"Strategy Return: {metrics['total_return_pct']:.2f}% (B&H: {metrics['buy_hold_return_pct']:.2f}%)")
    print(f"Sharpe Ratio:    {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown:    {metrics['max_drawdown_pct']:.2f}%")
    print(f"Win Rate:        {metrics['win_rate_pct']:.2f}%")
    print(f"Number of Trades:{metrics['num_trades']}")
    print()
    
    return metrics, df_clean

def plot_equity_curve(df_clean, ticker, model_name):
    plt.figure(figsize=(10, 6))
    plt.plot(df_clean.index, df_clean['buy_hold_equity'], label='Buy & Hold', color='gray', alpha=0.7)
    plt.plot(df_clean.index, df_clean['strategy_equity'], label=f'{model_name} Strategy', color='blue')
    plt.title(f'Equity Curve - {model_name} on {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    chart_path = os.path.join(RESULTS_DIR, f'{ticker}_{model_name.replace(" ", "_").lower()}_equity.png')
    plt.savefig(chart_path)
    plt.close()

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    backtest_metrics = {}
    
    for ticker in TICKERS:
        filepath = os.path.join(DATA_DIR, f'{ticker}_predictions.csv')
        if not os.path.exists(filepath):
            continue
            
        df = pd.read_csv(filepath, index_col='Date', parse_dates=True)
        
        backtest_metrics[ticker] = {}
        
        # Random Forest
        metrics_rf, df_clean_rf = backtest_strategy(df, 'pred_rf', ticker, "Random Forest")
        if df_clean_rf is not None:
            plot_equity_curve(df_clean_rf, ticker, "Random Forest")
            backtest_metrics[ticker]['Random_Forest'] = metrics_rf
            
        # LSTM
        metrics_lstm, df_clean_lstm = backtest_strategy(df, 'pred_lstm', ticker, "LSTM")
        if df_clean_lstm is not None:
            plot_equity_curve(df_clean_lstm, ticker, "LSTM")
            backtest_metrics[ticker]['LSTM'] = metrics_lstm
            
    # Save backtest metrics
    with open(os.path.join(RESULTS_DIR, 'python_backtest_metrics.json'), 'w') as f:
        json.dump(backtest_metrics, f, indent=4)

if __name__ == '__main__':
    main()
