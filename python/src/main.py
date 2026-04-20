import time
import os

from data_fetcher import fetch_data
from features import main as compute_features
from model import main as train_models
from backtest import main as run_backtest

def main():
    print("=== Python Algorithmic Trading System ===")
    
    # Phase 1: Data fetching
    start_time = time.time()
    fetch_data()
    t_fetch = time.time() - start_time
    print(f"Data fetching took {t_fetch:.2f} seconds\n")
    
    # Phase 2: Features
    start_time = time.time()
    compute_features()
    t_features = time.time() - start_time
    print(f"Feature computation took {t_features:.2f} seconds\n")
    
    # Phase 3: Model
    start_time = time.time()
    train_models()
    t_models = time.time() - start_time
    print(f"Model training & prediction took {t_models:.2f} seconds\n")
    
    # Phase 4: Backtest
    start_time = time.time()
    run_backtest()
    t_backtest = time.time() - start_time
    print(f"Backtesting took {t_backtest:.2f} seconds\n")
    
    # Save timings
    timings = {
        'data_loading_s': t_fetch,
        'feature_engineering_s': t_features,
        'model_training_s': t_models,
        'backtesting_s': t_backtest,
        'total_s': t_fetch + t_features + t_models + t_backtest
    }
    
    os.makedirs('../results', exist_ok=True)
    import json
    with open('../results/python_timings.json', 'w') as f:
        json.dump(timings, f, indent=4)
        
    print("=== Pipeline Complete ===")
    print(f"Total execution time: {timings['total_s']:.2f} seconds")

if __name__ == '__main__':
    main()
