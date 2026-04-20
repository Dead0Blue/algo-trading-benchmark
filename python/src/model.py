import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler
import os
import json

DATA_DIR = '../data'
RESULTS_DIR = '../results'
TICKERS = ['AAPL', 'MSFT', 'TSLA']

def evaluate_predictions(y_true, y_pred, model_name, ticker):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()
    
    metrics = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'confusion_matrix': cm
    }
    
    print(f"--- {model_name} on {ticker} ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Conf Matrix: {cm}")
    print()
    
    return metrics

def train_evaluate_rf(X_train, X_test, y_train, y_test, ticker):
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_pred_proba = rf.predict_proba(X_test)[:, 1]
    
    metrics = evaluate_predictions(y_test, y_pred, "Random Forest", ticker)
    
    # Return predictions for backtesting
    return y_pred, y_pred_proba, metrics

def train_evaluate_lstm(X_train, X_test, y_train, y_test, ticker):
    # LSTM needs sequential data. We'll use a sequence length of 10.
    seq_length = 10
    
    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Prepare sequences
    def create_sequences(X, y, seq_length):
        Xs, ys = [], []
        for i in range(len(X) - seq_length):
            Xs.append(X[i:(i + seq_length)])
            ys.append(y.iloc[i + seq_length] if isinstance(y, pd.Series) else y[i + seq_length])
        return np.array(Xs), np.array(ys)
    
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, seq_length)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, seq_length)
    
    # We need to trim the test dates in backtesting later by `seq_length`
    
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train
    model.fit(X_train_seq, y_train_seq, epochs=20, batch_size=32, validation_split=0.1, verbose=0)
    
    # Predict
    y_pred_proba = model.predict(X_test_seq).flatten()
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    metrics = evaluate_predictions(y_test_seq, y_pred, "LSTM", ticker)
    
    # To return predictions aligned with the test set (pads first `seq_length` with 0s)
    y_pred_padded = np.concatenate([np.zeros(seq_length, dtype=int), y_pred])
    y_pred_proba_padded = np.concatenate([np.zeros(seq_length), y_pred_proba])
    
    return y_pred_padded, y_pred_proba_padded, metrics

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    all_metrics = {}
    
    for ticker in TICKERS:
        filepath = os.path.join(DATA_DIR, f'{ticker}_features.csv')
        if not os.path.exists(filepath):
            continue
            
        df = pd.read_csv(filepath, index_col='Date', parse_dates=True)
        
        # Features & Target
        # drop raw price cols Except close for potential use, but let's just drop them all for strictly indicator-based
        # Actually, using prices with indicators can cause issues with scaling and stationarity if not careful, 
        # but the plan says "Compute technical indicators from raw OHLCV", let's use the indicators.
        target_col = 'target'
        cols_to_drop = ['open', 'high', 'low', 'close', 'volume', 'next_close', 'target']
        feature_cols = [c for c in df.columns if c not in cols_to_drop]
        
        X = df[feature_cols]
        y = df[target_col]
        
        # Chronological train/test split (80/20)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # We need this later for backtester
        df_test = df.iloc[split_idx:].copy()
        
        print(f"Training models for {ticker}...")
        
        # RF
        y_pred_rf, y_pred_proba_rf, metrics_rf = train_evaluate_rf(X_train, X_test, y_train, y_test, ticker)
        df_test['pred_rf'] = y_pred_rf
        
        # LSTM
        y_pred_lstm, y_pred_proba_lstm, metrics_lstm = train_evaluate_lstm(X_train, X_test, y_train, y_test, ticker)
        df_test['pred_lstm'] = y_pred_lstm
        
        all_metrics[ticker] = {
            'Random_Forest': metrics_rf,
            'LSTM': metrics_lstm
        }
        
        # Save predictions for backtester
        preds_path = os.path.join(DATA_DIR, f'{ticker}_predictions.csv')
        df_test.to_csv(preds_path)
        print(f"Saved test set predictions to {preds_path}")
        
    # Save metrics
    with open(os.path.join(RESULTS_DIR, 'python_metrics.json'), 'w') as f:
        json.dump(all_metrics, f, indent=4)

if __name__ == '__main__':
    main()
