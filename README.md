# Algorithmic Trading Benchmark: Python vs Java

This repository contains a full **Machine Learning Algorithmic Trading System** implemented simultaneously in both **Python** and **Java**. The project establishes a 1:1 behavioral baseline across both ecosystems to benchmark quantitative finance modeling speed, algorithmic feature engineering, and predictive trading outcomes.

## 🚀 Overview

The system independently operates within both languages entirely from scratch by:
1. **Fetching historical OHLCV data** (AAPL, MSFT, TSLA) spanning 5 years.
2. **Computing technical momentum indicators** (14-RSI, 10/50-EMA, MACD, Bollinger Bands, ATR, Volume Change).
3. **Training classification models** designed to predict next-day bullish price movements:
   - **Random Forest** classification
   - **LSTM (Long Short-Term Memory)** artificial neural networks
4. **Conducting a sequential backtest** on the trained models starting from a simulated $10,000 capital, rendering complete execution metrics and equity curve charts against a Buy-and-Hold baseline.

## 🛠️ Technology Stack

| Ecosystem | Data Manipulation | Feature Logic | Random Forest | LSTM Neural Net | Visualization |
|-----------|-------------------|---------------|---------------|-----------------|---------------|
| **Python**| `pandas` / `numpy`| `ta` (TA-Lib) | `scikit-learn`| `TensorFlow`    | `Matplotlib`  |
| **Java**  | `Tablesaw`        | Custom Arrays | `Smile 3.0`   | `Deeplearning4J`| `JFreeChart`  |

## 📊 Benchmark Results (TL;DR)

Extensive benchmarking recorded execution velocity and simulation metric differences heavily tied natively to how each compiled ecosystem interprets graph execution. 

* The JVM (Java) completely obliterated Python's execution bottleneck on identical models, logging end-to-end model training speeds up to **~4x faster**.
* Backtest accuracy varied directly based on library-level parameter differences inside random seeding and tree initializations, though both safely respected the exact same dataframes. 
* Python required far fewer lines of code natively compared to Java's verbose class strictness.

> Read the comprehensive comparison breakdowns here: [benchmark_results.md](/comparison/benchmark_results.md)

## 🗂️ Project Structure

```text
├── python/                     # Python logic implementation
│   ├── src/                    # Python trading framework
│   ├── data/                   # Downloaded raw and featured CSVs
│   └── requirements.txt        # Python pip dependencies
├── java/                       # Java logic implementation
│   ├── src/main/java/trading/  # Native Java quantitative framework 
│   └── pom.xml                 # Maven POM configuration
├── results/                    # Compiled JSON metrics and Equity Charts (.png)
└── comparison/                 # The Benchmark analysis outputs
```

## ⚙️ Running Locally

### Python Setup
Requires Python 3.10+
```bash
cd python
python -m venv venv
source venv/bin/activate  # (or .\venv\Scripts\activate on Windows)
pip install -r requirements.txt
python src/main.py
```

### Java Setup
Requires Java JDK 11+ and Maven.
```bash
cd java
mvn clean compile dependency:copy-dependencies
java -cp "target/classes;target/dependency/*" trading.Main
```

## 📈 Sample Results

*(Check the `/results` folder for the full suite of generated equity curve charting representing model vs model behavior over the 5-year timeline!)*

---
*Created as part of a high-performance quantitative pipeline experiment.*
