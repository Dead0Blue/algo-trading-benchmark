package trading;

import trading.data.DataFetcher;
import trading.features.FeatureEngine;
import trading.model.TradingModel;
import trading.backtest.BacktestEngine;

import tech.tablesaw.api.Table;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.File;
import java.util.HashMap;
import java.util.Map;

public class Main {
    private static final String[] TICKERS = {"AAPL", "MSFT", "TSLA"};
    
    public static void main(String[] args) {
        System.out.println("=== Java Algorithmic Trading System ===");
        
        DataFetcher dataFetcher = new DataFetcher();
        FeatureEngine featureEngine = new FeatureEngine();
        TradingModel modelEngine = new TradingModel();
        BacktestEngine backtestEngine = new BacktestEngine();
        
        long globalStart = System.currentTimeMillis();
        long tFetchTotal = 0;
        long tFeaturesTotal = 0;
        long tModelsTotal = 0;
        long tBacktestTotal = 0;
        
        Map<String, Map<String, Map<String, Double>>> allMetrics = new HashMap<>();
        
        for (String ticker : TICKERS) {
            try {
                // 1. Data Loading
                long start = System.currentTimeMillis();
                Table rawData = dataFetcher.loadData(ticker);
                tFetchTotal += (System.currentTimeMillis() - start);
                
                // 2. Feature Engineering
                long start2 = System.currentTimeMillis();
                Table dataWithFeatures = featureEngine.computeFeatures(rawData);
                System.out.println("Computed features. Rows: " + dataWithFeatures.rowCount());
                tFeaturesTotal += (System.currentTimeMillis() - start2);
                
                // Train/test split (80/20 chronological)
                int splitIdx = (int) (dataWithFeatures.rowCount() * 0.8);
                Table trainData = dataWithFeatures.inRange(0, splitIdx);
                Table testData = dataWithFeatures.inRange(splitIdx, dataWithFeatures.rowCount());
                
                // 3. Models
                start = System.currentTimeMillis();
                int[] predsRF = modelEngine.trainAndPredictRF(trainData, testData);
                int[] predsLSTM = modelEngine.trainAndPredictLSTM(trainData, testData);
                tModelsTotal += (System.currentTimeMillis() - start);
                
                // 4. Backtest
                start = System.currentTimeMillis();
                Map<String, Double> metricsRF = backtestEngine.backtest(testData, predsRF, ticker, "Random Forest");
                Map<String, Double> metricsLSTM = backtestEngine.backtest(testData, predsLSTM, ticker, "LSTM");
                tBacktestTotal += (System.currentTimeMillis() - start);
                
                // Store metrics
                Map<String, Map<String, Double>> tickerMetrics = new HashMap<>();
                tickerMetrics.put("Random_Forest", metricsRF);
                tickerMetrics.put("LSTM", metricsLSTM);
                allMetrics.put(ticker, tickerMetrics);
                
            } catch (Exception e) {
                System.err.println("Error processing " + ticker + ": " + e.getMessage());
                e.printStackTrace();
            }
        }
        
        long globalEnd = System.currentTimeMillis();
        
        System.out.println("=== Pipeline Complete ===");
        System.out.printf("Data loading: %.2fs%n", tFetchTotal / 1000.0);
        System.out.printf("Features:     %.2fs%n", tFeaturesTotal / 1000.0);
        System.out.printf("Models:       %.2fs%n", tModelsTotal / 1000.0);
        System.out.printf("Backtest:     %.2fs%n", tBacktestTotal / 1000.0);
        System.out.printf("Total Time:   %.2fs%n", (globalEnd - globalStart) / 1000.0);
        
        // Save timings and metrics mapping
        try {
            File resultsDir = new File("../results");
            if (!resultsDir.exists()) resultsDir.mkdirs();
            
            ObjectMapper mapper = new ObjectMapper();
            
            Map<String, Double> timings = new HashMap<>();
            timings.put("data_loading_s", tFetchTotal / 1000.0);
            timings.put("feature_engineering_s", tFeaturesTotal / 1000.0);
            timings.put("model_training_s", tModelsTotal / 1000.0);
            timings.put("backtesting_s", tBacktestTotal / 1000.0);
            timings.put("total_s", (globalEnd - globalStart) / 1000.0);
            
            mapper.writerWithDefaultPrettyPrinter().writeValue(new File(resultsDir, "java_timings.json"), timings);
            mapper.writerWithDefaultPrettyPrinter().writeValue(new File(resultsDir, "java_backtest_metrics.json"), allMetrics);
            
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
