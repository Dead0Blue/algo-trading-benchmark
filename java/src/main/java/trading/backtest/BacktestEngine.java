package trading.backtest;

import tech.tablesaw.api.Table;
import tech.tablesaw.api.DoubleColumn;
import tech.tablesaw.api.IntColumn;

import org.jfree.chart.ChartUtils;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.plot.PlotOrientation;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

public class BacktestEngine {

    private static final double INITIAL_CAPITAL = 10000.0;
    private static final String RESULTS_DIR = "../results";

    public Map<String, Double> backtest(Table testData, int[] predictions, String ticker, String modelName) {
        System.out.println("--- Backtest " + modelName + " on " + ticker + " ---");
        
        DoubleColumn closeCol = testData.doubleColumn("close");
        int n = testData.rowCount();
        
        double[] strategyEquity = new double[n];
        double[] bhEquity = new double[n];
        
        strategyEquity[0] = INITIAL_CAPITAL;
        bhEquity[0] = INITIAL_CAPITAL;
        
        int numTrades = 0;
        int winTrades = 0;
        
        double peak = INITIAL_CAPITAL;
        double maxDrawdown = 0;
        
        for (int i = 1; i < n; i++) {
            // Return from yesterday close to today close
            double dailyReturn = (closeCol.get(i) - closeCol.get(i-1)) / closeCol.get(i-1);
            
            // Buy & Hold Equity
            bhEquity[i] = bhEquity[i-1] * (1 + dailyReturn);
            
            // Strategy Equity (we hold today if predicted 1 yesterday)
            int predYesterday = predictions[i-1];
            double strategyReturn = predYesterday == 1 ? dailyReturn : 0;
            strategyEquity[i] = strategyEquity[i-1] * (1 + strategyReturn);
            
            if (predYesterday == 1) {
                numTrades++;
                if (strategyReturn > 0) winTrades++;
            }
            
            if (strategyEquity[i] > peak) {
                peak = strategyEquity[i];
            } else {
                double dd = (strategyEquity[i] / peak) - 1.0;
                if (dd < maxDrawdown) {
                    maxDrawdown = dd;
                }
            }
        }
        
        double totalReturn = (strategyEquity[n-1] / INITIAL_CAPITAL) - 1.0;
        double bhReturn = (bhEquity[n-1] / INITIAL_CAPITAL) - 1.0;
        double winRate = numTrades > 0 ? (double) winTrades / numTrades : 0.0;
        
        System.out.printf("Strategy Return: %.2f%% (B&H: %.2f%%)%n", totalReturn * 100, bhReturn * 100);
        System.out.printf("Max Drawdown:    %.2f%%%n", maxDrawdown * 100);
        System.out.printf("Win Rate:        %.2f%%%n", winRate * 100);
        System.out.println("Number of Trades:" + numTrades);
        System.out.println();
        
        plotEquity(ticker, modelName, bhEquity, strategyEquity);
        
        Map<String, Double> metrics = new HashMap<>();
        metrics.put("total_return_pct", totalReturn * 100);
        metrics.put("buy_hold_return_pct", bhReturn * 100);
        metrics.put("max_drawdown_pct", maxDrawdown * 100);
        metrics.put("win_rate_pct", winRate * 100);
        metrics.put("num_trades", (double) numTrades);
        
        return metrics;
    }
    
    private void plotEquity(String ticker, String modelName, double[] bh, double[] strat) {
        XYSeries bhSeries = new XYSeries("Buy & Hold");
        XYSeries stratSeries = new XYSeries(modelName + " Strategy");
        
        for (int i=0; i<bh.length; i++) {
            bhSeries.add(i, bh[i]);
            stratSeries.add(i, strat[i]);
        }
        
        XYSeriesCollection dataset = new XYSeriesCollection();
        dataset.addSeries(bhSeries);
        dataset.addSeries(stratSeries);
        
        JFreeChart chart = ChartFactory.createXYLineChart(
            "Equity Curve - " + modelName + " on " + ticker,
            "Days",
            "Portfolio Value ($)",
            dataset,
            PlotOrientation.VERTICAL,
            true, true, false
        );
        
        XYPlot plot = chart.getXYPlot();
        XYLineAndShapeRenderer renderer = new XYLineAndShapeRenderer();
        renderer.setSeriesPaint(0, java.awt.Color.GRAY);
        renderer.setSeriesPaint(1, java.awt.Color.BLUE);
        renderer.setSeriesShapesVisible(0, false);
        renderer.setSeriesShapesVisible(1, false);
        plot.setRenderer(renderer);
        
        try {
            File dir = new File(RESULTS_DIR);
            if (!dir.exists()) dir.mkdirs();
            String path = RESULTS_DIR + "/" + ticker + "_" + modelName.replace(" ", "_").toLowerCase() + "_java_equity.png";
            ChartUtils.saveChartAsPNG(new File(path), chart, 1000, 600);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
