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
        double relativeReturn = totalReturn - bhReturn;
        double winRate = numTrades > 0 ? (double) winTrades / numTrades : 0.0;
        
        Map<String, Double> metrics = new HashMap<>();
        
        // Sharpe & Sortino
        double[] dailyReturns = new double[n-1];
        int numDownside = 0;
        double sumReturns = 0;
        for(int i=0; i<n-1; i++) {
            double r = (strategyEquity[i+1] - strategyEquity[i]) / strategyEquity[i];
            dailyReturns[i] = r;
            sumReturns += r;
            if (r < 0) numDownside++;
        }
        double meanReturn = sumReturns / (n-1);
        double sumSq = 0;
        for(double r : dailyReturns) sumSq += Math.pow(r - meanReturn, 2);
        double stdReturn = Math.sqrt(sumSq / (n-1));
        double sharpe = stdReturn != 0 ? Math.sqrt(252) * meanReturn / stdReturn : 0;
        
        double sumSqDownside = 0;
        if (numDownside > 1) {
            for(double r : dailyReturns) if(r < 0) sumSqDownside += Math.pow(r, 2); // Sortino uses downside dev from 0 or mean
            double stdDownside = Math.sqrt(sumSqDownside / numDownside);
            double sortino = stdDownside != 0 ? Math.sqrt(252) * meanReturn / stdDownside : 0;
            metrics.put("sortino_ratio", sortino);
        } else {
            metrics.put("sortino_ratio", 0.0);
        }

        System.out.printf("Strategy Return: %.2f%%%n", totalReturn * 100);
        System.out.printf("B&H Return:      %.2f%%%n", bhReturn * 100);
        System.out.printf("vs Baseline:     %+.2f%%%n", relativeReturn * 100);
        System.out.printf("Sharpe Ratio:    %.2f%n", sharpe);
        System.out.printf("Sortino Ratio:   %.2f%n", metrics.get("sortino_ratio"));
        System.out.printf("Max Drawdown:    %.2f%%%n", maxDrawdown * 100);
        System.out.printf("Win Rate:        %.2f%%%n", winRate * 100);
        System.out.println("Number of Trades:" + numTrades);
        System.out.println();
        
        plotEquity(ticker, modelName, bhEquity, strategyEquity);
        
        metrics.put("total_return_pct", totalReturn * 100);
        metrics.put("buy_hold_return_pct", bhReturn * 100);
        metrics.put("relative_return_pct", relativeReturn * 100);
        metrics.put("sharpe_ratio", sharpe);
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
