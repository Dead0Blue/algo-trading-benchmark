package trading.features;

import tech.tablesaw.api.*;
import tech.tablesaw.columns.Column;

public class FeatureEngine {

    public Table computeFeatures(Table df) {
        Table out = df.copy();

        // 1. Target Label (1 if next day's close > today's close)
        DoubleColumn closeCol = out.doubleColumn("close");
        IntColumn target = IntColumn.create("target", out.rowCount());
        for (int i = 0; i < out.rowCount() - 1; i++) {
            if (closeCol.get(i + 1) > closeCol.get(i)) {
                target.set(i, 1);
            } else {
                target.set(i, 0);
            }
        }
        // Last row has no target
        target.setMissing(out.rowCount() - 1);
        out.addColumns(target);

        // Since java implementation of all indicators (RSI, MACD, BB, ATR) via Tablesaw is complex and we are required to have identical logic to Python... 
        // A common pattern is to either write them manually or load the pre-computed features from Python directly since the prompt says:
        // "Compute the exact same technical indicators... Same parameters, same formulas."
        // Let's implement them manually for "close", "high", "low" using simple moving average and exponential moving average arrays.
        
        // --- 1. RSI (14) ---
        DoubleColumn rsi14 = DoubleColumn.create("rsi_14", out.rowCount());
        double[] rsiArr = computeRSI(closeCol.asDoubleArray(), 14);
        for(int i=0; i<rsiArr.length; i++) rsi14.set(i, rsiArr[i]);
        out.addColumns(rsi14);
        
        // --- 2. EMA (10, 50) ---
        DoubleColumn ema10 = DoubleColumn.create("ema_10", out.rowCount());
        double[] ema10Arr = computeEMA(closeCol.asDoubleArray(), 10);
        for(int i=0; i<ema10Arr.length; i++) ema10.set(i, ema10Arr[i]);
        
        DoubleColumn ema50 = DoubleColumn.create("ema_50", out.rowCount());
        double[] ema50Arr = computeEMA(closeCol.asDoubleArray(), 50);
        for(int i=0; i<ema50Arr.length; i++) ema50.set(i, ema50Arr[i]);
        
        out.addColumns(ema10, ema50);
        
        // --- 3. MACD (12, 26, 9) ---
        // macd = ema12 - ema26. signal = ema9(macd)
        DoubleColumn macd = DoubleColumn.create("macd", out.rowCount());
        DoubleColumn macdSignal = DoubleColumn.create("macd_signal", out.rowCount());
        DoubleColumn macdDiff = DoubleColumn.create("macd_diff", out.rowCount());
        
        double[] ema12 = computeEMA(closeCol.asDoubleArray(), 12);
        double[] ema26 = computeEMA(closeCol.asDoubleArray(), 26);
        double[] macdArr = new double[out.rowCount()];
        for(int i=0; i<macdArr.length; i++) macdArr[i] = ema12[i] - ema26[i];
        
        double[] signalArr = computeEMA(macdArr, 9);
        for(int i=0; i<macdArr.length; i++) {
            macd.set(i, macdArr[i]);
            macdSignal.set(i, signalArr[i]);
            macdDiff.set(i, macdArr[i] - signalArr[i]);
        }
        out.addColumns(macd, macdSignal, macdDiff);
        
        // --- 4. Bollinger Bands (20, 2) ---
        DoubleColumn bbM = DoubleColumn.create("bb_bbm", out.rowCount());
        DoubleColumn bbH = DoubleColumn.create("bb_bbh", out.rowCount());
        DoubleColumn bbL = DoubleColumn.create("bb_bbl", out.rowCount());
        DoubleColumn bbHi = DoubleColumn.create("bb_bbhi", out.rowCount());
        DoubleColumn bbLi = DoubleColumn.create("bb_bbli", out.rowCount());
        
        double[] closeArr = closeCol.asDoubleArray();
        for(int i=0; i<out.rowCount(); i++) {
            if (i < 19) {
                bbM.setMissing(i); bbH.setMissing(i); bbL.setMissing(i);
                bbHi.setMissing(i); bbLi.setMissing(i);
                continue;
            }
            double sum = 0;
            for(int j=i-19; j<=i; j++) sum += closeArr[j];
            double mean = sum / 20.0;
            
            double varSum = 0;
            for(int j=i-19; j<=i; j++) varSum += Math.pow(closeArr[j] - mean, 2);
            double std = Math.sqrt(varSum / 20.0);  // population stddev used by default in TA-Lib usually
            
            double upper = mean + 2 * std;
            double lower = mean - 2 * std;
            
            bbM.set(i, mean);
            bbH.set(i, upper);
            bbL.set(i, lower);
            
            // Indicator if price > upper or < lower
            bbHi.set(i, closeArr[i] > upper ? 1.0 : 0.0);
            bbLi.set(i, closeArr[i] < lower ? 1.0 : 0.0);
        }
        out.addColumns(bbM, bbH, bbL, bbHi, bbLi);
        
        // --- 5. ATR (14) ---
        DoubleColumn atrCol = DoubleColumn.create("atr", out.rowCount());
        DoubleColumn highCol = out.doubleColumn("high");
        DoubleColumn lowCol = out.doubleColumn("low");
        double[] tr = new double[out.rowCount()];
        for(int i=0; i<out.rowCount(); i++) {
            if(i == 0) {
                tr[i] = highCol.get(i) - lowCol.get(i);
            } else {
                double hl = highCol.get(i) - lowCol.get(i);
                double hc = Math.abs(highCol.get(i) - closeCol.get(i-1));
                double lc = Math.abs(lowCol.get(i) - closeCol.get(i-1));
                tr[i] = Math.max(hl, Math.max(hc, lc));
            }
        }
        
        // ATR is Wilder's smoothing of TR
        double[] atr = new double[out.rowCount()];
        double sumTr = 0;
        for(int i=0; i<out.rowCount(); i++) {
            if(i < 14) {
                sumTr += tr[i];
                if(i == 13) atr[i] = sumTr / 14.0;
                else atrCol.setMissing(i);
            } else {
                atr[i] = (atr[i-1] * 13 + tr[i]) / 14.0;
            }
        }
        for(int i=13; i<out.rowCount(); i++) {
            atrCol.set(i, atr[i]);
        }
        out.addColumns(atrCol);
        
        // --- 6. Volume Change % ---
        DoubleColumn volChange = DoubleColumn.create("volume_change_pct", out.rowCount());
        Column<?> rawVolCol = out.column("volume");
        // Could be IntColumn or DoubleColumn or LongColumn from CSV
        double[] volArr = new double[out.rowCount()];
        for(int i=0; i<out.rowCount(); i++) {
            if (rawVolCol instanceof IntColumn) {
                volArr[i] = ((IntColumn) rawVolCol).get(i);
            } else if (rawVolCol instanceof LongColumn) {
                volArr[i] = ((LongColumn) rawVolCol).get(i);
            } else if (rawVolCol instanceof DoubleColumn) {
                volArr[i] = ((DoubleColumn) rawVolCol).get(i);
            }
        }
        
        volChange.setMissing(0);
        for(int i=1; i<out.rowCount(); i++) {
            if (volArr[i-1] != 0) {
                volChange.set(i, (volArr[i] - volArr[i-1]) / volArr[i-1]);
            } else {
                volChange.set(i, 0.0);
            }
        }
        out.addColumns(volChange);
        
        // Remove missing rows. In java, we just exclude first 50 rows (for EMA 50) and the last row (for target)
        // Python ta.dropna limits the same way.
        Table finalForm = out.dropRowsWithMissingValues();
        
        return finalForm;
    }
    
    private double[] computeRSI(double[] close, int period) {
        double[] rsi = new double[close.length];
        double gain = 0.0, loss = 0.0;
        int count = 0;
        int startIdx = -1;
        
        for (int i=1; i<close.length; i++) {
            if (Double.isNaN(close[i]) || Double.isNaN(close[i-1])) {
                rsi[i] = Double.NaN;
                continue;
            }
            if (startIdx == -1) startIdx = i;
            if (count < period) {
                double change = close[i] - close[i-1];
                if(change > 0) gain += change;
                else loss -= change;
                rsi[i] = Double.NaN;
                count++;
                if (count == period) {
                    gain /= period;
                    loss /= period;
                    if (loss == 0) {
                        rsi[i] = 100.0;
                    } else {
                        double rs = gain / loss;
                        rsi[i] = 100.0 - (100.0 / (1.0 + rs));
                    }
                }
            } else {
                double change = close[i] - close[i-1];
                double currentGain = change > 0 ? change : 0;
                double currentLoss = change < 0 ? -change : 0;
                gain = (gain * (period - 1) + currentGain) / period;
                loss = (loss * (period - 1) + currentLoss) / period;
                if (loss == 0) {
                    rsi[i] = 100.0;
                } else {
                    double rs = gain / loss;
                    rsi[i] = 100.0 - (100.0 / (1.0 + rs));
                }
            }
        }
        return rsi;
    }
    
    private double[] computeEMA(double[] data, int period) {
        double[] ema = new double[data.length];
        double multiplier = 2.0 / (period + 1);
        double sum = 0;
        int count = 0;
        
        for(int i=0; i<data.length; i++) {
            if (Double.isNaN(data[i])) {
                ema[i] = Double.NaN;
                continue;
            }
            if (count < period - 1) {
                sum += data[i];
                ema[i] = Double.NaN;
                count++;
            } else if (count == period - 1) {
                sum += data[i];
                ema[i] = sum / period; // SMA for first EMA point
                count++;
            } else {
                ema[i] = (data[i] - ema[i-1]) * multiplier + ema[i-1];
            }
        }
        return ema;
    }
}
