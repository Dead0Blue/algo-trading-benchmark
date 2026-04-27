package trading.model;

import smile.classification.RandomForest;
import java.util.Properties;
import smile.data.DataFrame;
import tech.tablesaw.api.Table;
import tech.tablesaw.api.IntColumn;
import tech.tablesaw.api.DoubleColumn;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class TradingModel {
    private RandomForest rf;

    public double[][] trainAndPredictRF(Table trainData, Table testData) {
        System.out.println("Training Random Forest...");

        double[][] xTrain = extractFeatures(trainData);
        int[] yTrain = extractTarget(trainData);
        
        Properties props = new Properties();
        props.setProperty("smile.random.forest.trees", "100");
        
        long startTrain = System.nanoTime();
        this.rf = RandomForest.fit(smile.data.formula.Formula.lhs("target"), toSmileDataFrame(trainData, xTrain, yTrain), props);
        double trainTime = (System.nanoTime() - startTrain) / 1e9;

        System.out.println("Predicting with Random Forest...");
        double[][] xTest = extractFeatures(testData);
        
        long startInfer = System.nanoTime();
        int[] yPred = rf.predict(toSmileDataFrame(testData, xTest, new int[testData.rowCount()]));
        double inferTime = (System.nanoTime() - startInfer) / 1e9;
        
        // Return results as a double array [yPred, trainTime, inferTime]
        // Actually, let's return a wrapper or just print and return yPred.
        // To keep it simple for now, I'll print them here and the caller can capture.
        System.out.println("RF_TRAIN_TIME: " + trainTime);
        System.out.println("RF_INFER_TIME: " + inferTime);
        
        double[] results = new double[yPred.length];
        for(int i=0; i<yPred.length; i++) results[i] = yPred[i];
        return new double[][]{results, {trainTime, inferTime}};
    }
    
    public double[][] trainAndPredictLSTM(Table trainData, Table testData) {
        System.out.println("Training LSTM...");
        
        int seqLength = 10;
        int nIn = 13;
        int nOut = 2;
        
        double[][] xTrainRaw = extractFeatures(trainData);
        int[] yTrainRaw = extractTarget(trainData);
        double[][] xTestRaw = extractFeatures(testData);
        
        // Standardization
        StandardScaler scaler = new StandardScaler();
        scaler.fit(xTrainRaw);
        double[][] xTrainScaled = scaler.transform(xTrainRaw);
        double[][] xTestScaled = scaler.transform(xTestRaw);
        
        int numTrainExamples = xTrainScaled.length - seqLength;
        INDArray trainFeatures = Nd4j.create(numTrainExamples, nIn, seqLength);
        INDArray trainLabels = Nd4j.create(numTrainExamples, nOut, seqLength);
        INDArray trainMask = Nd4j.zeros(numTrainExamples, seqLength);
        
        for (int i=0; i<numTrainExamples; i++) {
            for (int t=0; t<seqLength; t++) {
                for (int j=0; j<nIn; j++) {
                    trainFeatures.putScalar(new int[]{i, j, t}, xTrainScaled[i+t][j]);
                }
            }
            // Many-to-one: only provide label at the last time step and mask others
            int lastStep = seqLength - 1;
            trainLabels.putScalar(new int[]{i, 0, lastStep}, yTrainRaw[i+seqLength] == 0 ? 1.0 : 0.0);
            trainLabels.putScalar(new int[]{i, 1, lastStep}, yTrainRaw[i+seqLength] == 1 ? 1.0 : 0.0);
            trainMask.putScalar(new int[]{i, lastStep}, 1.0);
        }
        
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(123)
            .updater(new Adam(0.01))
            .list()
            .layer(new LSTM.Builder().nIn(nIn).nOut(50).activation(Activation.TANH).build())
            .layer(new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                    .activation(Activation.SOFTMAX).nIn(50).nOut(nOut).build())
            .build();
            
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        
        long startTrain = System.nanoTime();
        net.fit(new DataSet(trainFeatures, trainLabels, null, trainMask));
        double trainTime = (System.nanoTime() - startTrain) / 1e9;
        
        System.out.println("Predicting with LSTM...");
        int numTestExamples = xTestScaled.length - seqLength;
        INDArray testFeatures = Nd4j.create(numTestExamples, nIn, seqLength);
        
        for (int i=0; i<numTestExamples; i++) {
            for (int t=0; t<seqLength; t++) {
                for (int j=0; j<nIn; j++) {
                    testFeatures.putScalar(new int[]{i, j, t}, xTestScaled[i+t][j]);
                }
            }
        }
        
        long startInfer = System.nanoTime();
        INDArray output = net.output(testFeatures);
        double inferTime = (System.nanoTime() - startInfer) / 1e9;
        
        System.out.println("LSTM_TRAIN_TIME: " + trainTime);
        System.out.println("LSTM_INFER_TIME: " + inferTime);
        
        double[] yPred = new double[testData.rowCount()];
        for(int i=0; i<numTestExamples; i++) {
            double p1 = output.getDouble(i, 1, seqLength - 1);
            yPred[i + seqLength] = p1 > 0.5 ? 1.0 : 0.0;
        }
        
        return new double[][]{yPred, {trainTime, inferTime}};
    }

    private static class StandardScaler {
        private double[] mean;
        private double[] std;

        public void fit(double[][] data) {
            int n = data.length;
            int cols = data[0].length;
            mean = new double[cols];
            std = new double[cols];
            for (double[] row : data) {
                for (int j = 0; j < cols; j++) mean[j] += row[j];
            }
            for (int j = 0; j < cols; j++) mean[j] /= n;
            for (double[] row : data) {
                for (int j = 0; j < cols; j++) std[j] += Math.pow(row[j] - mean[j], 2);
            }
            for (int j = 0; j < cols; j++) std[j] = Math.sqrt(std[j] / n);
        }

        public double[][] transform(double[][] data) {
            double[][] scaled = new double[data.length][data[0].length];
            for (int i = 0; i < data.length; i++) {
                for (int j = 0; j < data[0].length; j++) {
                    scaled[i][j] = (std[j] == 0) ? 0 : (data[i][j] - mean[j]) / std[j];
                }
            }
            return scaled;
        }
    }
    
    private DataFrame toSmileDataFrame(Table t, double[][] x, int[] y) {
        String[] columns = {"rsi_14", "ema_10", "ema_50", "macd", "macd_signal", "macd_diff", "bb_bbm", "bb_bbh", "bb_bbl", "bb_bbhi", "bb_bbli", "atr", "volume_change_pct"};
        DataFrame df = DataFrame.of(x, columns);
        return df.merge(smile.data.vector.IntVector.of("target", y));
    }
    
    private double[][] extractFeatures(Table t) {
        String[] featureCols = {
            "rsi_14", "ema_10", "ema_50", "macd", "macd_signal", "macd_diff", 
            "bb_bbm", "bb_bbh", "bb_bbl", "bb_bbhi", "bb_bbli", "atr", "volume_change_pct"
        };
        double[][] rows = new double[t.rowCount()][featureCols.length];
        for (int i = 0; i < t.rowCount(); i++) {
            for (int c = 0; c < featureCols.length; c++) {
                rows[i][c] = t.doubleColumn(featureCols[c]).get(i);
            }
        }
        return rows;
    }

    private int[] extractTarget(Table t) {
        IntColumn targetCol = t.intColumn("target");
        int[] tgt = new int[t.rowCount()];
        for (int i = 0; i < t.rowCount(); i++) {
            tgt[i] = targetCol.get(i);
        }
        return tgt;
    }
}
