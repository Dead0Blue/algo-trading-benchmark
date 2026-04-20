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

    public int[] trainAndPredictRF(Table trainData, Table testData) {
        System.out.println("Training Random Forest...");

        // Tablesaw to 2D double array for Smile
        double[][] xTrain = extractFeatures(trainData);
        int[] yTrain = extractTarget(trainData);
        
        // Train Random Forest
        Properties props = new Properties();
        this.rf = RandomForest.fit(smile.data.formula.Formula.lhs("target"), toSmileDataFrame(trainData, xTrain, yTrain), props);

        System.out.println("Predicting with Random Forest...");
        double[][] xTest = extractFeatures(testData);
        int[] dummyTarget = new int[testData.rowCount()];
        String[] columns = {"rsi_14", "ema_10", "ema_50", "macd", "macd_signal", "macd_diff", "bb_bbm", "bb_bbh", "bb_bbl", "bb_bbhi", "bb_bbli", "atr", "volume_change_pct"};
        smile.data.DataFrame testDf = smile.data.DataFrame.of(xTest, columns).merge(smile.data.vector.IntVector.of("target", dummyTarget));
        int[] yPred = rf.predict(testDf);
        
        return yPred;
    }
    
    public int[] trainAndPredictLSTM(Table trainData, Table testData) {
        System.out.println("Training LSTM...");
        
        int seqLength = 10;
        int nIn = 13;
        int nOut = 2; // binary classification 0 or 1. If regression, nOut=1.
        
        double[][] xTrainRaw = extractFeatures(trainData);
        int[] yTrainRaw = extractTarget(trainData);
        
        // Very basic standardization (optional, but recommended for LSTM, we skip for brevity here since we just want it structured for now)
        // Let's create sequence data NDArrays.
        // For DL4J, sequence features shape: [minibatch, numFeatures, sequenceLength]
        int numTrainExamples = xTrainRaw.length - seqLength;
        INDArray trainFeatures = Nd4j.create(numTrainExamples, nIn, seqLength);
        INDArray trainLabels = Nd4j.create(numTrainExamples, nOut, seqLength);
        
        for (int i=0; i<numTrainExamples; i++) {
            for (int t=0; t<seqLength; t++) {
                for (int j=0; j<nIn; j++) {
                    trainFeatures.putScalar(new int[]{i, j, t}, xTrainRaw[i+t][j]);
                }
                trainLabels.putScalar(new int[]{i, 0, t}, yTrainRaw[i+t] == 0 ? 1.0 : 0.0);
                trainLabels.putScalar(new int[]{i, 1, t}, yTrainRaw[i+t] == 1 ? 1.0 : 0.0);
            }
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
        
        net.fit(trainFeatures, trainLabels);
        
        System.out.println("Predicting with LSTM...");
        double[][] xTestRaw = extractFeatures(testData);
        int numTestExamples = xTestRaw.length - seqLength;
        INDArray testFeatures = Nd4j.create(numTestExamples, nIn, seqLength);
        
        for (int i=0; i<numTestExamples; i++) {
            for (int t=0; t<seqLength; t++) {
                for (int j=0; j<nIn; j++) {
                    testFeatures.putScalar(new int[]{i, j, t}, xTestRaw[i+t][j]);
                }
            }
        }
        
        INDArray output = net.output(testFeatures);
        
        int[] yPred = new int[testData.rowCount()];
        for(int i=0; i<seqLength; i++) {
            yPred[i] = 0; // Pad for sequence length
        }
        for(int i=0; i<numTestExamples; i++) {
            // Get prediction from last time step
            double p0 = output.getDouble(i, 0, seqLength - 1);
            double p1 = output.getDouble(i, 1, seqLength - 1);
            yPred[i + seqLength] = p1 > p0 ? 1 : 0;
        }
        
        return yPred;
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
