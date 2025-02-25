package src.com.JakeKrayger.nn.utils;

import org.ejml.simple.SimpleMatrix;
import src.com.JakeKrayger.nn.components.Layer;

public class MathUtils {
    // weighted sum for single node ∑(wi⋅xi)+b
    public SimpleMatrix weightedSum(Layer prevLayer, Layer currLayer) {
        SimpleMatrix weights = currLayer.getWeights();
        SimpleMatrix biasT = currLayer.getBias().transpose();
        SimpleMatrix dot = prevLayer.getActivations().mult(weights);
        int rows = dot.getNumRows();
        int cols = dot.getNumCols();
        double[][] bias = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                bias[i][j] = biasT.get(j);
            }
        }

        return dot.plus(new SimpleMatrix(bias));
    }

    public SimpleMatrix weightedSum(SimpleMatrix inputData, Layer currLayer) {
        SimpleMatrix weights = currLayer.getWeights();
        SimpleMatrix biasT = currLayer.getBias().transpose();
        SimpleMatrix dot = inputData.mult(weights);
        int rows = dot.getNumRows();
        int cols = dot.getNumCols();
        double[][] bias = new double[rows][cols];
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                bias[i][j] = biasT.get(j);
            }
        }
        
        return dot.plus(new SimpleMatrix(bias));
    }

    public double mean(double[] dubs) {
        double sum = 0;
        for (int i = 0; i < dubs.length; i++) {
            sum = sum + dubs[i];
        }

        return sum / dubs.length;
    }

    public double std(double[] dubs) {
        double mean = this.mean(dubs);
        double v = 0;
        for (double d: dubs) {
            v += ((d - mean) * (d - mean));
        }

        return Math.sqrt(v / dubs.length);
    }

    public double std(SimpleMatrix v) {
        double mean = (v.elementSum() / v.getNumElements());
        double s = 0;

        for (int i = 0; i < v.getNumElements(); i++) {
            s += (v.get(i) - mean) * (v.get(i) - mean);
        }

        return Math.sqrt(s / v.getNumElements());
    }
}
