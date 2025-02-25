package src.com.JakeKrayger.nn.utils;

import org.ejml.simple.SimpleMatrix;
import src.com.JakeKrayger.nn.components.Layer;

public class MathUtils {
    // weighted sum for single node ∑(wi⋅xi)+b
    public SimpleMatrix weightedSum(Layer prevLayer, Layer currLayer) {
        double[][] bias = new double[currLayer.getWeights().getNumRows()][currLayer.getBias().getNumCols()];
        for (int i = 0; i < currLayer.getWeights().getNumRows(); i++) {
            for (int j = 0; j < currLayer.getWeights().getNumCols(); j++) {
                bias[i][j] = currLayer.getBias().transpose().get(j);
            }
        }

        SimpleMatrix bm = new SimpleMatrix(bias);
        return prevLayer.getActivations().mult(currLayer.getWeights()).plus(bm);
    }

    public SimpleMatrix weightedSum(double[][] inputData, Layer currLayer) {
        SimpleMatrix data = new SimpleMatrix(inputData);
        double[][] bias = new double[currLayer.getWeights().getNumRows()][currLayer.getWeights().getNumCols()];
        for (int i = 0; i < currLayer.getWeights().getNumRows(); i++) {
            for (int j = 0; j < currLayer.getWeights().getNumCols(); j++) {
                bias[i][j] = currLayer.getBias().transpose().get(j);
            }
        }

        SimpleMatrix b = new SimpleMatrix(bias);
        System.out.println(data);
        System.out.println(currLayer.getWeights());
        System.out.println(new SimpleMatrix(data.mult(currLayer.getWeights())).plus(b));
        return new SimpleMatrix(data.mult(currLayer.getWeights())).plus(b);
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
            v = v + ((d - mean) * (d - mean));
        }

        return Math.sqrt(v / dubs.length);
    }
}
