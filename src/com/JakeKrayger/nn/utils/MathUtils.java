package src.com.JakeKrayger.nn.utils;


import org.ejml.simple.SimpleMatrix;
import src.com.JakeKrayger.nn.components.Layer;

public class MathUtils {
    // weighted sum for single node ∑(wi⋅xi)+b
    public SimpleMatrix weightedSum(Layer prevLayer, Layer currLayer) {
        return prevLayer.getActivations().mult(currLayer.getWeights()).plus(currLayer.getBias());
    }

    public SimpleMatrix weightedSum(double[][] inputData, Layer currLayer) {
        SimpleMatrix data = new SimpleMatrix(inputData);
        System.out.println(data);
        System.out.println(currLayer.getWeights());
        return new SimpleMatrix(data.mult(currLayer.getWeights()));
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
