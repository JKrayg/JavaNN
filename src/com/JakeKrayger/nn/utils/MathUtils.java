package src.com.JakeKrayger.nn.utils;

import java.util.ArrayList;

import org.ejml.simple.SimpleMatrix;
import src.com.JakeKrayger.nn.components.Layer;
import src.com.JakeKrayger.nn.components.Node;

public class MathUtils {
    // weighted sum for single node ∑(wi⋅xi)+b
    public SimpleMatrix weightedSum(Layer prevLayer, Node currNode) {
        double[] sums = new double[prevLayer.getNodes().size()];

        for (int i = 0; i < prevLayer.getNodes().size(); i++) {
            sums[i] = currNode.getWeights().dot(prevLayer.getNodes().get(i).getValues()) + 0.1 ;
        }

        return new SimpleMatrix(sums);
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
