package src.com.JakeKrayger.nn.utils;

import java.util.ArrayList;

import org.ejml.simple.SimpleMatrix;
import src.com.JakeKrayger.nn.components.Layer;
import src.com.JakeKrayger.nn.components.Node;

public class MathUtils {
    // weighted sum for single node ∑(wi⋅xi)+b
    public double weightedSum(Layer prevLayer, Node currNode) {
        SimpleMatrix matrix = prevLayer.getNodes().get(0).getValues();

        for (int i = 1; i < prevLayer.getNodes().size(); i++) {
            matrix.concatColumns(prevLayer.getNodes().get(i).getValues());
        }

        return currNode.getWeights().dot(matrix) + currNode.getBias();

        // SimpleMatrix prevVals;
        // SimpleMatrix currWeights;

        // double[] vals = new double[prevLayer.getNodes().size()];
        // double[] weights = new double[currNode.getWeights().getNumElements()];

        // for (int i = 0; i < prevLayer.getNodes().size(); i++) {
        //     vals[i] = prevLayer.getNodes().get(i).getValues();
        //     weights[i] = currNode.getWeights().get(i);
        // }

        // prevVals = new SimpleMatrix(vals);
        // currWeights = new SimpleMatrix(weights);

        // return currWeights.dot(prevVals) + currNode.getBias();
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
