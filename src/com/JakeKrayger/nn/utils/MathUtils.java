package src.com.JakeKrayger.nn.utils;

import org.ejml.simple.SimpleMatrix;
import src.com.JakeKrayger.nn.components.Layer;
import src.com.JakeKrayger.nn.components.Node;

public class MathUtils {
    // weighted sum for single node ∑(wi⋅xi)+b
    public double weightedSum(Layer prevLayer, Node currNode) {
        SimpleMatrix prevVals;
        SimpleMatrix currWeights;

        double[] vals = new double[prevLayer.getNodes().size()];
        double[] weights = new double[currNode.getWeights().size()];

        for (int i = 0; i < prevLayer.getNodes().size(); i++) {
            vals[i] = prevLayer.getNodes().get(i).getValue();
            weights[i] = currNode.getWeights().get(i);
        }

        prevVals = new SimpleMatrix(vals);
        currWeights = new SimpleMatrix(weights);

        return currWeights.dot(prevVals) + currNode.getBias();
    }
}
