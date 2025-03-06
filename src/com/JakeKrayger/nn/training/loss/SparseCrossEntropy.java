package src.com.JakeKrayger.nn.training.loss;

import org.ejml.simple.SimpleMatrix;
import src.com.JakeKrayger.nn.components.Layer;

public class SparseCrossEntropy extends Loss {
    public double execute(SimpleMatrix activations, double[] labels) {
        return 0.0;
    }

    public SimpleMatrix outputGradientWeights(Layer out, Layer prev, double[] labels) {
        return new SimpleMatrix(0, 0);
    }

    public SimpleMatrix outputGradientBias(Layer out, double[] labels) {
        return new SimpleMatrix(0, 0);
    }
}
