package src.com.JakeKrayger.nn.training.loss;

import org.ejml.simple.SimpleMatrix;
import src.com.JakeKrayger.nn.components.Layer;

public class MSE extends Loss {
    public double execute(SimpleMatrix activations, SimpleMatrix labels) {
        return 0.0;
    }

    // public SimpleMatrix outputGradientWeights(Layer out, Layer prev, double[] labels) {
    //     return new SimpleMatrix(0, 0);
    // }

    public SimpleMatrix gradient(Layer out, SimpleMatrix labels) {
        // ***
        return out.getActivations().minus(labels);
    }
    
}
