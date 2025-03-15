package src.com.JakeKrayger.nn.training.loss;

import org.ejml.simple.SimpleMatrix;
import src.com.JakeKrayger.nn.components.Layer;

public class CatCrossEntropy extends Loss {
    public double execute(SimpleMatrix activations, SimpleMatrix labels) {
        // -sum(y[i] * Math.log(a[i])) over classes.
        return 0.0;
    }

    public SimpleMatrix gradient(Layer out, SimpleMatrix labels) {
        return out.getActivations().minus(labels);
    }
    
}
