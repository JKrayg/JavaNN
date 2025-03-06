package src.com.JakeKrayger.nn.layers;

import src.com.JakeKrayger.nn.activation.ActivationFunction;
import src.com.JakeKrayger.nn.components.Layer;

public class Output extends Layer {
    public Output(int numNeurons, ActivationFunction actFunc) {
        super(numNeurons, actFunc);
    }
}
