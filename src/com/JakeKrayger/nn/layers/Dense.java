package src.com.JakeKrayger.nn.layers;

import org.ejml.simple.SimpleMatrix;
import src.com.JakeKrayger.nn.activation.ActivationFunction;
import src.com.JakeKrayger.nn.components.*;

// OutputLayer and Dense are pretty much the same class
public class Dense extends Layer {
    public Dense(int numNeurons, ActivationFunction actFunc) {
        super(numNeurons, actFunc);
    }

    public Dense(int numNeurons, ActivationFunction actFunc, int inputSize) {
        super(numNeurons, actFunc, inputSize);
    }
}