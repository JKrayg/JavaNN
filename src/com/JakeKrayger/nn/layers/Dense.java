package src.com.JakeKrayger.nn.layers;

import org.ejml.simple.SimpleMatrix;
import src.com.JakeKrayger.nn.activation.ActivationFunction;
import src.com.JakeKrayger.nn.components.*;
import src.com.JakeKrayger.nn.training.regularizers.Regularizer;

// OutputLayer and Dense are pretty much the same class
public class Dense extends Layer {
    public Dense(int numNeurons, ActivationFunction actFunc) {
        super(numNeurons, actFunc);
    }

    public Dense(int numNeurons, ActivationFunction actFunc, Regularizer reg) {
        super(numNeurons, actFunc, reg);
    }

    public Dense(int numNeurons, ActivationFunction actFunc, int inputSize) {
        super(numNeurons, actFunc, inputSize);
    }

    public Dense(int numNeurons, ActivationFunction actFunc, Regularizer reg, int inputSize) {
        super(numNeurons, actFunc, reg, inputSize);
    }
}