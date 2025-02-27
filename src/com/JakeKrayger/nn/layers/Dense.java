package src.com.JakeKrayger.nn.layers;

import org.ejml.simple.SimpleMatrix;
import src.com.JakeKrayger.nn.activation.ActivationFunction;
import src.com.JakeKrayger.nn.components.*;

// OutputLayer and Dense are pretty much the same class
public class Dense extends Layer {

    public Dense(int numNeurons, ActivationFunction actFunc) {
        super(numNeurons, createBiasV(numNeurons), actFunc);
    }

    public Dense(int numNeurons, ActivationFunction actFunc, int inputSize) {
        super(numNeurons, createBiasV(numNeurons), actFunc, inputSize);
    }

    private static SimpleMatrix createBiasV(int numNeurons) {
        double[] b = new double[numNeurons];
        for (int i = 0; i < numNeurons; i++) {
            b[i] = 0.01;
        }
        return new SimpleMatrix(b);
    }
}