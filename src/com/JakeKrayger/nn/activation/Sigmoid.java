package src.com.JakeKrayger.nn.activation;

import src.com.JakeKrayger.nn.components.ActivationFunction;

public class Sigmoid extends ActivationFunction {
    public double execute(double z) {
        return 1 / (1 + Math.exp(-z));
    }
}
