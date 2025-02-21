package src.com.JakeKrayger.nn.activation;

public class Sigmoid extends ActivationFunction {
    public double execute(double z) {
        return 1 / (1 + Math.exp(-z));
    }
}
