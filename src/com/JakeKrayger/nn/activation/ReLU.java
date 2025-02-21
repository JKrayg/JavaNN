package src.com.JakeKrayger.nn.activation;

public class ReLU extends ActivationFunction {
    public double execute(double z) {
        if (z > 0) {
            return z;
        }
        return 0.0;
    }
}
