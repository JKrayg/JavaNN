package src.com.JakeKrayger.nn.activation;

import org.ejml.simple.SimpleMatrix;

public class Tanh extends ActivationFunction {
    public SimpleMatrix execute(SimpleMatrix z) {
        double[] v = new double[z.getNumRows()];

        for (int i = 0; i < z.getNumRows(); i++) {
            v[i] = (Math.exp(z.get(i)) - Math.exp(-z.get(i))) / (Math.exp(z.get(i)) + Math.exp(-z.get(i)));
        }
        return new SimpleMatrix(v);
    }
}
