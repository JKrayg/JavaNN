package src.com.JakeKrayger.nn.activation;

import org.ejml.simple.SimpleMatrix;

public class Tanh extends ActivationFunction {
    public SimpleMatrix execute(SimpleMatrix z) {
        double[] v = new double[z.getNumRows()];

        for (int i = 0; i < z.getNumRows(); i++) {
            double curr = z.get(i);
            v[i] = (Math.exp(curr) - Math.exp(-curr)) / (Math.exp(curr) + Math.exp(-curr));
        }
        return new SimpleMatrix(v);
    }
}
