package src.com.JakeKrayger.nn.activation;

import org.ejml.simple.SimpleMatrix;

public class ReLU extends ActivationFunction {
    public SimpleMatrix execute(SimpleMatrix z) {
        double[] v = new double[z.getNumRows()];
        for (int i = 0; i < z.getNumRows(); i++) {
            if (z.get(i) > 0) {
                v[i] = z.get(i);
            } else {
                v[i] = 0.0;
            }
        }
        return new SimpleMatrix(v);
    }
}
