package src.com.JakeKrayger.nn.activation;

import org.ejml.simple.SimpleMatrix;

public class Softmax extends ActivationFunction {
    // weighted sum for single node ∑(wi⋅xi)+b
    public SimpleMatrix execute(SimpleMatrix z) {

        double[] v = new double[z.getNumRows()];
        double sum = 0;

        for (int i = 0; i < z.getNumRows(); i++) {
            v[i] = Math.exp(z.get(i));
            sum = sum + v[i];
        }

        for (int j = 0; j < v.length; j++) {
            v[j] = v[j] / sum;
        }

        return new SimpleMatrix(v);
    }
    
}
