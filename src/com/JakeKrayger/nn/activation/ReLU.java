package src.com.JakeKrayger.nn.activation;

import org.ejml.simple.SimpleMatrix;

public class ReLU extends ActivationFunction {
    public SimpleMatrix execute(SimpleMatrix z) {
        // z = weighted sum matrix
        // double[] v = new double[z.getNumRows()];
        for (int i = 0; i < z.getNumRows(); i++) {
            for (int j = 0; j < z.getNumCols(); j++) {
                if (z.get(i, j) < 0) {
                    z.set(i, j, 0.0);
                }
            }
            
        }
        return z;
    }
}
