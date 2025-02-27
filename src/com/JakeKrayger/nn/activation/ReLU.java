package src.com.JakeKrayger.nn.activation;

import org.ejml.simple.SimpleMatrix;

public class ReLU extends ActivationFunction {
    public SimpleMatrix execute(SimpleMatrix z) {
        int rows = z.getNumRows();
        int cols = z.getNumCols();
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (z.get(i, j) < 0) {
                    z.set(i, j, 0.0);
                }
            }
            
        }
        return z;
    }
}
