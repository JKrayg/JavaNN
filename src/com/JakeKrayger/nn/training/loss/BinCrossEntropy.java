package src.com.JakeKrayger.nn.training.loss;

import org.ejml.simple.SimpleMatrix;

public class BinCrossEntropy extends Loss {
    public double execute(SimpleMatrix activations, double[] labels) {
        double sumLoss = 0.0;
        double n = activations.getNumElements();

        for (int i = 0; i < n; i++) {
            double pred = activations.get(i);
            double y = labels[i];
            // need to prevent log(0)
            sumLoss += -(y * Math.log(pred) + (1 - y) * Math.log(1 - pred));
        }
        return sumLoss / n;
    }
}
