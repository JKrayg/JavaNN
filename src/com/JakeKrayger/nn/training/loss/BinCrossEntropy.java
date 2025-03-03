package src.com.JakeKrayger.nn.training.loss;

import org.ejml.simple.SimpleMatrix;

import src.com.JakeKrayger.nn.components.Layer;

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

    public SimpleMatrix outputGradientWeights(Layer currLayer, Layer prevLayer, double[] labels) {
        SimpleMatrix error = outputGradientBias(currLayer, labels);
        return (prevLayer.getActivations().transpose()).mult(error).divide(labels.length);
    }

    public SimpleMatrix outputGradientBias(Layer currLayer, double[] labels) {
        return currLayer.getActivations().minus(new SimpleMatrix(labels));
    }
}
