package src.com.JakeKrayger.nn.training.optimizers;

import org.ejml.simple.SimpleMatrix;

import src.com.JakeKrayger.nn.components.Layer;

public class RMSProp extends Optimizer {
    private double learningRate;

    public RMSProp(double learningRate) {
        this.learningRate = learningRate;
    }

    public SimpleMatrix executeWeightsUpdate(Layer l) {
        // **
        return l.getWeights();
    }

    public SimpleMatrix executeBiasUpdate(Layer l) {
        // **
        return l.getBias();
    }

    public double getLearningRate() {
        return learningRate;
    }
}
