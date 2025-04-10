package src.com.JakeKrayger.nn.training.optimizers;

import org.ejml.simple.SimpleMatrix;

import src.com.JakeKrayger.nn.components.Layer;
import src.com.JakeKrayger.nn.training.normalization.Normalization;

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

    public SimpleMatrix executeShiftUpdate(Normalization n) {
        return new SimpleMatrix(0, 0);
    }

    public SimpleMatrix executeScaleUpdate(Normalization n) {
        return new SimpleMatrix(0, 0);
    }

    public double getLearningRate() {
        return learningRate;
    }
}
