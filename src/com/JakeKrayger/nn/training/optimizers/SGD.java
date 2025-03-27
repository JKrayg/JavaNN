package src.com.JakeKrayger.nn.training.optimizers;

import org.ejml.simple.SimpleMatrix;

import src.com.JakeKrayger.nn.components.Layer;
import src.com.JakeKrayger.nn.training.normalization.BatchNormalization;
import src.com.JakeKrayger.nn.training.normalization.Normalization;

public class SGD extends Optimizer {
    private double learningRate;

    public SGD(double learningRate) {
        this.learningRate = learningRate;
    }

    public SimpleMatrix executeWeightsUpdate(Layer l) {
        return l.getWeights().minus(l.getGradientWeights().scale(learningRate));
    }

    public SimpleMatrix executeBiasUpdate(Layer l) {
        return l.getBias().minus(l.getGradientBias().scale(learningRate));
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
