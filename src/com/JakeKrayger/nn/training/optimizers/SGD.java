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
        return n.getShift().minus(n.getGradientShift().scale(learningRate));
    }

    public SimpleMatrix executeScaleUpdate(Normalization n) {
        return n.getScale().minus(n.getGradientScale().scale(learningRate));
    }

    public double getLearningRate() {
        return learningRate;
    }
}
