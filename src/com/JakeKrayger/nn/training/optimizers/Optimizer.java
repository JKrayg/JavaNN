package src.com.JakeKrayger.nn.training.optimizers;

import org.ejml.simple.SimpleMatrix;

import src.com.JakeKrayger.nn.components.Layer;

public abstract class Optimizer {
    public abstract SimpleMatrix executeWeightsUpdate(Layer l);
    public abstract SimpleMatrix executeBiasUpdate(Layer l);
    public abstract double getLearningRate();
    // public abstract double getMomentumDecay();
    // public abstract double getVarianceDecay();
    // public abstract double getEpsilon();
}
