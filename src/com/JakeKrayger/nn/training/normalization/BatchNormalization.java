package src.com.JakeKrayger.nn.training.normalization;

import org.ejml.simple.SimpleMatrix;

public class BatchNormalization extends Normalization {
    private SimpleMatrix scale;
    private SimpleMatrix shift;
    private SimpleMatrix means;
    private SimpleMatrix variances;
    private double momentum = 0.99;
    private double epsilon = 1e-3;
    private boolean beforeActivation = true;

    public BatchNormalization() {}

    public void setScale(SimpleMatrix scale) {
        this.scale = scale;
    }

    public void setShift(SimpleMatrix shift) {
        this.shift = shift;
    }

    public void setMeans(SimpleMatrix means) {
        this.means = means;
    }

    public void setVariances(SimpleMatrix variances) {
        this.variances = variances;
    }

    public void setMomentum(double momentum) {
        this.momentum = momentum;
    }

    public void setEpsilon(double epsilon) {
        this.epsilon = epsilon;
    }

    public void beforeActivation(boolean b) {
        this.beforeActivation = b;
    }

    public boolean isBeforeActivation() {
        return beforeActivation;
    }

    public SimpleMatrix getScale() {
        return scale;
    }

    public SimpleMatrix getShift() {
        return shift;
    }

    public SimpleMatrix getMeans() {
        return means;
    }

    public SimpleMatrix getVariances() {
        return variances;
    }

    public double getMomentum() {
        return momentum;
    }
}
