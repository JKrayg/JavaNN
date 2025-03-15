package src.com.JakeKrayger.nn.training.optimizers;

import org.ejml.simple.SimpleMatrix;

import src.com.JakeKrayger.nn.components.Layer;

public class Adam extends Optimizer {
    private double learningRate;
    private double momentumDecay;
    private double varianceDecay;
    private double epsilon;
    private int updateCount = 1;

    public Adam(double learningRate) {
        this.learningRate = learningRate;
        this.momentumDecay = 0.9;
        this.varianceDecay = 0.999;
        this.epsilon = 1e-8;
    }

    public Adam(double learningRate, double momentumDecay) {
        this.learningRate = learningRate;
        this.momentumDecay = momentumDecay;
        this.varianceDecay = 0.999;
        this.epsilon = 1e-8;
    }

    public Adam(double learningRate, double momentumDecay, double varianceDecay) {
        this.learningRate = learningRate;
        this.momentumDecay = momentumDecay;
        this.varianceDecay = varianceDecay;
        this.epsilon = 1e-8;
    }

    public Adam(double learningRate, double momentumDecay, double varianceDecay, double epsilon) {
        this.learningRate = learningRate;
        this.momentumDecay = momentumDecay;
        this.varianceDecay = varianceDecay;
        this.epsilon = epsilon;
    }
    
    // clean these
    public SimpleMatrix executeWeightsUpdate(Layer l) {
        SimpleMatrix gWrtW = l.getGradientWeights();
        SimpleMatrix momentum = l.getWeightsMomentum();
        SimpleMatrix momentumD = momentum.scale(momentumDecay);
        SimpleMatrix momentumG = gWrtW.scale(1 - momentumDecay);
        SimpleMatrix momentumOfWeights = momentumD.plus(momentumG);

        SimpleMatrix variance = l.getWeightsVariance();
        SimpleMatrix varianceD = variance.scale(varianceDecay);
        SimpleMatrix varianceG = gWrtW.elementPower(2).scale(1 - varianceDecay);
        SimpleMatrix varianceOfWeights = varianceD.plus(varianceG);

        SimpleMatrix currWeights = l.getWeights();
        double learningRate = this.learningRate;
        SimpleMatrix biasCorrectedMomentum = momentumOfWeights.divide(1 - Math.pow(momentumDecay, updateCount));
        SimpleMatrix biasCorrectedVariance = varianceOfWeights.divide(1 - Math.pow(varianceDecay, updateCount));
        double epsilon = this.epsilon;
        SimpleMatrix biasCorrection = biasCorrectedMomentum.elementDiv(biasCorrectedVariance.elementPower(0.5).plus(epsilon));
        SimpleMatrix biasCorrectionLr = biasCorrection.scale(learningRate);
        SimpleMatrix updatedWeights = currWeights.minus(biasCorrectionLr);

        return updatedWeights;
    }

    public SimpleMatrix executeBiasUpdate(Layer l) {
        SimpleMatrix gWrtB = l.getGradientBias();
        SimpleMatrix momentum = l.getBiasMomentum();
        SimpleMatrix momentumD = momentum.scale(momentumDecay);
        SimpleMatrix momentumG = gWrtB.scale(1 - momentumDecay);
        SimpleMatrix momentumOfBiases = momentumD.plus(momentumG);

        SimpleMatrix variance = l.getBiasVariance();
        SimpleMatrix varianceD = variance.scale(varianceDecay);
        SimpleMatrix varianceG = gWrtB.elementPower(2).scale(1 - varianceDecay);
        SimpleMatrix varianceOfBias = varianceD.plus(varianceG);

        SimpleMatrix currBiases = l.getBias();
        double learningRate = this.learningRate;
        SimpleMatrix biasCorrectedMomentum = momentumOfBiases.divide(1 - Math.pow(momentumDecay, updateCount));
        SimpleMatrix biasCorrectedVariance = varianceOfBias.divide(1 - Math.pow(varianceDecay, updateCount));
        double epsilon = this.epsilon;
        SimpleMatrix biasCorrection = biasCorrectedMomentum.elementDiv(biasCorrectedVariance.elementPower(0.5).plus(epsilon));
        SimpleMatrix biasCorrectionLr = biasCorrection.scale(learningRate);
        SimpleMatrix updatedBiases = currBiases.minus(biasCorrectionLr);

        return updatedBiases;
    }

    public double getLearningRate() {
        return learningRate;
    }

    public double getMomentumDecay() {
        return momentumDecay;
    }

    public double getVarianceDecay() {
        return varianceDecay;
    }

    public double getEpsilon() {
        return epsilon;
    }

    public void updateCount() {
        this.updateCount += 1;
    }
}
