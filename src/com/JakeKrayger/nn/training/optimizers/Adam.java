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
    
    
    public SimpleMatrix executeWeightsUpdate(Layer l) {
        SimpleMatrix gWrtW = l.getGradientWeights();
        SimpleMatrix momentumOfWeights = l.getWeightsMomentum()
                                         .scale(momentumDecay)
                                         .plus(gWrtW.scale(1 - momentumDecay));

        SimpleMatrix varianceOfWeights = l.getWeightsVariance()
                                         .scale(varianceDecay)
                                         .plus(gWrtW.elementPower(2).scale(1 - varianceDecay));

        SimpleMatrix currWeights = l.getWeights();
        SimpleMatrix biasCorrectedMomentum = momentumOfWeights.divide(1 - Math.pow(momentumDecay, updateCount));
        SimpleMatrix biasCorrectedVariance = varianceOfWeights.divide(1 - Math.pow(varianceDecay, updateCount));
        SimpleMatrix biasCorrection = biasCorrectedMomentum
                                      .elementDiv(biasCorrectedVariance.elementPower(0.5).plus(epsilon))
                                      .scale(learningRate);

        SimpleMatrix updatedWeights = currWeights.minus(biasCorrection);

        return updatedWeights;
    }

    public SimpleMatrix executeBiasUpdate(Layer l) {
        SimpleMatrix gWrtB = l.getGradientBias();
        SimpleMatrix momentumOfBiases = l.getBiasMomentum()
                                        .scale(momentumDecay)
                                        .plus(gWrtB.scale(1 - momentumDecay));

        SimpleMatrix varianceOfBias = l.getBiasVariance()
                                      .scale(varianceDecay)
                                      .plus(gWrtB.elementPower(2).scale(1 - varianceDecay));

        SimpleMatrix currBiases = l.getBias();
        SimpleMatrix biasCorrectedMomentum = momentumOfBiases.divide(1 - Math.pow(momentumDecay, updateCount));
        SimpleMatrix biasCorrectedVariance = varianceOfBias.divide(1 - Math.pow(varianceDecay, updateCount));
        SimpleMatrix biasCorrection = biasCorrectedMomentum
                                      .elementDiv(biasCorrectedVariance.elementPower(0.5).plus(epsilon))
                                      .scale(learningRate);

        SimpleMatrix updatedBiases = currBiases.minus(biasCorrection);

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
