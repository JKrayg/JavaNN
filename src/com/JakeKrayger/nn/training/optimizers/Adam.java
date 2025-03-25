package src.com.JakeKrayger.nn.training.optimizers;

import org.ejml.simple.SimpleMatrix;

import src.com.JakeKrayger.nn.components.Layer;

public class Adam extends Optimizer {
    private double learningRate;
    private double momentumDecay = 0.9;
    private double varianceDecay = 0.999;
    private double epsilon = 1e-8;
    private int updateCount = 1;

    public Adam(double learningRate) {
        this.learningRate = learningRate;
    }
    
    public void setMomentumDecay(double md) {
        this.momentumDecay = md;
    }

    public void setVarianceDecay(double vd) {
        this.varianceDecay = vd;
    }

    public void setEpsilon(double epsilon) {
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

        l.setWeightsMomentum(momentumOfWeights);
        l.setWeightsVariance(varianceOfWeights);

        return updatedWeights;
    }

    // public SimpleMatrix executeShiftUpdate(Layer l) {
    //     SimpleMatrix gWrtS = l.getGradientShifts();
    //     SimpleMatrix momentumOfShifts = l.getShiftsMomentum()
    //                                      .scale(momentumDecay)
    //                                      .plus(gWrtS.scale(1 - momentumDecay));

    //     SimpleMatrix varianceOfShifts = l.getShiftsVariance()
    //                                      .scale(varianceDecay)
    //                                      .plus(gWrtS.elementPower(2).scale(1 - varianceDecay));

    //     SimpleMatrix currShifts = l.getShifts();
    //     SimpleMatrix biasCorrectedMomentum = momentumOfShifts.divide(1 - Math.pow(momentumDecay, updateCount));
    //     SimpleMatrix biasCorrectedVariance = varianceOfShifts.divide(1 - Math.pow(varianceDecay, updateCount));
    //     SimpleMatrix biasCorrection = biasCorrectedMomentum
    //                                   .elementDiv(biasCorrectedVariance.elementPower(0.5).plus(epsilon))
    //                                   .scale(learningRate);

    //     SimpleMatrix updatedShifts = currShifts.minus(biasCorrection);

        // l.setShiftMomentum(momentumOfWeights);
        // l.getShiftVariance(varianceOfWeights);

    //     return updatedShifts;
    // }

    public SimpleMatrix executeBiasUpdate(Layer l) {
        // System.out.println("called");
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

        l.setBiasesMomentum(momentumOfBiases);
        l.setBiasesVariance(varianceOfBias);

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
