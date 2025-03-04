package src.com.JakeKrayger.nn.components;

import org.ejml.simple.SimpleMatrix;
import src.com.JakeKrayger.nn.activation.ActivationFunction;

public class Layer {
    private int numNeurons;
    private SimpleMatrix activationsM;
    private SimpleMatrix weightsM;
    private SimpleMatrix biasV;
    private SimpleMatrix gradientWrtWeights;
    private SimpleMatrix gradientWrtBiases;
    private ActivationFunction func;
    private int inputSize;

    public Layer(int numNeurons, SimpleMatrix b, ActivationFunction func) {
        this.numNeurons = numNeurons;
        this.biasV = b;
        this.func = func;
    }

    public Layer(int numNeurons, SimpleMatrix b, ActivationFunction func, int inputSize) {
        this.numNeurons = numNeurons;
        this.biasV = b;
        this.func = func;
        this.inputSize = inputSize;
    }

    public int getNumNeurons() {
        return numNeurons;
    }

    public SimpleMatrix getActivations() {
        return activationsM;
    }

    public SimpleMatrix getWeights() {
        return weightsM;
    }

    public SimpleMatrix getBias() {
        return biasV;
    }

    public ActivationFunction getActFunc() {
        return func;
    }

    public int getInputSize() {
        return inputSize;
    }

    public void setWeights(SimpleMatrix weights) {
        this.weightsM = weights;
    }

    public void setGradientWeights(SimpleMatrix gWrtW) {
        this.gradientWrtWeights = gWrtW;
    }

    public void setGradientBiases(SimpleMatrix gWrtB) {
        this.gradientWrtBiases = gWrtB;
    }

    public void updateWeights(SimpleMatrix gradientWrtWeights, double learningRate) {
        this.weightsM = this.weightsM.minus(gradientWrtWeights.scale(learningRate));
    }

    public void updateBiases(SimpleMatrix gradientWrtBiases, double learningRate) {
        double mean = gradientWrtBiases.elementSum() / gradientWrtBiases.getNumRows();
        this.biasV = this.biasV.minus(mean * learningRate);
    }

    public void setActivations(SimpleMatrix activations) {
        this.activationsM = activations;
    }
    
}