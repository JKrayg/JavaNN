package src.com.JakeKrayger.nn.components;

import org.ejml.simple.SimpleMatrix;
import src.com.JakeKrayger.nn.activation.ActivationFunction;
import src.com.JakeKrayger.nn.layers.Output;

public class Layer {
    private int numNeurons;
    private SimpleMatrix activationsM;
    private SimpleMatrix weightsM;
    private SimpleMatrix biasV;
    private SimpleMatrix gradientWrtWeights;
    private SimpleMatrix gradientWrtBiases;
    private ActivationFunction func;
    private int inputSize;

    // public Layer(int numNeurons, SimpleMatrix b, ActivationFunction func) {
    //     this.numNeurons = numNeurons;
    //     this.biasV = b;
    //     this.func = func;
    // }

    public Layer(int numNeurons, ActivationFunction func) {
        this.numNeurons = numNeurons;
        this.func = func;
    }

    // public Layer(int numNeurons, SimpleMatrix b, ActivationFunction func, int inputSize) {
    //     this.numNeurons = numNeurons;
    //     this.biasV = b;
    //     this.func = func;
    //     this.inputSize = inputSize;
    // }

    public Layer(int numNeurons, ActivationFunction func, int inputSize) {
        this.numNeurons = numNeurons;
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

    public void setBiases(SimpleMatrix biases) {
        this.biasV = biases;
    }

    public void setGradientWeights(SimpleMatrix gWrtW) {
        this.gradientWrtWeights = gWrtW;
    }

    public void setGradientBiases(SimpleMatrix gWrtB) {
        this.gradientWrtBiases = gWrtB;
    }

    public SimpleMatrix getGradient() {
        if (this instanceof Output) {
            // switch statement for different loss function and activation
            // return gradient of loss wrt output
        } else {
            // switch statement for different activation
            // return gradient wrt pre-activation
        }

        return new SimpleMatrix(0, 0);
    }

    public SimpleMatrix outputGradientWeights(Layer currLayer, Layer prevLayer, double[] labels) {
        SimpleMatrix error = outputGradientBias(currLayer, labels);
        return (prevLayer.getActivations().transpose()).mult(error).divide(labels.length);
    }

    public SimpleMatrix outputGradientBias(Layer currLayer, double[] labels) {
        return currLayer.getActivations().minus(new SimpleMatrix(labels));
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