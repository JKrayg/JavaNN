package src.com.JakeKrayger.nn.components;

import java.util.function.BinaryOperator;

import org.ejml.simple.SimpleMatrix;
import src.com.JakeKrayger.nn.activation.ActivationFunction;
import src.com.JakeKrayger.nn.activation.Sigmoid;
import src.com.JakeKrayger.nn.layers.Output;
import src.com.JakeKrayger.nn.training.loss.BinCrossEntropy;
import src.com.JakeKrayger.nn.training.loss.Loss;

public class Layer {
    private int numNeurons;
    private SimpleMatrix preActivation;
    private SimpleMatrix activationsM;
    private SimpleMatrix weights;
    private SimpleMatrix bias;
    private SimpleMatrix gradientWrtWeights;
    private SimpleMatrix gradientWrtBiases;
    private ActivationFunction func;
    private Loss loss;
    private int inputSize;

    public Layer() {}

    public Layer(int numNeurons, ActivationFunction func) {
        this.numNeurons = numNeurons;
        this.func = func;
    }

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

    public SimpleMatrix getPreActivation() {
        return preActivation;
    }

    public SimpleMatrix getWeights() {
        return weights;
    }

    public SimpleMatrix getBias() {
        return bias;
    }

    public ActivationFunction getActFunc() {
        return func;
    }

    public int getInputSize() {
        return inputSize;
    }

    public Loss getLoss() {
        return loss;
    }

    public SimpleMatrix getGradientWeights() {
        return gradientWrtWeights;
    }

    public SimpleMatrix getGradientBias() {
        return gradientWrtBiases;
    }

    public void setWeights(SimpleMatrix weights) {
        this.weights = weights;
    }

    public void setBiases(SimpleMatrix biases) {
        this.bias = biases;
    }

    public void setPreActivations(SimpleMatrix preAct) {
        this.preActivation = preAct;
    }

    public void setActivations(SimpleMatrix activations) {
        this.activationsM = activations;
    }

    public void setGradientWeights(SimpleMatrix gWrtW) {
        this.gradientWrtWeights = gWrtW;
    }

    public void setGradientBiases(SimpleMatrix gWrtB) {
        this.gradientWrtBiases = gWrtB;
    }

    public void setLoss(Loss loss) {
        this.loss = loss;
    }

    public SimpleMatrix getGradient() {
        SimpleMatrix gradient = null;
        if (this instanceof Output) {
            gradient = loss.gradient(this, ((Output) this).getLabels());
        } else {
            gradient = func.gradient(this, preActivation);
        }

        return gradient;
    }

    public SimpleMatrix gradientWeights(Layer prevLayer, SimpleMatrix gradient) {
        SimpleMatrix gWrtW = prevLayer.getActivations().transpose().mult(gradient).divide(prevLayer.getActivations().getNumRows());
        return gWrtW;
    }

    public SimpleMatrix gradientBias(SimpleMatrix gradient) {
        double[] biasG = new double[gradient.getNumCols()];
        for (int i = 0; i < gradient.getNumCols(); i++) {
            SimpleMatrix col = gradient.extractVector(false, i);
            biasG[i] = col.elementSum() / gradient.getNumRows();
        }
        return new SimpleMatrix(biasG);
    }

    public void updateWeights(SimpleMatrix gradientWrtWeights, double learningRate) {
        this.weights = this.weights.minus(gradientWrtWeights.scale(learningRate));
    }

    public void updateBiases(SimpleMatrix gradientWrtBiases, double learningRate) {
        // double mean = gradientWrtBiases.elementSum() / gradientWrtBiases.getNumRows();
        // this.biasV = this.biasV.minus(mean * learningRate);
        this.bias = this.bias.minus(gradientWrtBiases.scale(learningRate));
    }
    
}