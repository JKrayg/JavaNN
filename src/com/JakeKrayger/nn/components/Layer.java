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
    private SimpleMatrix weightsM;
    private SimpleMatrix biasV;
    private SimpleMatrix gradientWrtWeights;
    private SimpleMatrix gradientWrtBiases;
    private ActivationFunction func;
    private Loss loss;
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

    public SimpleMatrix getPreActivation() {
        return preActivation;
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
            // switch statement for different loss function and activation
            // return gradient of loss wrt output
        } else {
            gradient = func.gradient(this, preActivation);
            // switch statement for different activation
            // return gradient wrt pre-activation
        }

        return gradient;
    }

    public SimpleMatrix outputGradientWeights(Layer currLayer, Layer prevLayer, SimpleMatrix labels) {
        SimpleMatrix error = outputGradientBias(currLayer, labels);
        return (prevLayer.getActivations().transpose()).mult(error).divide(labels.getNumElements());
    }

    public SimpleMatrix outputGradientBias(Layer currLayer, SimpleMatrix labels) {
        return currLayer.getActivations().minus(new SimpleMatrix(labels));
    }

    public void updateWeights(SimpleMatrix gradientWrtWeights, double learningRate) {
        this.weightsM = this.weightsM.minus(gradientWrtWeights.scale(learningRate));
    }

    public void updateBiases(SimpleMatrix gradientWrtBiases, double learningRate) {
        double mean = gradientWrtBiases.elementSum() / gradientWrtBiases.getNumRows();
        this.biasV = this.biasV.minus(mean * learningRate);
    }
    
}