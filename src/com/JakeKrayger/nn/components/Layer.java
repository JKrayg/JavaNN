package src.com.JakeKrayger.nn.components;

import java.util.ArrayList;

import org.ejml.simple.SimpleMatrix;
import src.com.JakeKrayger.nn.activation.ActivationFunction;

public class Layer {
    private int numNeurons;
    private SimpleMatrix activationsM;
    private SimpleMatrix weightsM;
    private SimpleMatrix biasV;
    // private ArrayList<Node> nodes;
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
    // public Layer(ArrayList<Node> nodes) {
    //     this.nodes = nodes;
    // }

    // public Layer(ArrayList<Node> nodes, ActivationFunction actFunc) {
    //     this.nodes = nodes;
    //     this.func = actFunc;
    // }

    // public Layer(ArrayList<Node> nodes, ActivationFunction actFunc, int inputSize) {
    //     this.nodes = nodes;
    //     this.func = actFunc;
    //     this.inputSize = inputSize;
    // }

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

    // public ArrayList<Node> getNodes() {
    //     return nodes;
    // }

    public ActivationFunction getActFunc() {
        return func;
    }

    public int getInputSize() {
        return inputSize;
    }

    public void setWeights(SimpleMatrix weights) {
        this.weightsM = weights;
    }

    public void setActivations(SimpleMatrix activations) {
        this.activationsM = activations;
    }
    
}