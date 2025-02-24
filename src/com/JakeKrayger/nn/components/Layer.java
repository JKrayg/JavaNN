package src.com.JakeKrayger.nn.components;

import java.util.ArrayList;

import org.ejml.simple.SimpleMatrix;
import src.com.JakeKrayger.nn.activation.ActivationFunction;

public class Layer {
    private SimpleMatrix activationsM;
    private SimpleMatrix weightsM;
    private SimpleMatrix biasV;
    private ArrayList<Node> nodes;
    private ActivationFunction func;
    private int inputSize;

    public Layer(ArrayList<Node> nodes) {
        this.nodes = nodes;
    }

    public Layer(ArrayList<Node> nodes, ActivationFunction actFunc) {
        this.nodes = nodes;
        this.func = actFunc;
    }

    public Layer(ArrayList<Node> nodes, ActivationFunction actFunc, int inputSize) {
        this.nodes = nodes;
        this.func = actFunc;
        this.inputSize = inputSize;
    }

    public ArrayList<Node> getNodes() {
        return nodes;
    }

    public ActivationFunction getActFunc() {
        return func;
    }

    public int getInputSize() {
        return inputSize;
    }
    
}