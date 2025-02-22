package src.com.JakeKrayger.nn.components;

import java.util.ArrayList;
import org.ejml.simple.SimpleMatrix;

public class Node {
    private SimpleMatrix values;
    private SimpleMatrix weights;
    private double bias;
    private ArrayList<Node> forwardConnections;
    private ArrayList<Node> backwardConnections;

    public Node() {}

    public Node(double bias) {
        // this.values = values;
        this.bias = bias;
    }

    // public Node(SimpleMatrix values) {
    //     this.values = values;
    // }

    public SimpleMatrix getValues() {
        return values;
    }

    public SimpleMatrix getWeights() {
        return weights;
    }

    public double getBias() {
        return bias;
    }

    public ArrayList<Node> getForwConnections() {
        return forwardConnections;
    }

    public ArrayList<Node> getBackConnections() {
        return backwardConnections;
    }

    public void setValues(double[] weights) {
        this.weights = new SimpleMatrix(weights);
    }

    public void setWeights(double[] weights) {
        this.weights = new SimpleMatrix(weights);
    }

    public void setForwConnection(Node node) {
        if (forwardConnections == null) {
            forwardConnections = new ArrayList<>();
        }
        
        this.forwardConnections.add(node);
    }

    public void setBackConnection(Node node) {
        if (backwardConnections == null) {
            backwardConnections = new ArrayList<>();
        }
        this.backwardConnections.add(node);
    }
}