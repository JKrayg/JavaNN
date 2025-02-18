package src.com.JakeKrayger.nn.components;
import java.util.ArrayList;

public class Node {
    private double value;
    private double weight;
    private double bias;
    private ActivationFunction func;
    private ArrayList<Node> forwardConnections;
    private ArrayList<Node> backwardConnections;

    // public Node(double value, double weight, double bias, ActivationFunction
    // func) {
    // this.value = value;
    // this.weight = weight;
    // this.bias = bias;
    // this.func = func;
    // }

    public Node(double value, double bias, ActivationFunction func) {
        this.value = value;
        this.bias = bias;
        this.func = func;
    }

    public Node(double value) {
        this.value = value;
    }

    public double getValue() {
        return value;
    }

    public double getWeight() {
        return weight;
    }

    public double getBias() {
        return bias;
    }

    public ActivationFunction getActFunc() {
        return func;
    }

    public ArrayList<Node> getForwConnections() {
        return forwardConnections;
    }

    public ArrayList<Node> getBackConnections() {
        return backwardConnections;
    }

    public void setWeight(double weight) {
        this.weight = weight;
    }

    public void setActFunc(ActivationFunction func) {
        this.func = func;
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