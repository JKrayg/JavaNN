package src.com.JakeKrayger.nn.components;

import java.util.ArrayList;

public class Layer {
    private ArrayList<Node> nodes;
    private ActivationFunction func;

    public Layer(ArrayList<Node> nodes) {
        this.nodes = nodes;
    }

    public Layer(ArrayList<Node> nodes, ActivationFunction actFunc) {
        this.nodes = nodes;
        this.func = actFunc;
    }

    public ArrayList<Node> getNodes() {
        return nodes;
    }

    public ActivationFunction getActFunc() {
        return func;
    }
    
}