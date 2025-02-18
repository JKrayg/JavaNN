package src.com.JakeKrayger.nn.components;

import java.util.ArrayList;

public class Layer {
    private ArrayList<Node> nodes;

    public Layer(ArrayList<Node> nodes) {
        this.nodes = nodes;
    }

    public ArrayList<Node> getNodes() {
        return nodes;
    }

    public void setNodes(ArrayList<Node> nodes) {
        this.nodes = nodes;
    }
}