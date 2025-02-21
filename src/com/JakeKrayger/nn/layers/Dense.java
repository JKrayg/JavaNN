package src.com.JakeKrayger.nn.layers;

import java.util.ArrayList;

import src.com.JakeKrayger.nn.activation.ActivationFunction;
import src.com.JakeKrayger.nn.components.*;

// OutputLayer and Dense are pretty much the same class
public class Dense extends Layer {

    public Dense(int numNodes, ActivationFunction actFunc) {
        super(createNodes(numNodes), actFunc);
    }

    private static ArrayList<Node> createNodes(int numNodes) {
        ArrayList<Node> nodes = new ArrayList<>();
        for (int i = 0; i < numNodes; i++) {
            nodes.add(new Node(0, 0));
            
        }

        return nodes;
    }
}