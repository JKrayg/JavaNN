package src.com.JakeKrayger.nn.layers;

import java.util.ArrayList;

import org.ejml.simple.SimpleMatrix;

import src.com.JakeKrayger.nn.activation.ActivationFunction;
import src.com.JakeKrayger.nn.components.*;

// OutputLayer and Dense are pretty much the same class
public class OutputLayer extends Layer {

    public OutputLayer(int numNodes, ActivationFunction actFunc) {
        super(createNodes(numNodes), actFunc);
    }

    private static ArrayList<Node> createNodes(int numNodes) {
        ArrayList<Node> nodes = new ArrayList<>();
        for (int i = 0; i < numNodes; i++) {
            nodes.add(new Node(0));
            
        }

        return nodes;
    }

}