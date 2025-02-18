package src.com.JakeKrayger.nn.layers;

import java.util.ArrayList;

import src.com.JakeKrayger.nn.components.*;
import src.com.JakeKrayger.nn.nodes.*;
import src.com.JakeKrayger.nn.Data;

public class InputLayer extends Layer {
    // public InputLayer(ArrayList<Node> nodes) {
    //     super(nodes);
    // }

    public InputLayer(Data data) {
        super(createNodes(data));
    }

    private static ArrayList<Node> createNodes(Data data) {
        ArrayList<Node> nodes = new ArrayList<>();
        for (Double d: data.getData()) {
            nodes.add(new InputNode(d));
        }

        return nodes;
    }
}