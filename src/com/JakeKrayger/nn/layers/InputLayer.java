package src.com.JakeKrayger.nn.layers;

import java.util.ArrayList;
import src.com.JakeKrayger.nn.components.*;
import src.com.JakeKrayger.nn.nodes.*;
import src.com.JakeKrayger.nn.Data;

public class InputLayer extends Layer {

    public InputLayer(Data data) {
        super(createNodes(data));
    }

    private static ArrayList<Node> createNodes(Data data) {
        ArrayList<Node> nodes = new ArrayList<>();
        for (int i = 0; i < data.getData().getNumCols(); i++) {
            nodes.add(new Node());
        }

        return nodes;
    }
}