package src.com.JakeKrayger.nn.components;
import java.util.ArrayList;

import src.com.JakeKrayger.nn.initialize.GlorotInit;

public class NeuralNet {
    private ArrayList<Layer> layers;

    // need to initialize weights (a list of weights with length = # of nodes in previous layer,
    //                             and stored in next layers nodes)
    // may need to modify this again for back propagtion
    public void addLayer(Layer l) {
        if (this.layers != null) {
            // set forward connections
            for (Node pNode: this.layers.get(this.layers.size() - 1).getNodes()) {
                for (Node nNode: l.getNodes()) {
                    pNode.setForwConnection(nNode);
                    nNode.addWeight(new GlorotInit().initWeight(this.layers.get(this.layers.size() - 1), l, nNode.getActFunc()));
                }
            }

            // set backward connections
            for (Node nNode: l.getNodes()) {
                for (Node pNode: this.layers.get(this.layers.size() - 1).getNodes()) {
                    nNode.setBackConnection(pNode);
                }
            }
        } else {
            layers = new ArrayList<>();
        }

        this.layers.add(l);
        
    }

    public ArrayList<Layer> getLayers() {
        return layers;
    }

}