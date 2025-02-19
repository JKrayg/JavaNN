package src.com.JakeKrayger.nn.components;

import java.util.ArrayList;
import src.com.JakeKrayger.nn.initialize.GlorotInit;
import src.com.JakeKrayger.nn.training.loss.Loss;
import src.com.JakeKrayger.nn.training.optimizers.Optimizer;

public class NeuralNet {
    private ArrayList<Layer> layers;

    // may need to modify this again for back propagtion
    public void addLayer(Layer l) {
        if (this.layers != null) {
            // set forward connections
            for (Node pNode: this.layers.get(this.layers.size() - 1).getNodes()) {
                for (Node nNode: l.getNodes()) {
                    pNode.setForwConnection(nNode);
                    nNode.addWeight(new GlorotInit().initWeight(this.layers.get(this.layers.size() - 1), l, l.getActFunc()));
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

    public void compile(Optimizer o, Loss l) {

    }

    public ArrayList<Layer> getLayers() {
        return layers;
    }

}