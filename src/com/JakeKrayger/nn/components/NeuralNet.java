package src.com.JakeKrayger.nn.components;

import java.util.ArrayList;

import src.com.JakeKrayger.nn.activation.*;
import src.com.JakeKrayger.nn.initialize.*;
import src.com.JakeKrayger.nn.layers.InputLayer;
import src.com.JakeKrayger.nn.training.loss.Loss;
import src.com.JakeKrayger.nn.training.optimizers.Optimizer;

public class NeuralNet {
    private ArrayList<Layer> layers;

    public void addLayer(Layer l) {
        if (this.layers != null) {
            // set forward connections
            for (Node pNode: this.layers.get(this.layers.size() - 1).getNodes()) {
                for (Node nNode: l.getNodes()) {
                    pNode.setForwConnection(nNode);
                    if (l.getActFunc() instanceof ReLU) {
                        nNode.setWeights(new HeInit().initWeight(this.layers.get(this.layers.size() - 1)));
                    } else if (l.getActFunc() instanceof Sigmoid || l.getActFunc() instanceof Tanh) {
                        nNode.setWeights(new GlorotInit().initWeight(this.layers.get(this.layers.size() - 1), l));
                    }
                    
                    // nNode.addWeight(new GlorotInit().initWeight(this.layers.get(this.layers.size() - 1), l, l.getActFunc()));
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

    public void singleForwardPass() {
        for (int i = 0; i < layers.size(); i++) {
            if (!(layers.get(i) instanceof InputLayer)) {
                
            }
        }
    }

    public void compile(Optimizer o, Loss l) {
        
    }

    // FORWARD PASS:
    // -> input new values
    // -> update values of nodes in hidden layer with activation function (repeat for all hidden layers)
    // -> update values of nodes in output layer using the activation function

    // BACK PROPAGATION
    // -> compute output layer error
    //      -> use loss function to get difference between predicted and actual values
    // -> compute error for hidden layers
    //      -> pass error backward using weight connections and activation function derivative
    // -> compute weight and bias updates
    //      -> use error and learning rate to adjust weights and biases
    // -> update weights and biases
    //      -> apply updates and prepare for next forward pass
    public void fit() {}

    public ArrayList<Layer> getLayers() {
        return layers;
    }

}