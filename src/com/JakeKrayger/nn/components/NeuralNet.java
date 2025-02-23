package src.com.JakeKrayger.nn.components;

import java.util.ArrayList;

import org.ejml.simple.SimpleMatrix;

import src.com.JakeKrayger.nn.Data;
import src.com.JakeKrayger.nn.activation.*;
import src.com.JakeKrayger.nn.initialize.*;
import src.com.JakeKrayger.nn.layers.*;
import src.com.JakeKrayger.nn.training.loss.*;
import src.com.JakeKrayger.nn.training.optimizers.*;

public class NeuralNet {
    private ArrayList<Layer> layers;
    private Optimizer optimizer;
    private Loss loss;

    public ArrayList<Layer> getLayers() {
        return layers;
    }

    public void addLayer(Layer l) {
        if (this.layers != null) {
            // set forward connections and weights
            for (Node pNode : this.layers.get(this.layers.size() - 1).getNodes()) {
                for (Node nNode : l.getNodes()) {
                    pNode.setForwConnection(nNode);
                    if (l.getActFunc() instanceof ReLU) {
                        nNode.setWeights(new HeInit().initWeight(this.layers.get(this.layers.size() - 1)));
                    } else if (l.getActFunc() instanceof Sigmoid || l.getActFunc() instanceof Tanh) {
                        nNode.setWeights(new GlorotInit().initWeight(this.layers.get(this.layers.size() - 1), l));
                    }
                }
            }

            // set backward connections
            for (Node nNode : l.getNodes()) {
                for (Node pNode : this.layers.get(this.layers.size() - 1).getNodes()) {
                    nNode.setBackConnection(pNode);
                }
            }
        } else {
            layers = new ArrayList<>();
            for (Node nNode : l.getNodes()) {
                if (l.getActFunc() instanceof ReLU) {
                    nNode.setWeights(new HeInit().initWeight(l.getInputSize()));
                } else if (l.getActFunc() instanceof Sigmoid || l.getActFunc() instanceof Tanh) {
                    nNode.setWeights(new GlorotInit().initWeight(l.getInputSize(), l));
                }
            }
        }

        this.layers.add(l);

    }

    public void singleForwardPass(Data d, int batchSize) {
        SimpleMatrix dater = d.getData().transpose();
        SimpleMatrix firstBatch = new SimpleMatrix(dater.getRow(0));

        for (int i = 1; i < batchSize; i++) {
            firstBatch = firstBatch.concatRows(new SimpleMatrix(dater.getRow(i)));
        }

        // for (Node n: layers.getFirst().getNodes()) {
        // ActivationFunction af = layers.getFirst().getActFunc();
        // n.setValues();
        // }

        // System.out.println(firstBatch);

    }

    public void compile(Optimizer o, Loss l) {
        this.optimizer = o;
        this.loss = l;
    }

    // FORWARD PASS:
    // -> input new values
    // -> update values of nodes in hidden layer with activation function (repeat
    // for all hidden layers)
    // -> update values of nodes in output layer using the activation function

    // BACK PROPAGATION
    // -> compute output layer error
    // -> use loss function to get difference between predicted and actual values
    // -> compute error for hidden layers
    // -> pass error backward using weight connections and activation function
    // derivative
    // -> compute weight and bias updates
    // -> use error and learning rate to adjust weights and biases
    // -> update weights and biases
    // -> apply updates and prepare for next forward pass
    public void fit() {
    }

}