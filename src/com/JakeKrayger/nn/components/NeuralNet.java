package src.com.JakeKrayger.nn.components;

import java.util.ArrayList;

import org.ejml.simple.SimpleMatrix;

import src.com.JakeKrayger.nn.Data;
import src.com.JakeKrayger.nn.activation.*;
import src.com.JakeKrayger.nn.initialize.*;
import src.com.JakeKrayger.nn.layers.*;
import src.com.JakeKrayger.nn.training.loss.*;
import src.com.JakeKrayger.nn.training.optimizers.*;
import src.com.JakeKrayger.nn.utils.MathUtils;

public class NeuralNet {
    private ArrayList<Layer> layers;
    private Optimizer optimizer;
    private Loss loss;
    private int batchSize;
    private Data singleBatch;

    public ArrayList<Layer> getLayers() {
        return layers;
    }

    public void setBatchSize(int batchSize) {
        this.batchSize = batchSize;
    }

    public void addLayer(Layer l) {
        ActivationFunction actFunc = l.getActFunc();
        if (this.layers != null) {
            Layer prevLayer = this.layers.get(this.layers.size() - 1);
            if (actFunc instanceof ReLU) {
                l.setWeights(new HeInit().initWeight(prevLayer, l));
            } else {
                l.setWeights(new GlorotInit().initWeight(prevLayer, l));
            }
        } else {
            layers = new ArrayList<>();
            if (actFunc instanceof ReLU) {
                l.setWeights(new SimpleMatrix(new HeInit().initWeight(l.getInputSize(), l)));
            } else {
                l.setWeights(new SimpleMatrix(new GlorotInit().initWeight(l.getInputSize(), l)));
            }
        }

        this.layers.add(l);

    }

    public void singleForwardPass(Data d, int batchSize) {
        SimpleMatrix dater = d.getData().transpose();
        SimpleMatrix firstBatch = new SimpleMatrix(dater.getRow(0));
        SimpleMatrix values = new SimpleMatrix(dater.getNumRows(), 0);

        for (int i = 1; i < batchSize; i++) {
            firstBatch = firstBatch.concatRows(new SimpleMatrix(dater.getRow(i)));
        }

        System.out.println(firstBatch);

        // System.out.println(firstBatch);

    }

    public void compile(Data singleBatch, Optimizer o, Loss l) {
        this.optimizer = o;
        this.loss = l;
        this.singleBatch = singleBatch;
    }

    public void singleForwardProp(Data data) {
        MathUtils maths = new MathUtils();
        Layer L1 = layers.get(0);
        SimpleMatrix act = L1.getActFunc().execute(maths.weightedSum(data.getData(), L1));
        System.out.println("L1 activation matrix after " + L1.getActFunc().getClass().getSimpleName() + " function: \n" + act);
        L1.setActivations(act);
        for (int i = 1; i < layers.size(); i++) {
            Layer curr = layers.get(i);
            Layer prev = layers.get(i - 1);
            SimpleMatrix currAct = curr.getActFunc().execute(maths.weightedSum(prev, curr));
            System.out.println("L" + (i + 1) + " activation matrix after " + curr.getActFunc().getClass().getSimpleName() + " function: \n" + currAct);
            curr.setActivations(currAct);
        }

        getOutputGradients(data);
    }

    public void getOutputGradients(Data data) {
        // get gradient of loss wrt to output
        // use to get gradient of loss wrt weights/biases
        // use to get gradient of loss wrt to previous layers activation
        Layer out = layers.get(layers.size() - 1);
        Layer prev = layers.get(layers.size() - 2);
        System.out.println("loss:");
        System.out.println(loss.execute(out.getActivations(), data.getLabels()));

        System.out.println("\ngradient of loss wrt to output:");
        System.out.println(out.getActivations().minus(new SimpleMatrix(data.getLabels())));
        
        SimpleMatrix gradientWrtWeights = loss.outputGradientWeights(out, prev, data.getLabels());
        out.setGradientWeights(gradientWrtWeights);
        System.out.println("\ngradient of output wrt to weights:");
        System.out.println(gradientWrtWeights);

        // System.out.println("d2 initialized weights:");
        // System.out.println(prev.getWeights());

        // System.out.println("d3 initialized weights:");
        // System.out.println(out.getWeights());

        // System.out.println("d3 updated weights:");
        // out.updateWeights(gradientWrtWeights, 0.1);
        // System.out.println(out.getWeights());

        SimpleMatrix gradientWrtBias = loss.outputGradientBias(out, data.getLabels());
        out.setGradientBiases(gradientWrtBias);
        System.out.println("\ngradient of output wrt to bias:");
        System.out.println(gradientWrtBias);

        System.out.println("gradient passed to previous layer:");
        System.out.println(gradientWrtBias.mult(out.getWeights().transpose()));

        // System.out.println("d3 initialized bias:");
        // System.out.println(out.getBias());

        // System.out.println("d3 updated biases:");
        // out.updateBiases(gradientWrtBias, 0.1);
        // System.out.println(out.getBias());
    }

}