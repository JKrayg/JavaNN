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
        
        SimpleMatrix biases = new SimpleMatrix(l.getNumNeurons(), 1);
        biases.fill(0.01);
        l.setBiases(biases);


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

    public void singlePass() {
        MathUtils maths = new MathUtils();
        Layer L1 = layers.get(0);
        SimpleMatrix act = L1.getActFunc().execute(maths.weightedSum(singleBatch.getData(), L1));
        System.out.println("L1 activation matrix after " + L1.getActFunc().getClass().getSimpleName() + " function: \n" + act);
        L1.setActivations(act);
        for (int i = 1; i < layers.size(); i++) {
            Layer curr = layers.get(i);
            Layer prev = layers.get(i - 1);
            SimpleMatrix currAct = curr.getActFunc().execute(maths.weightedSum(prev, curr));
            System.out.println("L" + (i + 1) + " activation matrix after " + curr.getActFunc().getClass().getSimpleName() + " function: \n" + currAct);
            curr.setActivations(currAct);
        }

        getGradients(layers.get(layers.size() - 1));
    }

    public void getGradients(Layer currLayer) {
        // get gradient of loss wrt to output
        // use to get gradient wrt weights/biases of current layer <----|
        // use to get gradient wrt to previous layers activation        |
        //      and pass this to next layer                             |
        // repeat ------------------------------------------------------|
        Layer curr = currLayer;
        Layer prev = layers.get(layers.indexOf(curr) - 1);

        System.out.println("\ngradient of loss wrt to output:");
        SimpleMatrix gradientWrtOutput = curr.getActivations().minus(new SimpleMatrix(singleBatch.getLabels()));
        System.out.println(gradientWrtOutput);
        
        SimpleMatrix gradientWrtWeights = curr.outputGradientWeights(curr, prev, singleBatch.getLabels());
        curr.setGradientWeights(gradientWrtWeights);
        System.out.println("\ngradient of output wrt to weights:");
        System.out.println(gradientWrtWeights);

        SimpleMatrix gradientWrtBias = curr.outputGradientBias(curr, singleBatch.getLabels());
        curr.setGradientBiases(gradientWrtBias);
        System.out.println("\ngradient of output wrt to bias:");
        System.out.println(gradientWrtBias);

        System.out.println("gradient passed to previous layer:");
        // SimpleMatrix gradientWrtOutput = curr.getActivations().minus(new SimpleMatrix(singleBatch.getLabels()));
        SimpleMatrix gradientForNextLayer = gradientWrtOutput.mult(curr.getWeights().transpose());
        System.out.println(gradientWrtOutput.mult(curr.getWeights().transpose()));
    }

}