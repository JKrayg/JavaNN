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
    private Data training;
    private Data testing;

    public ArrayList<Layer> getLayers() {
        return layers;
    }

    public void setBatchSize(int batchSize) {
        this.batchSize = batchSize;
    }

    public void addLayer(Layer l) {
        ActivationFunction actFunc = l.getActFunc();
        SimpleMatrix biases = new SimpleMatrix(l.getNumNeurons(), 1);
        if (this.layers != null) {
            Layer prevLayer = this.layers.get(this.layers.size() - 1);
            if (actFunc instanceof ReLU) {
                l.setWeights(new HeInit().initWeight(prevLayer, l));
                biases.fill(0.1);
                l.setBiases(biases);
            } else {
                l.setWeights(new GlorotInit().initWeight(prevLayer, l));
                biases.fill(0.0);
                l.setBiases(biases);
            }
        } else {
            layers = new ArrayList<>();
            if (actFunc instanceof ReLU) {
                l.setWeights(new SimpleMatrix(new HeInit().initWeight(l.getInputSize(), l)));
                biases.fill(0.1);
                l.setBiases(biases);
            } else {
                l.setWeights(new SimpleMatrix(new GlorotInit().initWeight(l.getInputSize(), l)));
                biases.fill(0.0);
                l.setBiases(biases);
            }
        }


        this.layers.add(l);

    }

    public void compile(Data training, Data testing, Optimizer o, Loss l) {
        this.optimizer = o;
        this.loss = l;
        this.training = training;
        this.testing = testing;

        for (Layer lyr: layers) {
            lyr.setLoss(l);
        }
    }

    public void compile(Data singleBatch, Optimizer o, Loss l) {
        this.optimizer = o;
        this.loss = l;
        this.singleBatch = singleBatch;

        for (Layer lyr: layers) {
            lyr.setLoss(l);
        }
    }

    public void singlePass() {
        MathUtils maths = new MathUtils();
        Layer L1 = layers.get(0);
        SimpleMatrix zL1 = maths.weightedSum(singleBatch.getData(), L1);
        SimpleMatrix act = L1.getActFunc().execute(zL1);
        System.out.println("L1 activation matrix after " + L1.getActFunc().getClass().getSimpleName() + " function: \n" + act);
        L1.setPreActivations(zL1);
        L1.setActivations(act);
        
        for (int i = 1; i < layers.size(); i++) {
            Layer curr = layers.get(i);
            Layer prev = layers.get(i - 1);
            SimpleMatrix z = maths.weightedSum(prev, curr);
            SimpleMatrix currAct = curr.getActFunc().execute(z);
            System.out.println("L" + (i + 1) + " activation matrix after " + curr.getActFunc().getClass().getSimpleName() + " function: \n" + currAct);
            // System.out.println("L" + (i + 1) + " weighted sums (zL" + (i + 1) + "):\n" + z);
            curr.setPreActivations(z);
            curr.setActivations(currAct);

            if (curr instanceof Output) {
                ((Output) curr).setLabels(singleBatch.getLabels());
            }
        }

        Output outLayer = (Output) layers.get(layers.size() - 1);
        SimpleMatrix gradientWrtOutput = outLayer.getLoss().gradient(outLayer, outLayer.getLabels());
        getGradients(outLayer, gradientWrtOutput);
    }

    public void getGradients(Layer currLayer, SimpleMatrix gradient) {
        // get gradient of loss wrt to output
        // use to get gradient wrt weights/biases of current layer <----|
        // use to get gradient wrt to previous layers activation        |
        //      and pass this to next layer                             |
        // repeat ------------------------------------------------------|

        // System.out.println("gwrtO:");
        // System.out.println(gradient);

        // System.out.println("curr weights mult:");
        // System.out.println(gradient.mult(currLayer.getWeights().transpose()));

        Layer curr = currLayer;
        Layer prev = layers.get(layers.indexOf(curr) - 1);

        if (currLayer instanceof Output) {
            Output out = (Output) curr;
            SimpleMatrix gradientWrtWeights = out.gradientWeights(prev, gradient);
            SimpleMatrix gradientWrtBias = out.gradientBias(curr);
            curr.setGradientWeights(gradientWrtWeights);
            curr.setGradientBiases(gradientWrtBias);
        } else {
            SimpleMatrix gradientWrtWeights = currLayer.gradientWeights(prev, gradient);
            SimpleMatrix gradientWrtBias = currLayer.gradientBias(gradient);
            curr.setGradientWeights(gradientWrtWeights);
            curr.setGradientBiases(gradientWrtBias);
        }

        // System.out.println("gradient wrt Output:");
        // System.out.println(gradient);

        // System.out.println("output weights transposed:");
        // System.out.println(currLayer.getWeights().transpose());

        // System.out.println("gradient wrt output dot output weights transposed:");
        // System.out.println(gradient.mult(currLayer.getWeights().transpose()));
        


                        // if (layers.get(layers.indexOf(curr) - 1) != null) {
                        //     System.out.println("gradient:");
                        //     System.out.println(gradient);
                        //     System.out.println("currLayer.getWeights().transpose()");
                        //     System.out.println(currLayer.getWeights().transpose());
                        //     System.out.println("gradient wrt output dot output weights transposed:");
                        //     System.out.println(gradient.mult(currLayer.getWeights().transpose()));

                        //     SimpleMatrix s = prev.getActFunc().gradient(prev, gradient.mult(currLayer.getWeights().transpose()));
                        //     System.out.println("S:");
                        //     System.out.println(s);
                        //     getGradients(prev, s);
                        // }


        // System.out.println("gradient of loss wrt output:");
        // System.out.println(currLayer.getGradient());
        // Layer curr = currLayer;
        // Layer prev = layers.get(layers.indexOf(curr) - 1);

        // System.out.println("\ngradient of loss wrt to output:");
        // SimpleMatrix gradientWrtOutput = curr.getActivations().minus(new SimpleMatrix(singleBatch.getLabels()));
        // System.out.println(gradientWrtOutput);
        
        // SimpleMatrix gradientWrtWeights = curr.outputGradientWeights(curr, prev, singleBatch.getLabels());
        // curr.setGradientWeights(gradientWrtWeights);
        // System.out.println("\ngradient of output wrt to weights:");
        // System.out.println(gradientWrtWeights);

        // SimpleMatrix gradientWrtBias = curr.outputGradientBias(curr, singleBatch.getLabels());
        // curr.setGradientBiases(gradientWrtBias);
        // System.out.println("\ngradient of output wrt to bias:");
        // System.out.println(gradientWrtBias);

        // System.out.println("gradient passed to previous layer:");
        // // SimpleMatrix gradientWrtOutput = curr.getActivations().minus(new SimpleMatrix(singleBatch.getLabels()));
        // SimpleMatrix gradientForNextLayer = gradientWrtOutput.mult(curr.getWeights().transpose());
        // System.out.println(gradientWrtOutput.mult(curr.getWeights().transpose()));
    }

}