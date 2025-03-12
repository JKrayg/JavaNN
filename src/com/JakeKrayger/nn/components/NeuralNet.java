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
    private Loss lossFunc;
    private double loss;
    private int batchSize;
    private Data singleBatch;
    private Data training;
    private Data testing;
    private double learningRate;

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
        this.lossFunc = l;
        this.training = training;
        this.testing = testing;

        for (Layer lyr : layers) {
            lyr.setLoss(l);
        }
    }

    public void compile(Data singleBatch, Optimizer o, Loss l, double learningRate) {
        this.optimizer = o;
        this.lossFunc = l;
        this.singleBatch = singleBatch;
        this.learningRate = learningRate;

        for (Layer lyr : layers) {
            lyr.setLoss(l);
        }
    }

    public void singlePass() {
        MathUtils maths = new MathUtils();
        Layer L1 = layers.get(0);
        SimpleMatrix zL1 = maths.weightedSum(singleBatch.getData(), L1);
        SimpleMatrix act = L1.getActFunc().execute(zL1);
        // System.out.println("L1 pre-activation:\n" + zL1);
        // System.out.println("L1 activation matrix after " + L1.getActFunc().getClass().getSimpleName() + " function: \n" + act);
        L1.setPreActivations(zL1);
        L1.setActivations(act);

        for (int i = 1; i < layers.size(); i++) {
            Layer curr = layers.get(i);
            Layer prev = layers.get(i - 1);
            SimpleMatrix z = maths.weightedSum(prev, curr);
            SimpleMatrix currAct = curr.getActFunc().execute(z);
            // System.out.println("L" + (i + 1) + " pre-activation (zL" + (i + 1) + "):\n" + z);
            // System.out.println("L" + (i + 1) + " activation matrix after "
                    // + curr.getActFunc().getClass().getSimpleName() + " function: \n" + currAct);
            curr.setPreActivations(z);
            curr.setActivations(currAct);

            if (curr instanceof Output) {
                ((Output) curr).setLabels(singleBatch.getLabels());
            }
        }

        Output outLayer = (Output) layers.get(layers.size() - 1);
        SimpleMatrix gradientWrtOutput = outLayer.getLoss().gradient(outLayer, outLayer.getLabels());
        this.loss = outLayer.getLoss().execute(outLayer.getActivations(), singleBatch.getLabels());
        System.out.println("LOSS: " + loss);
        getGradients(outLayer, gradientWrtOutput);
        int count = 0;

        for (Layer l: layers) {
            count += 1;
            // System.out.println("L" + count + " weights before update:");
            // System.out.println(l.getWeights());
            // System.out.println("L" + count + " bias before update:");
            // System.out.println(l.getBias());
            l.updateWeights(l.getGradientWeights(), learningRate);
            l.updateBiases(l.getGradientBias(), learningRate);
            // System.out.println("L" + count + " weights after update:");
            // System.out.println(l.getWeights());
            // System.out.println("L" + count + " bias after update:");
            // System.out.println(l.getBias());
        }
    }

    public void getGradients(Layer currLayer, SimpleMatrix gradient) {
        // get gradient of loss wrt to output
        // use to get gradient wrt weights/biases of current layer <----|
        // use to get gradient wrt to previous layers activation |
        // and pass this to next layer |
        // repeat ------------------------------------------------------|

        // System.out.println("gwrtO:");
        // System.out.println(gradient);

        // System.out.println("curr weights mult:");
        // System.out.println(gradient.mult(currLayer.getWeights().transpose()));

        Layer curr = currLayer;

        // System.out.println("gradient of loss wrt output for " + currLayer.getClass().getSimpleName() + " layer:");
        // System.out.println(gradient);
        // System.out.println(currLayer.getClass().getSimpleName() + " weights transposed:");
        // System.out.println(currLayer.getWeights().transpose());
        // System.out.println("gradient wrt output dot " + currLayer.getClass().getSimpleName() + " weights transposed:");
        // System.out.println(gradient.mult(currLayer.getWeights().transpose()));

        if (currLayer instanceof Output) {
            Output out = (Output) curr;
            Layer prev = layers.get(layers.indexOf(curr) - 1);
            SimpleMatrix gradientWrtWeights = out.gradientWeights(prev, gradient);
            // System.out.println("gradient of loss wrt output weights:");
            // System.out.println(gradientWrtWeights);
            SimpleMatrix gradientWrtBias = out.gradientBias(curr, gradient);
            // System.out.println("gradient of loss wrt output bias:");
            // System.out.println(gradientWrtBias);
            curr.setGradientWeights(gradientWrtWeights);
            curr.setGradientBiases(gradientWrtBias);
        } else {
            Layer prev;
            if (layers.indexOf(curr) > 0) {
                prev = layers.get(layers.indexOf(curr) - 1);
            } else {
                prev = new Layer();
                prev.setActivations(singleBatch.getData());
            }
            
            SimpleMatrix gradientWrtWeights = currLayer.gradientWeights(prev, gradient);
            // System.out.println("gradient of loss wrt " + currLayer.getClass().getSimpleName() + " weights:");
            // System.out.println(gradientWrtWeights);
            SimpleMatrix gradientWrtBias = currLayer.gradientBias(gradient);
            // System.out.println("gradient of loss wrt " + currLayer.getClass().getSimpleName() + " bias:");
            // System.out.println(gradientWrtBias);
            curr.setGradientWeights(gradientWrtWeights);
            curr.setGradientBiases(gradientWrtBias);
        }

        if (layers.indexOf(curr) > 0) {
            Layer prev = layers.get(layers.indexOf(curr) - 1);
            SimpleMatrix next = prev.getActFunc().gradient(prev, gradient.mult(currLayer.getWeights().transpose()));
            // System.out.println("gradient of loss wrt previous layers activation:");
            // System.out.println(next);
            getGradients(prev, next);
        }
    }

    // public void getGradients(Layer currLayer, SimpleMatrix gradient) {
    //     Layer curr = currLayer;
    //     Layer prev = layers.get(layers.indexOf(curr) - 1);

    //     if (currLayer instanceof Output) {
    //         Output out = (Output) curr;
    //         SimpleMatrix gradientWrtWeights = out.gradientWeights(prev, gradient);
    //         SimpleMatrix gradientWrtBias = out.gradientBias(curr);
    //         curr.setGradientWeights(gradientWrtWeights);
    //         curr.setGradientBiases(gradientWrtBias);
    //     } else {
    //         SimpleMatrix gradientWrtWeights = currLayer.gradientWeights(prev, gradient);
    //         SimpleMatrix gradientWrtBias = currLayer.gradientBias(gradient);
    //         curr.setGradientWeights(gradientWrtWeights);
    //         curr.setGradientBiases(gradientWrtBias);
    //     }

    //     if (layers.get(layers.indexOf(curr) - 1) != null) {
    //         SimpleMatrix next = prev.getActFunc().gradient(curr, gradient.mult(currLayer.getWeights().transpose()));
    //         getGradients(prev, next);
    //     }
    // }

}