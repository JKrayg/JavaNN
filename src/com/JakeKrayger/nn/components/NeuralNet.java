package src.com.JakeKrayger.nn.components;

import java.util.ArrayList;
import java.util.Collections;

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
    private double learningRate;
    private MathUtils maths = new MathUtils();

    public ArrayList<Layer> getLayers() {
        return layers;
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



    public void compile(Optimizer o, Loss l, double learningRate) {
        this.optimizer = o;
        this.lossFunc = l;
        this.learningRate = learningRate;

        for (Layer lyr : layers) {
            lyr.setLoss(l);
        }
    }



    public void miniBatchFit(SimpleMatrix train, SimpleMatrix test, int batchSize, int epochs) {
        // shuffle data and get new batches of size batchSize for each epoch
        for (int i = 0; i < epochs; i++) {
            ArrayList<SimpleMatrix> shuffled = new ArrayList<>();
            ArrayList<SimpleMatrix> batchesData = new ArrayList<>();
            ArrayList<SimpleMatrix> batchesLabels = new ArrayList<>();
            for (int j = 0; j < train.getNumRows(); j++) {
                shuffled.add(train.getRow(j));
            }
            Collections.shuffle(shuffled);

            // need to handle last batch
            for (int k = 0; k < shuffled.size() / batchSize; k++) {
                SimpleMatrix currBatch = new SimpleMatrix(batchSize, train.getNumCols());
                int count = 0;
                for (int p = k * batchSize; p < k * batchSize + batchSize; p++) {
                    currBatch.setRow(count, shuffled.get(p));
                    count += 1;
                }
                batchesData.add(currBatch.extractMatrix(0, batchSize, 0, train.getNumCols() - 1));
                batchesLabels.add(currBatch.extractVector(false, train.getNumCols() - 1));
            }

            // do below for each batch
            for (int v = 0; v < batchesData.size(); v++) {
                forwardPass(batchesData.get(v), batchesLabels.get(v));
                backprop(batchesData.get(v), batchesLabels.get(v));
            }

            // get loss

            SimpleMatrix data = train.extractMatrix(0, train.getNumRows(), 0, train.getNumCols() - 1);
            SimpleMatrix labels = train.extractVector(false, train.getNumCols() - 1);

            forwardPass(data, labels);
            // System.out.println("LOSS: " + loss);
        }
        double[] lab = new double[1];
        lab[0] = test.get(3, test.getNumCols() - 1);

        SimpleMatrix testData = test.extractMatrix(0, test.getNumRows(), 0, test.getNumCols() - 1);
        SimpleMatrix testLabels = test.extractVector(false, test.getNumCols() - 1);

        // test
        forwardPass(testData, testLabels);
        Output outLayer = (Output) layers.get(layers.size() - 1);
        System.out.println("Prediction          : True Value");
        for (int h = 0; h < testData.getNumRows(); h++) {
            System.out.print(outLayer.getActivations().get(h));
            System.out.print(" : " + testLabels.get(h));
            System.out.println();
        }
        
        
    }



    public void batchFit(SimpleMatrix trainData, int epochs) {
        for (int i = 0; i < epochs; i++) {
            SimpleMatrix data = trainData.extractMatrix(0, trainData.getNumRows(), 0, trainData.getNumCols() - 1);
            SimpleMatrix labels = trainData.extractVector(false, trainData.getNumCols() - 1);

            forwardPass(data, labels);
            System.out.println("LOSS: " + loss);
            backprop(data, labels);
        }
    }



    public void forwardPass(SimpleMatrix data, SimpleMatrix labels) {
        Layer L1 = layers.get(0);
        SimpleMatrix zL1 = maths.weightedSum(data, L1);
        SimpleMatrix act = L1.getActFunc().execute(zL1);
        L1.setPreActivations(zL1);
        L1.setActivations(act);

        for (int q = 1; q < layers.size(); q++) {
            Layer curr = layers.get(q);
            Layer prev = layers.get(q - 1);
            SimpleMatrix z = maths.weightedSum(prev, curr);
            SimpleMatrix currAct = curr.getActFunc().execute(z);
            curr.setPreActivations(z);
            curr.setActivations(currAct);

            if (curr instanceof Output) {
                ((Output) curr).setLabels(labels);
            }
        }

        Output outLayer = (Output) layers.get(layers.size() - 1);
        this.loss = outLayer.getLoss().execute(outLayer.getActivations(), labels);
    }



    public void backprop(SimpleMatrix data, SimpleMatrix labels) {
        Output outLayer = (Output) layers.get(layers.size() - 1);
        SimpleMatrix gradientWrtOutput = outLayer.getLoss().gradient(outLayer, outLayer.getLabels());
        getGradients(outLayer, gradientWrtOutput, data);

        for (Layer l : layers) {
            l.updateWeights(l.getGradientWeights(), learningRate);
            l.updateBiases(l.getGradientBias(), learningRate);
        }
    }



    public void getGradients(Layer currLayer, SimpleMatrix gradient, SimpleMatrix data) {
        Layer curr = currLayer;

        if (currLayer instanceof Output) {
            Output out = (Output) curr;
            Layer prev = layers.get(layers.indexOf(curr) - 1);
            SimpleMatrix gradientWrtWeights = out.gradientWeights(prev, gradient);
            SimpleMatrix gradientWrtBias = out.gradientBias(curr, gradient);
            curr.setGradientWeights(gradientWrtWeights);
            curr.setGradientBiases(gradientWrtBias);
        } else {
            Layer prev;
            if (layers.indexOf(curr) > 0) {
                prev = layers.get(layers.indexOf(curr) - 1);
            } else {
                prev = new Layer();
                prev.setActivations(data);
            }

            SimpleMatrix gradientWrtWeights = currLayer.gradientWeights(prev, gradient);
            SimpleMatrix gradientWrtBias = currLayer.gradientBias(gradient);
            curr.setGradientWeights(gradientWrtWeights);
            curr.setGradientBiases(gradientWrtBias);
        }

        if (layers.indexOf(curr) > 0) {
            Layer prev = layers.get(layers.indexOf(curr) - 1);
            SimpleMatrix next = prev.getActFunc().gradient(prev, gradient.mult(currLayer.getWeights().transpose()));
            getGradients(prev, next, data);
        }
    }

}