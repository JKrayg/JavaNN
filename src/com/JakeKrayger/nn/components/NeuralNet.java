package src.com.JakeKrayger.nn.components;

import java.util.ArrayList;
import java.util.Collections;

import org.ejml.simple.SimpleMatrix;
import src.com.JakeKrayger.nn.activation.*;
import src.com.JakeKrayger.nn.initialize.*;
import src.com.JakeKrayger.nn.layers.*;
import src.com.JakeKrayger.nn.training.loss.*;
import src.com.JakeKrayger.nn.training.metrics.Metrics;
import src.com.JakeKrayger.nn.training.optimizers.*;
import src.com.JakeKrayger.nn.utils.MathUtils;

public class NeuralNet {
    private ArrayList<Layer> layers;
    private Optimizer optimizer;
    private Metrics metrics;
    private Loss lossFunc;
    private double loss;
    private double valLoss;
    private double learningRate;
    private int numClasses;
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

    public void compile(Optimizer o, Loss l, Metrics m) {
        this.optimizer = o;
        this.lossFunc = l;
        this.learningRate = o.getLearningRate();
        this.metrics = m;

        for (Layer lyr : layers) {
            if (lyr instanceof Output) {
                this.numClasses = lyr.getNumNeurons();
                lyr.setLoss(l);
            }

            if (optimizer instanceof Adam) {
                SimpleMatrix weightsO = new SimpleMatrix(lyr.getWeights().getNumRows(), lyr.getWeights().getNumCols());
                SimpleMatrix biasO = new SimpleMatrix(lyr.getBias().getNumRows(), lyr.getBias().getNumCols());
                lyr.setWeightsMomentum(weightsO);
                lyr.setWeightsVariance(weightsO);
                lyr.setBiasesMomentum(biasO);
                lyr.setBiasesVariance(biasO);
            }
        }
    }

    public void miniBatchFit(SimpleMatrix train, SimpleMatrix test, SimpleMatrix validation, int batchSize, int epochs) {
        // shuffle data and get new batches of size batchSize for each epoch
        for (int i = 0; i < epochs; i++) {
            ArrayList<SimpleMatrix> shuffled = new ArrayList<>();
            ArrayList<SimpleMatrix> batchesData = new ArrayList<>();
            ArrayList<SimpleMatrix> batchesLabels = new ArrayList<>();
            for (int j = 0; j < train.getNumRows(); j++) {
                shuffled.add(train.getRow(j));
            }
            Collections.shuffle(shuffled);

            for (int k = 0; k < shuffled.size() / batchSize; k++) {
                SimpleMatrix currBatch = new SimpleMatrix(batchSize, train.getNumCols());
                int count = 0;
                for (int p = k * batchSize; p < k * batchSize + batchSize; p++) {
                    currBatch.setRow(count, shuffled.get(p));
                    count += 1;
                }
                batchesData.add(currBatch.extractMatrix(
                        0, currBatch.getNumRows(), 0, currBatch.getNumCols() - (numClasses > 2 ? numClasses : 1)));
                batchesLabels.add(currBatch.extractMatrix(
                        0, currBatch.getNumRows(), currBatch.getNumCols() - (numClasses > 2 ? numClasses : 1),
                        currBatch.getNumCols()));
            }

            // last batch - find a better way
            if (shuffled.size() % batchSize > 0) {
                SimpleMatrix lastBatch = new SimpleMatrix(shuffled.size() % batchSize, train.getNumCols());
                int count = 0;
                for (int m = batchSize * (shuffled.size() / batchSize); m < shuffled.size(); m++) {
                    lastBatch.setRow(count, shuffled.get(m));
                    count += 1;
                }
                batchesData.add(lastBatch.extractMatrix(
                        0, lastBatch.getNumRows(), 0, lastBatch.getNumCols() - (numClasses > 2 ? numClasses : 1)));
                batchesLabels.add(lastBatch.extractMatrix(
                        0, lastBatch.getNumRows(), lastBatch.getNumCols() - (numClasses > 2 ? numClasses : 1),
                        lastBatch.getNumCols()));
            }

            // do below for each batch
            for (int v = 0; v < batchesData.size(); v++) {
                forwardPass(batchesData.get(v), batchesLabels.get(v));
                backprop(batchesData.get(v), batchesLabels.get(v));

                if (optimizer instanceof Adam) {
                    ((Adam) optimizer).updateCount();
                }
            }

            this.loss = loss(train);
            this.valLoss = loss(validation);
            System.out.println("loss: " + loss + " - val loss: " + valLoss);
        }
        // SimpleMatrix testData = test.extractMatrix(0, test.getNumRows(), 0,
        //         test.getNumCols() - (numClasses > 2 ? numClasses : 1));
        // SimpleMatrix testLabels = test.extractMatrix(0, test.getNumRows(),
        //         test.getNumCols() - (numClasses > 2 ? numClasses : 1), test.getNumCols());

        // forwardPass(testData, testLabels);
        // Output outLayer = (Output) layers.get(layers.size() - 1);
        // // System.out.println("Prediction : one hot label");
        // // for (int h = 0; h < testData.getNumRows(); h++) {
        // //     System.out.print(outLayer.getActivations().getRow(h).concatColumns(testLabels.getRow(h)));
        // // }

        // System.out.println("Prediction : True Value");
        //  for (int h = 0; h < testData.getNumRows(); h++) {
        //      System.out.print(outLayer.getActivations().get(h));
        //      System.out.print(" : " + testLabels.get(h));
        //      System.out.println();
        //  }

        // test
        System.out.println();
        System.out.println("train metrics: ");
        metrics(train);
        System.out.println("test metrics: ");
        metrics(test);

    }

    public void batchFit(SimpleMatrix train, SimpleMatrix test, SimpleMatrix validation, int epochs) {
        for (int i = 0; i < epochs; i++) {
            SimpleMatrix data = train.extractMatrix(
                    0, train.getNumRows(), 0,
                    train.getNumCols() - (numClasses > 2 ? numClasses : 1));
            SimpleMatrix labels = train.extractMatrix(
                    0, train.getNumRows(), train.getNumCols() - (numClasses > 2 ? numClasses : 1), train.getNumCols());

            // System.out.println(data);
            // System.out.println(labels);

            forwardPass(data, labels);
            backprop(data, labels);

            if (optimizer instanceof Adam) {
                ((Adam) optimizer).updateCount();
            }

            this.loss = loss(train);
            this.valLoss = loss(validation);
            System.out.println("loss: " + loss + " - val loss: " + valLoss);
        }

        System.out.println("train metrics: ");
        metrics(train);
        System.out.println("test metrics: ");
        metrics(test);
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
    }


    public void metrics(SimpleMatrix test) {
        SimpleMatrix testData = test.extractMatrix(
                0, test.getNumRows(), 0, test.getNumCols() - (numClasses > 2 ? numClasses : 1));
        SimpleMatrix testLabels = test.extractMatrix(
                0, test.getNumRows(), test.getNumCols() - (numClasses > 2 ? numClasses : 1), test.getNumCols());

        forwardPass(testData, testLabels);
        Output outLayer = (Output) layers.get(layers.size() - 1);
        metrics.getMetrics(outLayer.getActivations(), testLabels);
    }


    // public void loss(SimpleMatrix train) {
    //     SimpleMatrix trainData = train.extractMatrix(
    //             0, train.getNumRows(), 0, train.getNumCols() - (numClasses > 2 ? numClasses : 1));
    //     SimpleMatrix trainLabels = train.extractMatrix(
    //             0, train.getNumRows(), train.getNumCols() - (numClasses > 2 ? numClasses : 1), train.getNumCols());

    //     forwardPass(trainData, trainLabels);
    //     Output outLayer = (Output) layers.get(layers.size() - 1);
    //     this.loss = outLayer.getLoss().execute(outLayer.getActivations(), trainLabels);
    //     System.out.print("loss: " + loss + " - ");
    // }

    public double loss(SimpleMatrix d) {
        SimpleMatrix data = d.extractMatrix(
                0, d.getNumRows(), 0, d.getNumCols() - (numClasses > 2 ? numClasses : 1));
        SimpleMatrix labels = d.extractMatrix(
                0, d.getNumRows(), d.getNumCols() - (numClasses > 2 ? numClasses : 1), d.getNumCols());

        forwardPass(data, labels);
        Output outLayer = (Output) layers.get(layers.size() - 1);
        return outLayer.getLoss().execute(outLayer.getActivations(), labels);
        // this.loss = outLayer.getLoss().execute(outLayer.getActivations(), labels);
        // System.out.print("loss: " + loss + " - ");
    }


    // public void validationLoss(SimpleMatrix val) {
    //     SimpleMatrix valData = val.extractMatrix(
    //             0, val.getNumRows(), 0, val.getNumCols() - (numClasses > 2 ? numClasses : 1));
    //     SimpleMatrix valLabels = val.extractMatrix(
    //             0, val.getNumRows(), val.getNumCols() - (numClasses > 2 ? numClasses : 1), val.getNumCols());

    //     forwardPass(valData, valLabels);
    //     Output outLayer = (Output) layers.get(layers.size() - 1);
    //     this.valLoss = outLayer.getLoss().execute(outLayer.getActivations(), valLabels);
    //     System.out.print("val loss: " + valLoss + "\n");
    // }


    public void backprop(SimpleMatrix data, SimpleMatrix labels) {
        Output outLayer = (Output) layers.get(layers.size() - 1);
        SimpleMatrix gradientWrtOutput = outLayer.getLoss().gradient(outLayer, outLayer.getLabels());
        getGradients(outLayer, gradientWrtOutput, data);

        for (Layer l : layers) {
            l.updateWeights(l.getGradientWeights(), optimizer);
            l.updateBiases(l.getGradientBias(), optimizer);
        }
    }


    public void getGradients(Layer currLayer, SimpleMatrix gradient, SimpleMatrix data) {
        Layer curr = currLayer;
        SimpleMatrix gradientWrtWeights;
        SimpleMatrix gradientWrtBias;

        if (currLayer instanceof Output) {
            Output out = (Output) curr;
            Layer prev = layers.get(layers.indexOf(curr) - 1);
            gradientWrtWeights = out.gradientWeights(prev, gradient);
            gradientWrtBias = out.gradientBias(curr, gradient);
            // curr.setGradientWeights(gradientWrtWeights);
            // curr.setGradientBiases(gradientWrtBias);
        } else {
            Layer prev;
            if (layers.indexOf(curr) > 0) {
                prev = layers.get(layers.indexOf(curr) - 1);
            } else {
                prev = new Layer();
                prev.setActivations(data);
            }

            gradientWrtWeights = currLayer.gradientWeights(prev, gradient);
            gradientWrtBias = currLayer.gradientBias(gradient);
            // curr.setGradientWeights(gradientWrtWeights);
            // curr.setGradientBiases(gradientWrtBias);
        }

        if (curr.getRegularizer() != null) {
            gradientWrtWeights = gradientWrtWeights.plus(curr.getRegularizer().regularize(currLayer.getWeights()));
        }

        curr.setGradientWeights(gradientWrtWeights);
        curr.setGradientBiases(gradientWrtBias);

        if (layers.indexOf(curr) > 0) {
            Layer prev = layers.get(layers.indexOf(curr) - 1);
            SimpleMatrix next = prev.getActFunc().gradient(prev, gradient.mult(currLayer.getWeights().transpose()));
            getGradients(prev, next, data);
        }
    }

    

}