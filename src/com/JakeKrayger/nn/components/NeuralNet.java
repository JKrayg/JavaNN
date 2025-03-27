package src.com.JakeKrayger.nn.components;

import java.text.Normalizer;
import java.util.ArrayList;
import java.util.Collections;

import org.ejml.simple.SimpleMatrix;
import src.com.JakeKrayger.nn.activation.*;
import src.com.JakeKrayger.nn.initialize.*;
import src.com.JakeKrayger.nn.layers.*;
import src.com.JakeKrayger.nn.training.loss.*;
import src.com.JakeKrayger.nn.training.metrics.Metrics;
import src.com.JakeKrayger.nn.training.normalization.BatchNormalization;
import src.com.JakeKrayger.nn.training.normalization.Normalization;
import src.com.JakeKrayger.nn.training.optimizers.*;
import src.com.JakeKrayger.nn.utils.MathUtils;

public class NeuralNet {
    private ArrayList<Layer> layers;
    private Optimizer optimizer;
    private Metrics metrics;
    private double loss;
    private double valLoss;
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

        Normalization norm = l.getNormalization();
        int numNeur = l.getNumNeurons();
        if (norm != null) {
            SimpleMatrix scVar = new SimpleMatrix(numNeur, 1);
            scVar.fill(1.0);
            SimpleMatrix shMeans = new SimpleMatrix(numNeur, 1);
            norm.setScale(scVar);
            norm.setShift(shMeans);
            norm.setMeans(shMeans);
            norm.setVariances(scVar);
        }

        this.layers.add(l);

    }

    public void compile(Optimizer o, Metrics m) {
        this.optimizer = o;
        this.metrics = m;

        for (Layer lyr : layers) {
            if (lyr instanceof Output) {
                this.numClasses = lyr.getNumNeurons();
            }

            if (optimizer instanceof Adam) {
                SimpleMatrix weightsO = new SimpleMatrix(lyr.getWeights().getNumRows(), lyr.getWeights().getNumCols());
                SimpleMatrix biasO = new SimpleMatrix(lyr.getBias().getNumRows(), lyr.getBias().getNumCols());
                lyr.setWeightsMomentum(weightsO);
                lyr.setWeightsVariance(weightsO);
                lyr.setBiasesMomentum(biasO);
                lyr.setBiasesVariance(biasO);
                Normalization norm = lyr.getNormalization();
                if (norm != null) {
                    SimpleMatrix shiftO = new SimpleMatrix(norm.getShift().getNumRows(), norm.getShift().getNumCols());
                    SimpleMatrix scaleO = new SimpleMatrix(norm.getScale().getNumRows(), norm.getScale().getNumCols());
                    norm.setShiftMomentum(shiftO);
                    norm.setShiftVariance(shiftO);
                    norm.setScaleMomentum(scaleO);
                    norm.setScaleVariance(scaleO);
                }
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
        System.out.println("val metrics: ");
        metrics(validation);
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
        ActivationFunction actF = L1.getActFunc();
        SimpleMatrix act = actF.execute(zL1);
        Normalization nL1 = L1.getNormalization();
        SimpleMatrix aL1;
        L1.setPreActivations(zL1);
        if (nL1 != null) {
            if (nL1.isBeforeActivation()) {
                zL1 = nL1.normalize(zL1);
                aL1 = actF.execute(zL1);
            } else {
                aL1 = actF.execute(zL1);
                aL1 = nL1.normalize(aL1);
            }
            
        } else {
            aL1 = actF.execute(zL1);
        }
        L1.setActivations(aL1);

        for (int q = 1; q < layers.size(); q++) {
            Layer curr = layers.get(q);
            Layer prev = layers.get(q - 1);
            SimpleMatrix z = maths.weightedSum(prev, curr);
            curr.setPreActivations(z);
            Normalization norm = curr.getNormalization();
            ActivationFunction actFunc = curr.getActFunc();
            SimpleMatrix activated;

            // normalization
            if (norm != null) {
                if (norm.isBeforeActivation()) {
                    z = norm.normalize(z);
                    activated = actFunc.execute(z);
                } else {
                    activated = actFunc.execute(z);
                    activated = norm.normalize(activated);
                }
                
            } else {
                activated = actFunc.execute(z);
            }
            curr.setActivations(activated);

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


    public void backprop(SimpleMatrix data, SimpleMatrix labels) {
        Output outLayer = (Output) layers.get(layers.size() - 1);
        SimpleMatrix gradientWrtOutput = outLayer.getLoss().gradient(outLayer, outLayer.getLabels());
        getGradients(outLayer, gradientWrtOutput, data);

        for (Layer l : layers) {
            l.updateWeights(optimizer);
            l.updateBiases(optimizer);
            Normalization norm = l.getNormalization();
            if (norm instanceof BatchNormalization) {
                ((BatchNormalization) norm).updateShift(optimizer);
                ((BatchNormalization) norm).updateScale(optimizer);
            }
        }
    }


    public double loss(SimpleMatrix d) {
        SimpleMatrix data = d.extractMatrix(
                0, d.getNumRows(), 0, d.getNumCols() - (numClasses > 2 ? numClasses : 1));
        SimpleMatrix labels = d.extractMatrix(
                0, d.getNumRows(), d.getNumCols() - (numClasses > 2 ? numClasses : 1), d.getNumCols());

        forwardPass(data, labels);
        Output outLayer = (Output) layers.get(layers.size() - 1);
        
        return outLayer.getLoss().execute(outLayer.getActivations(), labels);
    }


    public void getGradients(Layer currLayer, SimpleMatrix gradient, SimpleMatrix data) {
        Layer curr = currLayer;
        SimpleMatrix gradientWrtWeights;
        SimpleMatrix gradientWrtBias;
        Normalization norm = curr.getNormalization();

        if (currLayer instanceof Output) {
            Output out = (Output) curr;
            Layer prev = layers.get(layers.indexOf(curr) - 1);
            gradientWrtWeights = out.gradientWeights(prev, gradient);
            gradientWrtBias = out.gradientBias(curr, gradient);
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
            
        }

        if (curr.getRegularizer() != null) {
            gradientWrtWeights = gradientWrtWeights.plus(curr.getRegularizer().regularize(currLayer.getWeights()));
        }

        if (norm instanceof BatchNormalization) {
            BatchNormalization batchNorm = (BatchNormalization) norm;
            batchNorm.setGradientShift(batchNorm.gradientShift(gradient));
            batchNorm.setGradientScale(batchNorm.gradientScale(gradient));
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