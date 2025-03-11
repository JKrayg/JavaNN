package src.com.JakeKrayger.nn.layers;

import org.ejml.simple.SimpleMatrix;
import src.com.JakeKrayger.nn.activation.ActivationFunction;
import src.com.JakeKrayger.nn.components.Layer;
import src.com.JakeKrayger.nn.training.loss.Loss;

public class Output extends Layer {
    private SimpleMatrix labels;
    public Output(int numNeurons, ActivationFunction actFunc) {
        super(numNeurons, actFunc);
    }

    public void setLabels(SimpleMatrix labels) {
        this.labels = labels;
    }

    public SimpleMatrix getLabels() {
        return labels;
    }

    public SimpleMatrix gradientWeights(Layer prevLayer, SimpleMatrix gradientWrtOutput) {
        return (prevLayer.getActivations().transpose()).mult(gradientWrtOutput).divide(labels.getNumElements());
    }

    public SimpleMatrix gradientBias(Layer currLayer) {
        SimpleMatrix gradientWrtOutput = currLayer.getLoss().gradient(currLayer, labels);
        double[] biasG = new double[currLayer.getNumNeurons()];
        for (int i = 0; i < gradientWrtOutput.getNumCols(); i++) {
            SimpleMatrix col = gradientWrtOutput.extractVector(false, i);
            biasG[i] = col.elementSum() / labels.getNumElements();
        }
        return new SimpleMatrix(biasG);
    }
}
