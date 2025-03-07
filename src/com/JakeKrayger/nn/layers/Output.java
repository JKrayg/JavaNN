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
}
