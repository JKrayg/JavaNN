package src.com.JakeKrayger.nn.layers.normalization;

import org.ejml.simple.SimpleMatrix;

public class BatchNormalization {
    private SimpleMatrix scale;
    private SimpleMatrix shift;
    private SimpleMatrix means;
    private SimpleMatrix variances;
    private double momentum;
    private double epsilon;
}
