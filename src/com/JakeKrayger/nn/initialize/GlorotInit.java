package src.com.JakeKrayger.nn.initialize;

import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import src.com.JakeKrayger.nn.activation.ActivationFunction;
import src.com.JakeKrayger.nn.components.*;

public class GlorotInit extends InitWeights {
    public double[] initWeight(Layer in, Layer out) {
        int inL = in.getNodes().size();
        int outL = out.getNodes().size();
        double varW = 1 / (inL + outL);
        double[] weights = new double[inL];
        for (int i = 0; i < in.getNodes().size(); i++) {
            Random rand = new Random();
            weights[i] = rand.nextGaussian() * Math.sqrt(varW);
        }

        return weights;

    }
}
