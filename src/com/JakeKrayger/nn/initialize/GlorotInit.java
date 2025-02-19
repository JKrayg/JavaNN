package src.com.JakeKrayger.nn.initialize;

import java.util.Random;
import src.com.JakeKrayger.nn.activation.ReLU;
import src.com.JakeKrayger.nn.components.*;

public class GlorotInit extends InitWeights {
    public double initWeight(Layer in, Layer out, ActivationFunction actFunc) {
        double inL = in.getNodes().size();
        double outL = out.getNodes().size();
        double std;

        if (actFunc instanceof ReLU) {
            double varW = 2 / (inL + outL);
            std = Math.sqrt(varW);
        } else {
            double varW = 1 / (inL + outL);
            std = Math.sqrt(varW);
        }

        Random rand = new Random();
        return rand.nextGaussian() * std;

    }
}
