package src.com.JakeKrayger.nn.initialize;

import java.util.Random;

import src.com.JakeKrayger.nn.components.Layer;
import src.com.JakeKrayger.nn.layers.*;

public class HeInit extends InitWeights {
    public double[] initWeight(Layer in) {
        int inL = in.getNodes().size();
        double std = Math.sqrt(2.0 / inL);
        double[] weights = new double[inL];

        for (int i = 0; i < in.getNodes().size(); i++) {
            Random rand = new Random();
            weights[i] = rand.nextGaussian() * std;
        }

        return weights;
    }
}
