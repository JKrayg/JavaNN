package src.com.JakeKrayger.nn.initialize;

import java.util.Random;
import src.com.JakeKrayger.nn.layers.*;

public class HeInit extends InitWeights {
    public double initWeight(InputLayer in, OutputLayer out) {
        double inL = in.getNodes().size();
        Random rand = new Random();
        double std = Math.sqrt(2 / inL);

        return rand.nextGaussian() * std;
    }
}
