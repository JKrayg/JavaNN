import java.util.Random;

public class GlorotInit {
    public double initWeight(InputLayer in, OutputLayer out, String actFunc) {
        double inL = in.getNodes().size();
        double outL = out.getNodes().size();
        double std;

        if (actFunc.equals("relu")) {
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
