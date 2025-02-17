import java.util.Random;

public class HeInit {
    public double initWeight(InputLayer in, OutputLayer out) {
        double inL = in.getNodes().size();
        Random rand = new Random();
        double std = Math.sqrt(2 / inL);

        return rand.nextGaussian() * std;
    }
}
