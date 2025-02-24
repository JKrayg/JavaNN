package src.com.JakeKrayger.nn.initialize;

import java.util.Random;
import org.ejml.simple.SimpleMatrix;
import src.com.JakeKrayger.nn.components.Layer;

public class HeInit extends InitWeights {
    public SimpleMatrix initWeight(Layer prev, Layer curr) {
        int prevNeurons = prev.getNumNeurons();
        int currNeurons = curr.getNumNeurons();
        double std = Math.sqrt(2.0 / prevNeurons);
        double[][] weights = new double[prevNeurons][currNeurons];

        for (int i = 0; i < prevNeurons; i++) {
            for (int j = 0; j < currNeurons; j++) {
                Random rand = new Random();
                weights[i][j] = rand.nextGaussian() * std;
            }
        }

        return new SimpleMatrix(weights);
    }

    public SimpleMatrix initWeight(int inputSize, Layer curr) {
        int currNeurons = curr.getNumNeurons();
        double std = Math.sqrt(2.0 / inputSize);
        double[][] weights = new double[inputSize][currNeurons];

        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < currNeurons; j++) {
                Random rand = new Random();
                weights[i][j] = rand.nextGaussian() * std;
            }
        }

        return new SimpleMatrix(weights);
    }
}
