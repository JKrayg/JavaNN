package src.com.JakeKrayger.nn.activation;


import java.util.ArrayList;
import src.com.JakeKrayger.nn.components.*;
import src.com.JakeKrayger.nn.utils.MathUtils;

public class Softmax extends ActivationFunction {
    // weighted sum for single node ∑(wi⋅xi)+b
    public ArrayList<Double> execute(Layer prev, Layer output) {
        MathUtils ws = new MathUtils();
        ArrayList<Double> outWeightedSums = new ArrayList<>();
        double expWeightedSum = 0;
        ArrayList<Double> probabilities = new ArrayList<>();

        // get weighted sums for each output node and exp weighted sum 
        // for (Node n: output.getNodes()) {
        //     outWeightedSums.add(ws.weightedSum(prev, n));
        //     expWeightedSum = expWeightedSum + Math.exp(ws.weightedSum(prev, n));
        // }
        
        // get probabilities
        for (double d: outWeightedSums) {
            probabilities.add(Math.exp(d) / expWeightedSum);
        }

        return probabilities;
    }
    
}
