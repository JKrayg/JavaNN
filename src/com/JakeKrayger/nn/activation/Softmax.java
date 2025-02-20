package src.com.JakeKrayger.nn.activation;


import java.util.ArrayList;
import java.util.Arrays;
import org.ejml.simple.SimpleMatrix;
import src.com.JakeKrayger.nn.components.*;
import src.com.JakeKrayger.nn.utils.MathUtils;

public class Softmax extends ActivationFunction {
    private Layer previousLayer;
    private Layer outputLayer;

    // weighted sum for single node ∑(wi⋅xi)+b
    public double execute(double z, Layer prev, Node currNode) {
        MathUtils ws = new MathUtils();
        double zed = ws.weightedSum(prev, currNode);
        return zed;
    }
    
}
