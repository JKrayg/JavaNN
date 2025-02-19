package src.com.JakeKrayger.nn;

// Jake Krayger

import java.util.ArrayList;
import java.util.Arrays;

import src.com.JakeKrayger.nn.components.*;
import src.com.JakeKrayger.nn.layers.*;

public class Main {
    public static void main(String[] args) {
        Data data = new Data(new ArrayList<Double>(Arrays.asList(34.0, 180.0, 74.0)));

        InputLayer in = new InputLayer(data);
        Dense d1 = new Dense(4, "relu");
        // Dense d2 = new Dense(5, "relu");
        // Dense d3 = new Dense(3, "relu");
        OutputLayer out = new OutputLayer(1, "sigmoid");

        NeuralNet nn = new NeuralNet();
        nn.addLayer(in);
        nn.addLayer(d1);
        // nn.addLayer(d2);
        // nn.addLayer(d3);
        nn.addLayer(out);

        for (Layer l: nn.getLayers()) {
            System.out.println(l);
            System.out.println(l.getNodes());
            for (Node n: l.getNodes()) {
                System.out.println(n);
                System.out.println(n.getForwConnections());
                System.out.println(n.getWeights());
                System.out.println(n.getBackConnections());
            }
            System.out.println();
        }

    }
}
