package src.com.JakeKrayger.nn;


// Jake Krayger
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Scanner;
import java.io.File;
import java.io.FileNotFoundException;
import src.com.JakeKrayger.nn.activation.*;
import src.com.JakeKrayger.nn.components.*;
import src.com.JakeKrayger.nn.layers.*;
import src.com.JakeKrayger.nn.nodes.InputNode;
import src.com.JakeKrayger.nn.training.loss.CatCrossEntropy;
import src.com.JakeKrayger.nn.training.optimizers.Adam;
import src.com.JakeKrayger.nn.utils.MathUtils;

public class Main {
    public static void main(String[] args) {
        // String filePath = "src\\resources\\datasets\\wdbc.data";
        // ArrayList<double[]> data = new ArrayList<>();
        // ArrayList<String> labels = new ArrayList<>();

        // try {
        //     File f = new File(filePath);
        //     Scanner scan = new Scanner(f);
        //     while (scan.hasNextLine()) {
        //         String line = scan.nextLine();
        //         labels.add(line.split(",", 3)[1]);
        //         double[] toDub;
        //         String d = line.split(",", 3)[2];
        //         String[] splt = d.split(",");
        //         toDub = new double[splt.length];
        //         for (int i = 0; i < splt.length; i++) {
        //             toDub[i] = Double.parseDouble(splt[i]);
        //         }
        //         data.add(toDub);


        //     }
        //     for (double d: data.get(0)) {
        //         System.out.println(d);
        //     }
        //     System.out.println(Arrays.asList(labels));
        //     scan.close();
        // } catch (FileNotFoundException e) {
        //     e.printStackTrace();
        // }

        Data data = new Data(new ArrayList<Double>(Arrays.asList(34.0, 180.0, 74.0)));

        InputLayer in = new InputLayer(data);
        Dense d1 = new Dense(3, new ReLU());
        // Dense d2 = new Dense(5, "relu");
        // Dense d3 = new Dense(3, "relu");
        OutputLayer out = new OutputLayer(data.getData().size(), new Sigmoid());

        NeuralNet nn = new NeuralNet();
        nn.addLayer(in);
        nn.addLayer(d1);
        // nn.addLayer(d2);
        // nn.addLayer(d3);
        nn.addLayer(out);

        // nn.compile(new Adam(), new CatCrossEntropy());










        for (Layer l: nn.getLayers()) {
            System.out.println(l.getClass().getSimpleName() + " - Layer");
            // if (!(l instanceof InputLayer)) {
            //     System.out.println(l.getActFunc().getClass().getSimpleName() + " - Activation function");
            // }
            System.out.println("oooooooooooooooooo");
            for (Node n: l.getNodes()) {
                if (!(n instanceof InputNode)) {
                    System.out.println(n.getClass().getSimpleName() + " - Node");
                    System.out.println("value: " + n.getValue());
                    System.out.println("weights: " + n.getWeights());
                    System.out.println("bias: " + n.getBias());
                    // System.out.println("forward conns: " + n.getForwConnections());
                    // System.out.println("backward conns: " + n.getBackConnections());
                } else {
                    System.out.println(n.getClass().getSimpleName() + " - Node");
                    System.out.println("value: " + n.getValue());
                    // System.out.println("forward conns: " + n.getForwConnections());
                }
            }
            System.out.println();
        }

        

        Softmax sm = new Softmax();

        System.out.println("Probabilties");
        for(double d: sm.execute(d1, out)) {
            System.out.println(d);
        }

        // MathUtils ws = new MathUtils();
        // for (Node n: d1.getNodes()) {
        //     System.out.println(ws.weightedSum(in, n));
        // }

    }
}
