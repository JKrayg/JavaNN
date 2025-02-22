package src.com.JakeKrayger.nn;


// Jake Krayger
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Scanner;

import org.ejml.simple.SimpleMatrix;

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

        // nn.compile(new Adam(), new CatCrossEntropy());

        double[] sample1 = new double[]{1, 2, 3, 4};
        double[] sample2 = new double[]{5, 6, 7, 8};
        double[] sample3 = new double[]{9, 10, 11, 12};
        double[] sample4 = new double[]{13, 14, 15, 16};
        double[][] allData = new double[][]{sample1, sample2, sample3, sample4};
        double[][] batch1 = new double[][]{sample1, sample2};
        double[][] batch2 = new double[][]{sample3, sample4};

        Data data = new Data(allData);
        NeuralNet nn = new NeuralNet();

        InputLayer in = new InputLayer(data);
        Dense d1 = new Dense(5, new ReLU());

        nn.addLayer(in);
        nn.addLayer(d1);



        for (Layer l: nn.getLayers()) {
            System.out.println("\n" + l.getClass().getSimpleName() + " - Layer");
            System.out.println("oooooooooooooooooo");
            for (Node n: l.getNodes()) {
                if (!(l instanceof InputLayer)) {
                    System.out.println("\n" + n + " - Node");
                    System.out.println("values: " + n.getValues());
                    System.out.println("weights: " + n.getWeights());
                    System.out.println("bias: " + n.getBias());
                    System.out.println("forward conns: ");
                    if (n.getForwConnections() != null) {
                        for (Node fCon : n.getForwConnections()) {
                            System.out.println(fCon);
                        }
                    }
                    
                    System.out.println();
                    System.out.println("backward conns: ");
                    for (Node bCon : n.getBackConnections()) {
                        System.out.println(bCon);
                    }
                    System.out.println();
                } else {
                    System.out.println("\n" + n + " - Node");
                    System.out.println("values: " + n.getValues());
                    System.out.println("forward conns: ");
                    if (n.getForwConnections() != null) {
                        for (Node fCon : n.getForwConnections()) {
                            System.out.println(fCon);
                        }
                    }
                }
            }
            System.out.println();
        }

    }
}
