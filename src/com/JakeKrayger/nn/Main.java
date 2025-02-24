package src.com.JakeKrayger.nn;


// Jake Krayger
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Scanner;
import java.util.Set;
import java.util.AbstractMap.SimpleEntry;

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
        double[] weights = new double[]{0.1, 0.2, 0.3, 0.4};
        double[][] sampleData = new double[][]{sample1, sample2, sample3, sample4};
        String[] labels = {"R", "B", "B", "G"};
        Set<String> classes = new HashSet<>(List.of(labels));


        Data data = new Data(sampleData);
        // System.out.println(data.getData());
        NeuralNet nn = new NeuralNet();

        // InputLayer in = new InputLayer(data);
        Dense d1 = new Dense(3, new Sigmoid(), 4);
        Dense d2 = new Dense(3, new ReLU());
        nn.addLayer(d1);
        nn.addLayer(d2);


        // for (Layer l: nn.getLayers()) {
        //     System.out.println(l);
        //     System.out.println(l.getWeights());
        //     System.out.println(l.getbias());
        //     System.out.println();
        // }

        // System.out.println(d1.getWeights());

        MathUtils maths = new MathUtils();
        System.out.println(maths.weightedSum(sampleData, d1));


        // Dense d3 = new Dense(classes.size(), new Softmax());

        
        // nn.addLayer(d2);
        // nn.addLayer(d3);

        // nn.singleForwardPass(data, 2);

        // SimpleMatrix test = new SimpleMatrix(new double[][]{{1}, {2}, {3}});
        // Tanh tanH = new Tanh();

        // System.out.println(tanH.execute(test));

        // nn.compile(new Adam(), new BinCrossEntropy());







        // Node n1 = new Node(0.0);
        // Node n2 = new Node(0.0);
        // Node n3 = new Node(0.0);
        // n1.setValues(new SimpleMatrix(sample1));
        // n2.setValues(new SimpleMatrix(sample2));
        // n3.setValues(new SimpleMatrix(sample3));
        // ArrayList<Node> nodes = new ArrayList<>();
        // nodes.add(n1);
        // nodes.add(n2);
        // nodes.add(n3);
        // Layer l = new Layer(nodes);

        // Node nW = new Node(0.0);
        // nW.setWeights(weights);

        // MathUtils maths = new MathUtils();

        // System.out.println(maths.weightedSum(l, nW));




        // for (Layer l: nn.getLayers()) {
        //     System.out.println("\noooooooooooooooooo " + l.getClass().getSimpleName() + " - Layer");
        //     System.out.println("activation function: " + l.getActFunc().getClass().getSimpleName());
        //     for (Node n: l.getNodes()) {
        //         if (l.getInputSize() == 0) {
        //             System.out.println("\n" + n + " - Node");
        //             System.out.println("values: " + n.getValues());
        //             System.out.println("weights: " + n.getWeights());
        //             System.out.println("bias: " + n.getBias());
        //             System.out.println("forward conns: ");
        //             if (n.getForwConnections() != null) {
        //                 for (Node fCon : n.getForwConnections()) {
        //                     System.out.println(fCon);
        //                 }
        //             }
                    
        //             System.out.println();
        //             System.out.println("backward conns: ");
        //             for (Node bCon : n.getBackConnections()) {
        //                 System.out.println(bCon);
        //             }
        //             System.out.println();
        //         } else {
        //             System.out.println("\n" + n + " - Node");
        //             System.out.println("values: " + n.getValues());
        //             System.out.println("weights: " + n.getWeights());
        //             System.out.println("bias: " + n.getBias());
        //             System.out.println("forward conns: ");
        //             if (n.getForwConnections() != null) {
        //                 for (Node fCon : n.getForwConnections()) {
        //                     System.out.println(fCon);
        //                 }
        //             }
        //         }
        //     }
        //     System.out.println();
        // }

    }
}
