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

        double[] sample1 = new double[]{0.1, 0.2, 0.3, 0.4};
        double[] sample2 = new double[]{0.5, 0.6, 0.7, 0.8};
        double[] sample3 = new double[]{0.9, 0.10, 0.11, 0.12};
        double[] sample4 = new double[]{0.13, 0.14, 0.15, 0.16};
        double[][] sampleData = new double[][]{sample1, sample2, sample3, sample4};
        String[] labels = {"R", "B", "B", "G"};
        Set<String> classes = new HashSet<>(List.of(labels));


        Data data = new Data(sampleData);
        // System.out.println(data.getData());
        NeuralNet nn = new NeuralNet();

        // InputLayer in = new InputLayer(data);
        Dense d1 = new Dense(3, new ReLU(), 4);
        Dense d2 = new Dense(3, new ReLU());
        nn.addLayer(d1);
        nn.addLayer(d2);


        // for (Layer l: nn.getLayers()) {
        //     System.out.println(l);
        //     System.out.println(l.getWeights());
        //     System.out.println(l.getbias());
        //     System.out.println();
        // }

        // System.out.println(d1.getBias());

        MathUtils maths = new MathUtils();
        ReLU relu = new ReLU();
        System.out.println(relu.execute(maths.weightedSum(sampleData, d1)));


    }
}
