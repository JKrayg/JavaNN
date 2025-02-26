package src.com.JakeKrayger.nn;


// Jake Krayger
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
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
        Random random = new Random();
        double[] sample1 = new double[]{random.nextDouble() * 50, random.nextDouble() * 50, random.nextDouble() * 50};
        double[] sample2 = new double[]{random.nextDouble() * 50, random.nextDouble() * 50, random.nextDouble() * 50};
        double[] sample3 = new double[]{random.nextDouble() * 50, random.nextDouble() * 50, random.nextDouble() * 50};
        double[] sample4 = new double[]{random.nextDouble() * 50, random.nextDouble() * 50, random.nextDouble() * 50};
        double[] sample5 = new double[]{random.nextDouble() * 50, random.nextDouble() * 50, random.nextDouble() * 50};
        double[][] sampleData = new double[][]{sample1, sample2, sample3,sample4, sample5};
        String[] labels = {"T", "F", "T", "T", "F"};
        Set<String> classes = new HashSet<>(List.of(labels));

        MathUtils maths = new MathUtils();
        // for (int i = 0; i < )
        Data data = new Data(sampleData, labels);
        // System.out.println(data.getData());
        data.zScoreNormalization();
        System.out.println("Scaled input data: \n" + data.getData());



        NeuralNet nn = new NeuralNet();

        // InputLayer in = new InputLayer(data);
        Dense d1 = new Dense(3, new ReLU(), 3);
        Dense d2 = new Dense(3, new ReLU());
        Dense d3 = new Dense(3, new Softmax());
        nn.addLayer(d1);
        nn.addLayer(d2);
        nn.addLayer(d3);

        ReLU relu = new ReLU();
        Softmax softmax = new Softmax();
        SimpleMatrix d1Act = relu.execute(maths.weightedSum(data.getData(), d1));
        System.out.println("d1 activation matrix after ReLU function: \n" + d1Act);
        d1.setActivations(d1Act);
        SimpleMatrix d2Act = relu.execute(maths.weightedSum(d1, d2));
        System.out.println("d2 activation matrix after ReLU function: \n" + d2Act);
        d2.setActivations(d2Act);
        // System.out.println(maths.weightedSum(d2, d3));
        SimpleMatrix d3Act = softmax.execute(maths.weightedSum(d2, d3));
        System.out.println("d3 activation matrix after Softmax function: \n" + d3Act);
        d2.setActivations(d3Act);


    }
}
