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
import java.util.function.BinaryOperator;

import org.ejml.simple.SimpleMatrix;
import java.io.File;
import java.io.FileNotFoundException;
import src.com.JakeKrayger.nn.activation.*;
import src.com.JakeKrayger.nn.components.*;
import src.com.JakeKrayger.nn.layers.*;
import src.com.JakeKrayger.nn.training.loss.BinCrossEntropy;
import src.com.JakeKrayger.nn.training.loss.CatCrossEntropy;
import src.com.JakeKrayger.nn.training.optimizers.Adam;
import src.com.JakeKrayger.nn.utils.MathUtils;

public class Main {
    public static void main(String[] args) {
        // String filePath = "src\\resources\\datasets\\wdbc.data";
        // ArrayList<double[]> data = new ArrayList<>();
        // ArrayList<String> l = new ArrayList<>();
        // SimpleMatrix d;
        // int rows = 0;
        // int cols = 0;

        // try {
        //     File f = new File(filePath);
        //     Scanner scan = new Scanner(f);
        //     while (scan.hasNextLine()) {
        //         rows += 1;
        //         String line = scan.nextLine();
        //         String[] splitLine = line.split(",", 3);
        //         String label = splitLine[1];
        //         l.add(label);
        //         double[] toDub;
        //         String values = splitLine[2];
        //         String[] splitValues = values.split(",");
        //         ArrayList<String> strings = new ArrayList<>(Arrays.asList(splitValues));
        //         toDub = new double[splitValues.length];
        //         for (int i = 0; i < splitValues.length; i++) {
        //             toDub[i] = Double.parseDouble(splitValues[i]);
        //         }

        //         if (cols == 0) {
        //             cols = splitValues.length;
        //         }
        //         data.add(toDub);


        //     }

        //     d = new SimpleMatrix(rows, cols);

        //     for (int i = 0; i < data.size(); i++) {
        //         d.setRow(i, (new SimpleMatrix(data.get(i)).transpose()));
        //     }
        //     System.out.println(d.getNumCols());

        //     // for (int j = d.getNumRows(); j >= d.getNumRows() - 6; j--) {
        //     //     System.out.println(d.getRow(j));
        //     // }

        //     // System.out.println();
        //     // System.out.println(Arrays.asList(labels));
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
        int[] labels = {1, 0, 1, 1, 0};
        // Set<String> classes = new HashSet<>(List.of(labels));

        MathUtils maths = new MathUtils();
        Data data = new Data(sampleData, labels);
        data.zScoreNormalization();

        System.out.println("Scaled input data: \n" + data.getData());

        NeuralNet nn = new NeuralNet();
        Dense d1 = new Dense(3, new ReLU(), 3);
        Dense d2 = new Dense(4, new ReLU());
        Dense d3 = new Dense(1, new Sigmoid());
        nn.addLayer(d1);
        nn.addLayer(d2);
        nn.addLayer(d3);

        ReLU relu = new ReLU();
        Softmax softmax = new Softmax();
        Sigmoid sigmoid = new Sigmoid();

        SimpleMatrix d1Act = relu.execute(maths.weightedSum(data.getData(), d1));
        System.out.println("d1 activation matrix after ReLU function: \n" + d1Act);
        d1.setActivations(d1Act);

        SimpleMatrix d2Act = relu.execute(maths.weightedSum(d1, d2));
        System.out.println("d2 activation matrix after ReLU function: \n" + d2Act);
        d2.setActivations(d2Act);

        SimpleMatrix d3Act = sigmoid.execute(maths.weightedSum(d2, d3));
        System.out.println("d3 activation matrix after Sigmoid function: \n" + d3Act);
        d3.setActivations(d3Act);

        // nn.compile(new Adam(), new BinCrossEntropy());

        BinCrossEntropy bce = new BinCrossEntropy();
        System.out.println(bce.execute(d3.getActivations(), labels));




    }
}
