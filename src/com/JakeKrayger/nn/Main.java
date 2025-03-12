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
        String filePath = "src\\resources\\datasets\\wdbc.data";
        ArrayList<double[]> dta = new ArrayList<>();
        ArrayList<String> lbls = new ArrayList<>();
        int rows = 0;
        int cols = 0;

        try {
            File f = new File(filePath);
            Scanner scan = new Scanner(f);
            while (scan.hasNextLine()) {
                rows += 1;
                String line = scan.nextLine();
                String[] splitLine = line.split(",", 3);
                String label = splitLine[1];
                lbls.add(label);
                double[] toDub;
                String values = splitLine[2];
                String[] splitValues = values.split(",");
                toDub = new double[splitValues.length];

                for (int i = 0; i < splitValues.length; i++) {
                    toDub[i] = Double.parseDouble(splitValues[i]);
                }

                dta.add(toDub);
            }

            scan.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        cols = dta.get(0).length;

        String[] labs = new String[30];
        double[][] dats = new double[30][cols];

        for (int i = 0; i < 30; i++) {
            Random rand = new Random();
            int r = rand.nextInt(rows);
            dats[i] = dta.get(r);
            labs[i] = lbls.get(r);
        }

        Data data = new Data(dats, labs, 0.2);
        data.zScoreNormalization();

        // Random random = new Random();
        // double[] sample1 = new double[]{1, 6, 11};
        // double[] sample2 = new double[]{2, 7, 12};
        // double[] sample3 = new double[]{3, 8, 13};
        // double[] sample4 = new double[]{4, 9, 14};
        // double[] sample5 = new double[]{5, 10, 15};
        // // double[] sample1 = new double[]{random.nextDouble(), random.nextDouble(), random.nextDouble()};
        // // double[] sample2 = new double[]{random.nextDouble(), random.nextDouble(), random.nextDouble()};
        // // double[] sample3 = new double[]{random.nextDouble(), random.nextDouble(), random.nextDouble()};
        // // double[] sample4 = new double[]{random.nextDouble(), random.nextDouble(), random.nextDouble()};
        // // double[] sample5 = new double[]{random.nextDouble(), random.nextDouble(), random.nextDouble()};
        // double[][] sampleData = new double[][]{sample1, sample2, sample3, sample4, sample5};
        // String[] labels = {"T", "F", "T", "T", "F"};
        // Data data = new Data(sampleData, labels, 0.4);
        // data.zScoreNormalization();

        // System.out.println("train data:");
        // System.out.println(data.getTrainData());

        // System.out.println("train labels:");
        // System.out.println(data.getTrainLabels());

        System.out.println("test data:");
        System.out.println(data.getData());

        System.out.println("test labels:");
        System.out.println(data.getLabels());



        NeuralNet nn = new NeuralNet();
        Dense d1 = new Dense(2, new ReLU(), 30);
        Dense d2 = new Dense(3, new ReLU());
        Output d3 = new Output(1, new Sigmoid());
        nn.addLayer(d1);
        nn.addLayer(d2);
        nn.addLayer(d3);
        nn.compile(data, new Adam(), new BinCrossEntropy(), 0.03);
        // nn.singlePass();
        for (int i = 0; i < 50; i++) {
            nn.singlePass();
            // double loss = computeLoss(X, y);
            // System.out.println("Epoch " + i + ", Loss: " + loss);
        }

        // System.out.println("\nclasses:");
        // System.out.println(data.getClasses());

        // System.out.println("\nlabels:");
        // for (int i = 0; i < data.getLabels().getNumElements(); i++) {
        //     System.out.println(data.getLabels().get(i));
        // }

    }
}
