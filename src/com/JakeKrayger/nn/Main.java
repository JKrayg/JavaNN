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

        String[] labs = new String[3];
        double[][] dats = new double[3][cols];

        for (int i = 0; i < 3; i++) {
            Random rand = new Random();
            int r = rand.nextInt(rows);
            dats[i] = dta.get(r);
            labs[i] = lbls.get(r);
        }

        // Random random = new Random();
        // double[] sample1 = new double[]{random.nextDouble() * 50, random.nextDouble()
        // * 50, random.nextDouble() * 50};
        // double[] sample2 = new double[]{random.nextDouble() * 50, random.nextDouble()
        // * 50, random.nextDouble() * 50};
        // double[] sample3 = new double[]{random.nextDouble() * 50, random.nextDouble()
        // * 50, random.nextDouble() * 50};
        // double[] sample4 = new double[]{random.nextDouble() * 50, random.nextDouble()
        // * 50, random.nextDouble() * 50};
        // double[] sample5 = new double[]{random.nextDouble() * 50, random.nextDouble()
        // * 50, random.nextDouble() * 50};
        // double[][] sampleData = new double[][]{sample1, sample2, sample3,sample4,
        // sample5};
        // String[] labels = {"T", "F", "T", "T", "F"};
        // Set<String> classes = new HashSet<>(List.of(labels));
        // Data data = new Data(sampleData, labels);
        // data.zScoreNormalization();

        // System.out.println("Scaled input data: \n" + data.getData());
        Data data = new Data(dats, labs);
        data.zScoreNormalization();

        NeuralNet nn = new NeuralNet();
        Dense d1 = new Dense(6, new ReLU(), 30);
        Dense d2 = new Dense(7, new ReLU());
        Dense d3 = new Dense(1, new Sigmoid());
        nn.addLayer(d1);
        nn.addLayer(d2);
        nn.addLayer(d3);

        MathUtils maths = new MathUtils();
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
        System.out.println("d3 activation matrix after Sigmoid function: \n" +
        d3Act);
        d3.setActivations(d3Act);

        // nn.compile(new Adam(), new BinCrossEntropy());

        BinCrossEntropy bce = new BinCrossEntropy();
        System.out.println("value after binary cross entropy:");
        System.out.println(bce.execute(d3.getActivations(), data.getLabels()));

        System.out.println("\ngradient of output:");
        System.out.println(bce.outputGradient(d3.getActivations(), data.getLabels()));

        System.out.println("\nclasses:");
        System.out.println(data.getClasses());
        System.out.println("\nlabels:");
        for (int i = 0; i < data.getLabels().length; i++) {
        System.out.println(data.getLabels()[i]);
        }

    }
}
