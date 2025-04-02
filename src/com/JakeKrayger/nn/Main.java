package src.com.JakeKrayger.nn;

// Jake Krayger
import java.util.ArrayList;
import java.util.Random;
import java.util.Scanner;

import org.ejml.simple.SimpleMatrix;
import java.io.File;
import java.io.FileNotFoundException;
import src.com.JakeKrayger.nn.activation.*;
import src.com.JakeKrayger.nn.components.*;
import src.com.JakeKrayger.nn.layers.*;
import src.com.JakeKrayger.nn.training.loss.*;
import src.com.JakeKrayger.nn.training.metrics.*;
import src.com.JakeKrayger.nn.training.normalization.BatchNormalization;
import src.com.JakeKrayger.nn.training.optimizers.*;
import src.com.JakeKrayger.nn.training.regularizers.*;

public class Main {
    public static void main(String[] args) {
        // String filePath = "src\\resources\\datasets\\wdbc.data";
        // String filePath = "src\\resources\\datasets\\iris.data";
        String filePath = "src\\resources\\datasets\\mnist.csv";
        ArrayList<double[]> dataArrayList = new ArrayList<>();
        // ArrayList<String> labelsArrayList = new ArrayList<>();
        ArrayList<Integer> labelsArrayList = new ArrayList<>();
        

        try {
            File f = new File(filePath);
            Scanner scan = new Scanner(f);
            while (scan.hasNextLine()) {
                // ** iris data **
                // String line = scan.nextLine();
                // String values = line.substring(0, line.lastIndexOf(","));
                // double[] toDub;
                // String[] splitValues = values.split(",");
                // toDub = new double[splitValues.length];

                // for (int i = 0; i < splitValues.length; i++) {
                //     toDub[i] = Double.parseDouble(splitValues[i]);
                // }

                // dataArrayList.add(toDub);
                // String label = line.substring(line.lastIndexOf(",") + 1);
                // labelsArrayList.add(label);


                // ** wdbc data **
                // String line = scan.nextLine();
                // String[] splitLine = line.split(",", 3);
                // String label = splitLine[1];
                // labelsArrayList.add(label);
                // double[] toDub;
                // String values = splitLine[2];
                // String[] splitValues = values.split(",");
                // toDub = new double[splitValues.length];

                // for (int i = 0; i < splitValues.length; i++) {
                //     toDub[i] = Double.parseDouble(splitValues[i]);
                // }

                // dataArrayList.add(toDub);

                // ** mnist **
                String line = scan.nextLine();
                String[] splitLine = line.split(",", 2);
                int label = Integer.parseInt(splitLine[0]);
                labelsArrayList.add(label);
                double[] toDub;
                String values = splitLine[1];
                String[] splitValues = values.split(",");
                toDub = new double[splitValues.length];

                for (int i = 0; i < splitValues.length; i++) {
                    toDub[i] = Double.parseDouble(splitValues[i]);
                }

                dataArrayList.add(toDub);

            }
            scan.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }


        double[][] data_ = dataArrayList.toArray(new double[0][]);
        // String[] labels = labelsArrayList.toArray(new String[0]);
        Integer[] labels = labelsArrayList.toArray(new Integer[0]);

        // System.out.println(new SimpleMatrix(data_).getRow(0));
        // System.out.println(labelsI[0]);

        // double[][] testerData = new double[75][];
        // String[] testerLabels = new String[75];
        // Random rand = new Random();

        // for (int i = 0; i < 75; i++) {
        //     int r = rand.nextInt(0, labels.length);
        //     testerData[i] = data_[r].clone();
        //     testerLabels[i] = labels[r];
        // }

        // Data data = new Data(testerData, testerLabels);

        Data data = new Data(data_, labels);
        data.minMaxNormalization();
        // data.zScoreNormalization();

        data.split(0.20, 0.20);

        NeuralNet nn = new NeuralNet();
        Dense d1 = new Dense(
            128,
            new ReLU(),
            784);
        d1.addRegularizer(new L2(0.01));
        d1.addNormalization(new BatchNormalization());

        Dense d2 = new Dense(
            64,
            new ReLU());
        d2.addRegularizer(new L2(0.01));
        d2.addNormalization(new BatchNormalization());

        Dense d3 = new Dense(
            64,
            new ReLU());
        d3.addRegularizer(new L2(0.01));
        d3.addNormalization(new BatchNormalization());

        Dense d4 = new Dense(
            32,
            new ReLU());
        d4.addRegularizer(new L2(0.01));
        d4.addNormalization(new BatchNormalization());

        Output d5 = new Output(
            data.getClasses().size(),
            new Softmax(),
            new CatCrossEntropy());
        d5.addRegularizer(new L2(0.01));

        nn.addLayer(d1);
        nn.addLayer(d2);
        nn.addLayer(d3);
        nn.addLayer(d4);
        nn.addLayer(d5);
        nn.compile(new Adam(0.001), new MultiClassMetrics());
        nn.miniBatchFit(data.getTrainData(), data.getTestData(), data.getValData(), 32, 20);

        // BatchNormalization b = new BatchNormalization();
        // b.setScale(new SimpleMatrix(new double[]{1, 1, 1}));
        // b.setShift(new SimpleMatrix(new double[]{12, 12, 12}));
        // b.setMeans(new SimpleMatrix(new double[]{2, 2, 2}));
        // b.setVariances(new SimpleMatrix(new double[]{3, 3, 3}));
        // b.setPreNormZ(new SimpleMatrix(new double[][]{{13, 14, 15}, {16, 17, 18}, {19, 20, 21}, {22, 23, 24}}));
        // SimpleMatrix dLdzHat = new SimpleMatrix(new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}});
        // // System.out.println(b.means(dLdzHat));
        // // b.setMeans(b.means(dLdzHat));
        // // System.out.println(b.variances(dLdzHat));
        // // System.out.println(b.getScale());
        // // System.out.println(b.getMeans());
        // // System.out.println(b.getVariances());
        // // System.out.println(b.getPreNormZ());
        // // System.out.println(dLdzHat);

        // System.out.println(b.gradientPreBN(dLdzHat));

    }
}
