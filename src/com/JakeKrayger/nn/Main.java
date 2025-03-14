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
import src.com.JakeKrayger.nn.training.loss.BinCrossEntropy;
import src.com.JakeKrayger.nn.training.loss.CatCrossEntropy;
import src.com.JakeKrayger.nn.training.optimizers.Adam;
import src.com.JakeKrayger.nn.utils.MathUtils;

public class Main {
    public static void main(String[] args) {
        // String filePath = "src\\resources\\datasets\\wdbc.data";
        String filePath = "src\\resources\\datasets\\iris.data";
        ArrayList<double[]> dataArrayList = new ArrayList<>();
        ArrayList<String> labelsArrayList = new ArrayList<>();

        try {
            File f = new File(filePath);
            Scanner scan = new Scanner(f);
            while (scan.hasNextLine()) {
                // ** iris data **
                String line = scan.nextLine();
                String values = line.substring(0, line.lastIndexOf(","));
                double[] toDub;
                String[] splitValues = values.split(",");
                toDub = new double[splitValues.length];

                for (int i = 0; i < splitValues.length; i++) {
                    toDub[i] = Double.parseDouble(splitValues[i]);
                }

                dataArrayList.add(toDub);
                String label = line.substring(line.lastIndexOf(",") + 1);
                labelsArrayList.add(label);


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
            }

            scan.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }


        double[][] data_ = dataArrayList.toArray(new double[0][]);
        String[] labels = labelsArrayList.toArray(new String[0]);

        double[][] testerData = new double[30][];
        String[] testerLabels = new String[30];
        Random rand = new Random();

        for (int i = 0; i < 30; i++) {
            int r = rand.nextInt(0, labels.length);
            testerData[i] = data_[r].clone();
            testerLabels[i] = labels[r];
        }

        Data data = new Data(testerData, testerLabels);

        // Data data = new Data(data_, labels);
        data.zScoreNormalization();
        data.split(0.2);

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

        //  |    |   /""""\  \          /    ?
        //  |____|  |      |  \   /\   /     ?
        //  |    |  |      |   \ /  \ /      ?
        //  |    |   \____/     V    V       ?

        NeuralNet nn = new NeuralNet();
        Dense d1 = new Dense(16, new ReLU(), 4);
        Dense d2 = new Dense(8, new ReLU());
        Output d3 = new Output(data.getClasses().size(), new Sigmoid());
        nn.addLayer(d1);
        nn.addLayer(d2);
        nn.addLayer(d3);
        nn.compile(new Adam(), new CatCrossEntropy(), 0.03);
        nn.forwardPass(data.getData(), data.getLabels());

        for (Layer l: nn.getLayers()) {
            System.out.println(l.getClass().getSimpleName() + " - Activation Function: " + l.getActFunc().getClass().getSimpleName());
            System.out.println("activation matrix:");
            System.out.println(l.getActivations());
            System.out.println("weights:");
            System.out.println(l.getWeights());
            System.out.println("biases:");
            System.out.println(l.getBias());
        }

        // nn.miniBatchFit(data.getTrainData(), data.getTestData(), 32, 75);
        // nn.batchFit(data.getTrainData(), 75);

    }
}
