package src.com.JakeKrayger.nn;

// Jake Krayger
import java.util.ArrayList;
import java.util.Scanner;

import org.ejml.simple.SimpleMatrix;
import java.io.File;
import java.io.FileNotFoundException;
import src.com.JakeKrayger.nn.activation.*;
import src.com.JakeKrayger.nn.components.*;
import src.com.JakeKrayger.nn.layers.*;
import src.com.JakeKrayger.nn.training.loss.*;
import src.com.JakeKrayger.nn.training.metrics.BinaryMetrics;
import src.com.JakeKrayger.nn.training.optimizers.*;

public class Main {
    public static void main(String[] args) {
        String filePath = "src\\resources\\datasets\\wdbc.data";
        // String filePath = "src\\resources\\datasets\\iris.data";
        ArrayList<double[]> dataArrayList = new ArrayList<>();
        ArrayList<String> labelsArrayList = new ArrayList<>();

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
                String line = scan.nextLine();
                String[] splitLine = line.split(",", 3);
                String label = splitLine[1];
                labelsArrayList.add(label);
                double[] toDub;
                String values = splitLine[2];
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
        String[] labels = labelsArrayList.toArray(new String[0]);

        // double[][] testerData = new double[150][];
        // String[] testerLabels = new String[150];
        // Random rand = new Random();

        // for (int i = 0; i < 150; i++) {
        //     int r = rand.nextInt(0, labels.length);
        //     testerData[i] = data_[r].clone();
        //     testerLabels[i] = labels[r];
        // }

        // Data data = new Data(testerData, testerLabels);

        Data data = new Data(data_, labels);
        data.zScoreNormalization();
        data.split(0.2);

        NeuralNet nn = new NeuralNet();
        Dense d1 = new Dense(4, new ReLU(), 30);
        Dense d2 = new Dense(8, new ReLU());
        Output d3 = new Output(1, new Sigmoid());
        nn.addLayer(d1);
        nn.addLayer(d2);
        nn.addLayer(d3);
        nn.compile(new Adam(0.001), new BinCrossEntropy(), new BinaryMetrics(0.7));
        // nn.compile(new SGD(0.001), new BinCrossEntropy());
        nn.miniBatchFit(data.getTrainData(), data.getTestData(), 32, 50);
        // nn.batchFit(data.getTrainData(),data.getTestData(), 30);
        // nn.forwardPass(data.getData(), data.getLabels());

        // for (Layer l: nn.getLayers()) {
        //     System.out.println(l.getClass().getSimpleName() + " - Activation Function: " + l.getActFunc().getClass().getSimpleName());
        //     System.out.println("activation matrix:");
        //     System.out.println(l.getActivations());
        //     System.out.println("weights:");
        //     System.out.println(l.getWeights());
        //     System.out.println("weights momentum:");
        //     System.out.println(l.getWeightsMomentum());
        //     System.out.println("weights variance:");
        //     System.out.println(l.getWeightsVariance());
        //     System.out.println("biases:");
        //     System.out.println(l.getBias());
        //     System.out.println("biases momentum:");
        //     System.out.println(l.getBiasMomentum());
        //     System.out.println("biases variance:");
        //     System.out.println(l.getBiasVariance());
        // }

        // CatCrossEntropy c = new CatCrossEntropy();
        // System.out.println(c.execute(d3.getActivations(), data.getLabels()));



        // nn.miniBatchFit(data.getTrainData(), data.getTestData(), 32, 75);
        // System.out.println(d3.getActivations());
        

        // System.out.println(data.getTrainData());
        // System.out.println(data.getTestData());

    }
}
