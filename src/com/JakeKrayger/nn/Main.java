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
import src.com.JakeKrayger.nn.training.optimizers.*;
import src.com.JakeKrayger.nn.training.regularizers.*;

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
        data.zScoreNormalization();

        // SimpleMatrix dater = data.getData().extractMatrix(0, 30, 0, 30);
        // SimpleMatrix labs = data.getLabels().extractMatrix(0, 30, 0, 1);
        // SimpleMatrix trainDater = dater.extractMatrix(0, 20, 0, 30)
        //                                .concatColumns(labs.extractMatrix(0, 20, 0, 1));
        // SimpleMatrix testDater = dater.extractMatrix(20, 25, 0, 30)
        //                                .concatColumns(labs.extractMatrix(20, 25, 0, 1));
        // SimpleMatrix valDater = dater.extractMatrix(25, 30, 0, 30)
        //                                .concatColumns(labs.extractMatrix(25, 30, 0, 1));

        data.split(0.20, 0.20);

        NeuralNet nn = new NeuralNet();
        Dense d1 = new Dense(8, new ReLU(), new L2(0.01), 30);
        Dense d2 = new Dense(4, new ReLU(), new L2(0.01));
        Output d3 = new Output(1, new Sigmoid(), new L2(0.01));
        nn.addLayer(d1);
        nn.addLayer(d2);
        nn.addLayer(d3);
        nn.compile(new Adam(0.001), new BinCrossEntropy(), new BinaryMetrics());
        // nn.compile(new SGD(0.01), new BinCrossEntropy(), new BinaryMetrics());
        nn.miniBatchFit(data.getTrainData(), data.getTestData(), data.getValData(), 16, 15);
        // nn.batchFit(data.getTrainData(), data.getTestData(), data.getValData(), 25);

        // MultiClassMetrics m = new MultiClassMetrics();

        // SimpleMatrix preds = new SimpleMatrix(new double[][]{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}});
        // SimpleMatrix truth = new SimpleMatrix(new double[][]{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}});
        // SimpleMatrix trainD = data.getTrainData();
        // SimpleMatrix testData = trainD.extractMatrix(
        //     0, trainD.getNumRows(), 0, trainD.getNumCols() - (data.getClasses().size() > 2 ? data.getClasses().size() : 1));
        // SimpleMatrix testLabels = trainD.extractMatrix(
        //     0, trainD.getNumRows(), trainD.getNumCols() - (data.getClasses().size() > 2 ? data.getClasses().size() : 1), trainD.getNumCols());

        // nn.forwardPass(testData, testLabels);
        // // System.out.println("de act: ");
        // System.out.println("predictions: ");
        // System.out.println(new SimpleMatrix(m.thresh(d3.getActivations())));
        // System.out.println("truths: ");
        // System.out.println(d3.getLabels());
        // System.out.println(m.confusion(d3.getActivations(), d3.getLabels()));
        // 
        // System.out.println(preds);
        // 
        // System.out.println(truth);
        // System.out.println(m.confusion(preds, truth));
        
        // System.out.println("d3 thresh:");
        // System.out.println(new SimpleMatrix(m.thresh(d3.getActivations())));
        








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
        
        // System.out.println(data.getTrainData());
        // System.out.println(data.getTestData());

    }
}
