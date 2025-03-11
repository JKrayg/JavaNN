package src.com.JakeKrayger.nn;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import org.ejml.simple.SimpleMatrix;
import src.com.JakeKrayger.nn.utils.MathUtils;

public class Data {
    private SimpleMatrix data;
    private SimpleMatrix labels;
    private SimpleMatrix trainLabels;
    private SimpleMatrix trainData;
    private SimpleMatrix testLabels;
    private SimpleMatrix testData;
    private HashMap<String, Integer> classes;

    public Data() {}

    public Data(double[][] data) {
        this.data = new SimpleMatrix(data);
    }

    public Data(SimpleMatrix data, SimpleMatrix labels) {
        this.testData = data;
        this.testLabels = labels;
    }

    public Data(double[][] data, String[] labels, double testSize) {
        this.data = new SimpleMatrix(data);

        // create a hashtable of distinct labels mapped to an integer
        HashMap<String, Integer> h = new HashMap<>();
        Set<String> c = new HashSet<>(List.of(labels));
        int count = 0;
        for (String s: c) {
            h.put(s, count);
            count++;
        }
        this.classes = h;

        // create list of a label values
        double[] ls = new double[labels.length];
        for (int i = 0; i < labels.length; i++) {
            ls[i] = classes.get(labels[i]);
        }

        this.labels = new SimpleMatrix(ls);

        // gotta be a better way
        ArrayList<double[]> allData = new ArrayList<>();
        ArrayList<Double> allLabels = new ArrayList<>();
        int numOfTest = (int) Math.floor(labels.length * testSize);
        double[][] testD = new double[numOfTest][data[0].length];
        double[] testL = new double[numOfTest];
        double[][] trainD = new double[labels.length - numOfTest][data[0].length];
        double[] trainL = new double[labels.length - numOfTest];
        Random rand = new Random();
        Set<Integer> used = new HashSet<>();

        for (int i = 0; i < data.length; i++) {
            allData.add(data[i]);
            allLabels.add(ls[i]);
        }

        for (int j = 0; j < numOfTest; j++) {
            int newRand = rand.nextInt(0, allData.size());
            while (used.contains(newRand)) {
                newRand = rand.nextInt(0, allData.size());
            }
            used.add(newRand);
            testD[j] = allData.get(newRand);
            testL[j] = allLabels.get(newRand);
        }

        ArrayList<double[]> trainDList = new ArrayList<>();
        ArrayList<Double> trainLList = new ArrayList<>();

        for (int p = 0; p < allData.size(); p++) {
            if (!used.contains(p)) {
                trainDList.add(allData.get(p));
                trainLList.add(allLabels.get(p));
            }
        }

        for (int k = 0; k < trainDList.size(); k++) {
            trainD[k] = trainDList.get(k);
            trainL[k] = trainLList.get(k);
        }

        this.trainLabels = new SimpleMatrix(trainL);
        this.trainData = new SimpleMatrix(trainD);
        this.testLabels = new SimpleMatrix(testL);
        this.testData = new SimpleMatrix(testD);
    }

    public SimpleMatrix getData() {
        return data;
    }

    public SimpleMatrix getLabels() {
        return labels;
    }

    public SimpleMatrix getTestData() {
        return testData;
    }

    public SimpleMatrix getTestLabels() {
        return testLabels;
    }

    public SimpleMatrix getTrainData() {
        return trainData;
    }

    public SimpleMatrix getTrainLabels() {
        return trainLabels;
    }

    public HashMap<String, Integer> getClasses() {
        return classes;
    }

    public void zScoreNormalization() {
        MathUtils maths = new MathUtils();
        int cols = data.getNumCols();
        int rows = data.getNumRows();
        if (data != null) {
            for (int i = 0; i < cols; i++) {
                SimpleMatrix col = new SimpleMatrix(data.getColumn(i));
                double mean = (col.elementSum() / rows);
                double std = maths.std(col);
                for (int j = 0; j < rows; j++) {
                    data.set(j, i, (data.get(j, i) - mean) / std);
                }
            }

        }
    }

    public void split(double testSize) {
        Random rand = new Random();
        Set<Integer> used = new HashSet<>();
        int numOfTest = (int) Math.round(labels.getNumElements() * testSize);
        double[][] testData = new double[numOfTest][data.getNumCols()];
        double[] testLabels = new double[numOfTest];

        for (int i = 0; i < numOfTest; i++) {
            int newRand = rand.nextInt(0, data.getNumRows());
            while (used.contains(newRand)) {
                newRand = rand.nextInt(0, data.getNumRows());
            }
            used.add(newRand);

            for (int j = 0; j < data.getNumCols(); j++) {
                testData[i][j] = data.get(newRand, j);
            }
            testLabels[i] = labels.get(newRand);
        }

        double[][] trainD = new double[data.getNumRows() - numOfTest][data.getNumCols()];
        double[] trainL = new double[data.getNumRows() - numOfTest];
        for (int k = 0; k < data.getNumRows(); k++) {
            if (!used.contains(k)) {
                for (int p = 0; p < data.getNumCols(); p++) {
                    trainD[k][p] = data.get(k, p);
                }
                trainL[k] = labels.get(k);
            }
        }

        // this.testData = new SimpleMatrix(testData);
        // this.testLabels = new SimpleMatrix(testLabels);

        // System.out.println("test data:");
        // System.out.println(new SimpleMatrix(testData));

        // System.out.println("test labels:");
        // System.out.println(new SimpleMatrix(testLabels));

        // ArrayList<SimpleMatrix> test= new ArrayList<>();
        // test.add(new SimpleMatrix(testData));
        // test.add(new SimpleMatrix(trainD));
        // test.add(new SimpleMatrix(testLabels));
        // test.add(new SimpleMatrix(trainL));


    }
}
