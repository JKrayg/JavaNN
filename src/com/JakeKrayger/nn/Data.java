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
    private SimpleMatrix train;
    private SimpleMatrix test;
    private HashMap<String, Integer> classes;

    public Data() {}

    public Data(double[][] data) {
        this.data = new SimpleMatrix(data);
    }

    public Data(double[][] data, String[] labels) {
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

        if (classes.size() > 2) {
            this.labels = oneHot(new SimpleMatrix(ls));
        } else {
            this.labels = new SimpleMatrix(ls);
        }

        
    }

    public SimpleMatrix oneHot(SimpleMatrix labels) {
        SimpleMatrix encoded = new SimpleMatrix(labels.getNumRows(), classes.size());
        for (int i = 0; i < labels.getNumRows(); i++) {
            SimpleMatrix curr = new SimpleMatrix(1, classes.size());
            curr.set((int)labels.get(i), 1.0);
            encoded.setRow(i, curr);
        }

        return encoded;
    }

    public SimpleMatrix getData() {
        return data;
    }

    public SimpleMatrix getLabels() {
        return labels;
    }

    public SimpleMatrix getTestData() {
        return test;
    }

    public SimpleMatrix getTrainData() {
        return train;
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
        // gotta be a better way
        int rows = data.getNumRows();
        int cols = data.getNumCols();
        int numOfTest = (int) Math.floor(rows * testSize);
        double[][] testD = new double[numOfTest][cols];
        double[][] trainD = new double[rows - numOfTest][cols];
        double[][] testL = new double[numOfTest][classes.size() > 2 ? classes.size() : 0];
        double[][] trainL = new double[rows - numOfTest][classes.size() > 2 ? classes.size() : 0];
        Random rand = new Random();
        Set<Integer> used = new HashSet<>();

        for (int j = 0; j < numOfTest; j++) {
            int newRand = rand.nextInt(0, rows);
            while (used.contains(newRand)) {
                newRand = rand.nextInt(0, rows);
            }
            used.add(newRand);
            testD[j] = data.extractVector(true, newRand).toArray2()[0];
            testL[j] = labels.getRow(newRand).toArray2()[0];
        }

        ArrayList<double[]> trainDList = new ArrayList<>();
        ArrayList<double[]> trainLList = new ArrayList<>();

        for (int p = 0; p < rows; p++) {
            if (!used.contains(p)) {
                trainDList.add(data.getRow(p).toArray2()[0]);
                trainLList.add(labels.getRow(p).toArray2()[0]);
            }
        }

        for (int k = 0; k < trainDList.size(); k++) {
            trainD[k] = trainDList.get(k);
            trainL[k] = trainLList.get(k);
        }

        SimpleMatrix train = new SimpleMatrix(trainD);
        train = train.concatColumns(new SimpleMatrix(trainL));

        SimpleMatrix test = new SimpleMatrix(testD);
        test = test.concatColumns(new SimpleMatrix(testL));

        this.train = train;
        this.test = test;


    }
}
