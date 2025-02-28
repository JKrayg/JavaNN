package src.com.JakeKrayger.nn;

import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import org.ejml.simple.SimpleMatrix;
import src.com.JakeKrayger.nn.utils.MathUtils;

public class Data {
    private SimpleMatrix data;
    private double[] labels;
    private HashMap<String, Integer> classes;

    public Data() {
    }

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
        this.labels = ls;
    }

    public SimpleMatrix getData() {
        return data;
    }

    public double[] getLabels() {
        return labels;
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
}
