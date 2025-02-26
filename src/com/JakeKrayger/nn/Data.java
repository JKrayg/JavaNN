package src.com.JakeKrayger.nn;

import org.ejml.simple.SimpleMatrix;

import src.com.JakeKrayger.nn.utils.MathUtils;

public class Data {
    private SimpleMatrix data;
    private String[] labels;

    public Data() {}

    public Data(double[][] data) {
        this.data = new SimpleMatrix(data);
    }

    public Data(double[][] data, String[] labels) {
        this.data = new SimpleMatrix(data);
        this.labels = labels;
    }

    public Data(String filename) {
        
    }

    public SimpleMatrix getData() {
        return data;
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
