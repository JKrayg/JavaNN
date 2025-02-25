package src.com.JakeKrayger.nn;

import org.ejml.simple.SimpleMatrix;

import src.com.JakeKrayger.nn.utils.MathUtils;

public class Data {
    private SimpleMatrix data;

    public Data() {}

    public Data(double[][] data) {
        this.data = new SimpleMatrix(data);
    }

    public Data(String filename) {
        
    }

    public SimpleMatrix getData() {
        return data;
    }

    public void scale() {
        MathUtils maths = new MathUtils();
        if (data != null) {
            for (int i = 0; i < data.getNumCols(); i++) {
                SimpleMatrix col = new SimpleMatrix(data.getColumn(i));
                double mean = (col.elementSum() / data.getNumRows());
                double std = maths.std(col);
                for (int j = 0; j < data.getNumRows(); j++) {
                    data.set(j, i, (data.get(j, i) - mean) / std);
                }
            }
            
        }
    }
}
