package src.com.JakeKrayger.nn;

import org.ejml.simple.SimpleMatrix;

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
}
