package src.com.JakeKrayger.nn;

import java.util.ArrayList;
import java.util.Scanner;
import java.io.File;
import java.io.FileNotFoundException;

public class Data {
    private ArrayList<Double> data;

    public Data() {}

    public Data(ArrayList<Double> data) {
        this.data = data;
    }

    public Data(String filename) {
        
    }

    public ArrayList<Double> getData() {
        return data;
    }
}
