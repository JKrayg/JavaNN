package src.com.JakeKrayger.nn.training.metrics;

import org.ejml.simple.SimpleMatrix;

public class MultiClassMetrics extends Metrics{
    private double threshold;

    public MultiClassMetrics(double threshold) {
        this.threshold = threshold;
    }

    public void getMetrics(SimpleMatrix pred, SimpleMatrix trueVals) {
        String dis = "Accuracy: " + accuracy(pred, trueVals) + "\n";
        dis += "Precision: " + precision(pred, trueVals) + "\n";
        dis += "Recall: " + recall(pred, trueVals) + "\n";
        System.out.println(dis);
    }

    public double accuracy(SimpleMatrix pred, SimpleMatrix trueVals) {
        return 0.0;
    }

    public double precision(SimpleMatrix pred, SimpleMatrix trueVals) {
        return 0.0;
    }

    public double recall(SimpleMatrix pred, SimpleMatrix trueVals) {
        return 0.0;
    }

    public double f1(SimpleMatrix pred, SimpleMatrix trueVals) {
        return 0.0;
    }
    
}
