package src.com.JakeKrayger.nn.training.metrics;

import org.ejml.simple.SimpleMatrix;

public class BinaryMetrics extends Metrics {
    private double threshold;

    public BinaryMetrics(double threshold) {
        this.threshold = threshold;
    }

    public void getMetrics(SimpleMatrix pred, SimpleMatrix trueVals) {
        String dis = "Accuracy: " + accuracy(pred, trueVals) + "\n";
        dis += "Precision: " + precision(pred, trueVals) + "\n";
        dis += "Recall: " + recall(pred, trueVals) + "\n";
        System.out.println(dis);
    }

    public double accuracy(SimpleMatrix pred, SimpleMatrix trueVals) {
        int correct = 0;
        double[] preds = thresh(pred);

        for (int i = 0; i < preds.length; i++) {
            if (preds[i] == trueVals.get(i)) {
                correct += 1;
            }
        }
        return ((double) correct) / ((double) preds.length);
    }

    public double precision(SimpleMatrix pred, SimpleMatrix trueVals) {
        int correct = 0;
        int wrong = 0;
        double[] preds = thresh(pred);

        for (int i = 0; i < preds.length; i++) {
            if (trueVals.get(i) == 1.0 && preds[i] == 1.0) {
                correct += 1;
            } else if (preds[i] == 1.0) {
                wrong += 1;
            }
        }

        double prec = ((double) correct) / (((double) correct) + ((double) wrong));
        if (Double.isNaN(prec)) {
            return 0.0;
        } else {
            return prec;
        }
    }

    public double recall(SimpleMatrix pred, SimpleMatrix trueVals) {
        int correct = 0;
        int ones = 0;
        double[] preds = thresh(pred);

        for (int i = 0; i < preds.length; i++) {
            if (trueVals.get(i) == 1.0) {
                ones += 1;
                if (preds[i] == 1.0) {
                    correct += 1;
                }
            } 
        }
        
        double prec = ((double) correct) / ((double) ones);
        if (Double.isNaN(prec)) {
            return 0.0;
        } else {
            return prec;
        }
    }

    public double f1(SimpleMatrix pred, SimpleMatrix trueVals) {
        int correct = 0;
        int ones = 0;
        double[] preds = thresh(pred);

        for (int i = 0; i < preds.length; i++) {
            if (trueVals.get(i) == 1.0) {
                ones += 1;
                if (preds[i] == 1.0) {
                    correct += 1;
                }
            } 
        }
        
        double prec = ((double) correct) / ((double) ones);
        if (Double.isNaN(prec)) {
            return 0.0;
        } else {
            return prec;
        }
    }

    public double[] thresh(SimpleMatrix pred) {
        double[] preds = new double[pred.getNumRows()];
        for (int i = 0; i < preds.length; i++) {
            if (pred.get(i) > threshold) {
                preds[i] = 1.0;
            } else {
                preds[i] = 0.0;
            }
        }

        return preds;
    }
}
