package src.com.JakeKrayger.nn.training.metrics;

import java.util.Arrays;

import org.ejml.simple.SimpleMatrix;

public class MultiClassMetrics extends Metrics{
    private double threshold;

    public MultiClassMetrics() {
        this.threshold = 0.5;
    }

    public MultiClassMetrics(double threshold) {
        this.threshold = threshold;
    }

    public void getMetrics(SimpleMatrix pred, SimpleMatrix trueVals) {
        String dis = "Accuracy: " + accuracy(pred, trueVals) + "\n";
        dis += "Precision: ";
        for (double d: precision(pred, trueVals)) {
            dis += Double.toString(d) + ", ";
        }
        dis += "\n";
        // dis += "Precision: " + precision(pred, trueVals) + "\n";
        dis += "Recall: " + recall(pred, trueVals) + "\n";
        System.out.println(dis);
    }

    public double accuracy(SimpleMatrix pred, SimpleMatrix trueVals) {
        int correct = 0;
        double[][] preds = thresh(pred);

        for (int i = 0; i < preds.length; i++) {
            for (int j = 0; j < preds[0].length; j++) {
                if (preds[i][j] == 1.0 && trueVals.get(i, j) == 1.0) {
                    correct += 1;
                }
            }
            
        }
        return ((double) correct) / ((double) preds.length);
    }


    public double[] precision(SimpleMatrix pred, SimpleMatrix trueVals) {
        // int correct = 0;
        // int wrong = 0;
        double[][] preds = thresh(pred);
        double[] classPrecisions = new double[preds[0].length];

        for (int i = 0; i < preds[0].length; i++) {
            SimpleMatrix currClassPred = new SimpleMatrix(new SimpleMatrix(preds).getColumn(i));
            SimpleMatrix currClassTrue = trueVals.getColumn(i);
            int tp = 0;
            int fp = 0;
            for (int j = 0; j < preds.length; j++) {
                if (currClassPred.get(j) == 1.0 && currClassTrue.get(j) == 1.0) {
                    tp += 1;
                } else if (currClassPred.get(j) == 1.0) {
                    fp += 1;
                }
            }

            double prec = ((double) tp) / (((double) tp) + ((double) fp));
            if (Double.isNaN(prec)) {
                classPrecisions[i] = 0.0;
            } else {
                classPrecisions[i] = prec;
            }
            for (int j = 0; j < preds[0].length; j++) {
                if (trueVals.get(i, j) == 1.0 && preds[i][j] == 1.0) {
                    tp += 1;
                } else if (preds[i][j] == 1.0) {
                    fp += 1;
                }
            }  
        }

        return classPrecisions;
    }

    public double recall(SimpleMatrix pred, SimpleMatrix trueVals) {
        return 0.0;
    }

    public double f1(SimpleMatrix pred, SimpleMatrix trueVals) {
        return 0.0;
    }

    public double[][] thresh(SimpleMatrix pred) {
        double[][] preds = new double[pred.getNumRows()][pred.getNumCols()];
        for (int i = 0; i < pred.getNumRows(); i++) {
            for (int j = 0; j < pred.getNumCols(); j++) {
                double highestProb = pred.getRow(i).elementMax();
                if (pred.get(i, j) == highestProb) {
                    preds[i][j] = 1.0;
                } else {
                    preds[i][j] = 0.0;
                }

            }
        }

        return preds;
    }
    
}
