package src.com.JakeKrayger.nn.training.regularizers;

public class L2 extends Regularizer {
    private double lambda;

    public L2() {
        this.lambda = 0.01;
    }

    public L2(double lam) {
        this.lambda = lam;
    }
    
}
