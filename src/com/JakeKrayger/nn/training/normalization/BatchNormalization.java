package src.com.JakeKrayger.nn.training.normalization;

import org.ejml.simple.SimpleMatrix;

import src.com.JakeKrayger.nn.training.optimizers.Optimizer;

public class BatchNormalization extends Normalization {
    private SimpleMatrix scale;
    private SimpleMatrix shift;
    private SimpleMatrix means;
    private SimpleMatrix variances;
    private SimpleMatrix shiftMomentum;
    private SimpleMatrix shiftVariance;
    private SimpleMatrix scaleMomentum;
    private SimpleMatrix scaleVariance;
    private double momentum = 0.99;
    private double epsilon = 1e-3;
    private boolean beforeActivation = true;
    private SimpleMatrix gradientWrtShift;
    private SimpleMatrix gradientWrtScale;
    private SimpleMatrix normalizedZ;

    public BatchNormalization() {}

    public void setScale(SimpleMatrix scale) {
        this.scale = scale;
    }

    public void setShift(SimpleMatrix shift) {
        this.shift = shift;
    }

    public void setMeans(SimpleMatrix means) {
        this.means = means;
    }

    public void setVariances(SimpleMatrix variances) {
        this.variances = variances;
    }

    public void setMomentum(double momentum) {
        this.momentum = momentum;
    }

    public void setEpsilon(double epsilon) {
        this.epsilon = epsilon;
    }

    public void setShiftMomentum(SimpleMatrix shM) {
        this.shiftMomentum = shM;
    }

    public void setShiftVariance(SimpleMatrix shV) {
        this.shiftVariance = shV;
    }

    public void setScaleMomentum(SimpleMatrix scM) {
        this.scaleMomentum = scM;
    }

    public void setScaleVariance(SimpleMatrix scV) {
        this.scaleVariance = scV;
    }

    public void setGradientShift(SimpleMatrix gWrtSh) {
        this.gradientWrtShift = gWrtSh;
    }

    public void setGradientScale(SimpleMatrix gWrtSc) {
        this.gradientWrtShift = gWrtSc;
    }

    public void beforeActivation(boolean b) {
        this.beforeActivation = b;
    }

    public boolean isBeforeActivation() {
        return beforeActivation;
    }

    public SimpleMatrix getScale() {
        return scale;
    }

    public SimpleMatrix getShift() {
        return shift;
    }

    public SimpleMatrix getMeans() {
        return means;
    }

    public SimpleMatrix getVariances() {
        return variances;
    }

    public double getMomentum() {
        return momentum;
    }

    public SimpleMatrix getShiftMomentum() {
        return shiftMomentum;
    }

    public SimpleMatrix getShiftVariance() {
        return shiftVariance;
    }

    public SimpleMatrix getScaleMomentum() {
        return scaleMomentum;
    }

    public SimpleMatrix getScaleVariance() {
        return scaleVariance;
    }

    public SimpleMatrix getGradientShift() {
        return gradientWrtShift;
    }

    public SimpleMatrix getGradientScale() {
        return gradientWrtScale;
    }

    public SimpleMatrix gradientShift(SimpleMatrix gradient) {
        SimpleMatrix gWrtSh = new SimpleMatrix(gradient.getNumCols(), 1);
        int cols = gradient.getNumCols();

        for (int i = 0; i < cols; i++) {
            gWrtSh.set(i, 0, gradient.getColumn(i).elementSum());
        }

        return gWrtSh;
    }

    public SimpleMatrix gradientScale(SimpleMatrix gradient) {
        SimpleMatrix gWrtSc = new SimpleMatrix(gradient.getNumCols(), 1);
        int cols = gradient.getNumCols();

        for (int i = 0; i < cols; i++) {
            gWrtSc.set(i, 0, gradient.getColumn(i).elementMult(normalizedZ.getColumn(i)).elementSum());
        }
        return gWrtSc;
    }

    public void updateScale(Optimizer o) {
        this.setScale(o.executeScaleUpdate(this));
    }

    public void updateShift(Optimizer o) {
        this.setScale(o.executeShiftUpdate(this));
    }

    public SimpleMatrix normalize(SimpleMatrix z) {
        int rows = z.getNumRows();
        int cols = z.getNumCols();
        SimpleMatrix means = new SimpleMatrix(cols, 1);
        SimpleMatrix variances = new SimpleMatrix(cols, 1);
        SimpleMatrix norm = new SimpleMatrix(rows, cols);
        // if (scale == null && shift == null) {
        //     scale = new SimpleMatrix(cols, 1);
        //     scale.fill(1.0);
        //     shift = new SimpleMatrix(cols, 1);
        // }
        
        for (int i = 0; i < cols; i++) {
            SimpleMatrix currCol = z.getColumn(i);
            double currMean = currCol.elementSum() / rows;
            means.set(i, 0, currMean);

            double currVariance = currCol.minus(currMean).elementPower(2).elementSum() / rows;
            variances.set(i, 0, currVariance);

            SimpleMatrix set = currCol.minus(currMean).divide(Math.sqrt(currVariance + epsilon));
            norm.setColumn(i, set.scale(scale.get(i)).plus(shift.get(i)));
        }

        this.normalizedZ = norm;

        return norm;
    }
}
