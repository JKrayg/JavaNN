package src.com.JakeKrayger.nn.training.normalization;

import org.ejml.simple.SimpleMatrix;

import src.com.JakeKrayger.nn.training.optimizers.Optimizer;

public class BatchNormalization extends Normalization {
    private SimpleMatrix scale;
    private SimpleMatrix shift;
    private SimpleMatrix means;
    private SimpleMatrix variances;
    private SimpleMatrix runningMeans;
    private SimpleMatrix runningVariances;
    private SimpleMatrix shiftMomentum;
    private SimpleMatrix shiftVariance;
    private SimpleMatrix scaleMomentum;
    private SimpleMatrix scaleVariance;
    private double momentum = 0.99;
    private double epsilon = 1e-3;
    private boolean beforeActivation = true;
    private SimpleMatrix gradientWrtShift;
    private SimpleMatrix gradientWrtScale;
    private SimpleMatrix preNormZ;
    private SimpleMatrix preScaleShiftZ;
    private SimpleMatrix normalizedZ;

    public BatchNormalization() {
    }

    public void setScale(SimpleMatrix scale) {
        this.scale = scale;
    }

    public void setShift(SimpleMatrix shift) {
        this.shift = shift;
    }

    public void setMeans(SimpleMatrix means) {
        this.means = means;
    }

    public void setRunningMeans(SimpleMatrix means) {
        this.runningMeans = means;
    }

    public void setVariances(SimpleMatrix variances) {
        this.variances = variances;
    }
    

    public void setRunningVariances(SimpleMatrix variances) {
        this.runningVariances = variances;
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
        this.gradientWrtScale = gWrtSc;
    }

    public void beforeActivation(boolean b) {
        this.beforeActivation = b;
    }

    public boolean isBeforeActivation() {
        return beforeActivation;
    }

    public double getEpsilon() {
        return epsilon;
    }

    public SimpleMatrix getScale() {
        return scale;
    }

    public SimpleMatrix getShift() {
        return shift;
    }

    public SimpleMatrix getMeans() {
        return runningMeans;
    }

    public SimpleMatrix getVariances() {
        return runningVariances;
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

    public SimpleMatrix getNormZ() {
        return normalizedZ;
    }

    public SimpleMatrix getPreScaleShiftZ() {
        return preScaleShiftZ;
    }

    public SimpleMatrix getPreNormZ() {
        return preNormZ;
    }

    public SimpleMatrix gradientShift(SimpleMatrix gradient) {
        int cols = gradient.getNumCols();
        SimpleMatrix gWrtSh = new SimpleMatrix(cols, 1);

        for (int i = 0; i < cols; i++) {
            gWrtSh.set(i, 0, gradient.getColumn(i).elementSum());
        }

        return gWrtSh;
    }

    public SimpleMatrix gradientScale(SimpleMatrix gradient) {
        SimpleMatrix gWrtSc = new SimpleMatrix(gradient.getNumCols(), 1);
        int cols = gradient.getNumCols();

        for (int i = 0; i < cols; i++) {
            gWrtSc.set(i, 0, gradient.getColumn(i).elementMult(preScaleShiftZ.getColumn(i)).elementSum());
        }
        return gWrtSc;
    }

    public void updateScale(Optimizer o) {
        this.setScale(o.executeScaleUpdate(this));
    }

    public void updateShift(Optimizer o) {
        this.setShift(o.executeShiftUpdate(this));
    }

    public SimpleMatrix means(SimpleMatrix z) {
        int rows = z.getNumRows();
        int cols = z.getNumCols();
        SimpleMatrix means = new SimpleMatrix(cols, 1);

        for (int i = 0; i < cols; i++) {
            means.set(i, z.getColumn(i).elementSum() / rows);
        }

        return means;
    }

    public SimpleMatrix variances(SimpleMatrix z) {
        int rows = z.getNumRows();
        int cols = z.getNumCols();
        SimpleMatrix variances = new SimpleMatrix(cols, 1);
        SimpleMatrix means = means(z);

        for (int i = 0; i < cols; i++) {
            variances.set(i, z.getColumn(i).minus(means.get(i)).elementPower(2).elementSum() / rows);
        }

        return variances;
    }

    public SimpleMatrix normalize(SimpleMatrix z) {
        int rows = z.getNumRows();
        int cols = z.getNumCols();
        SimpleMatrix means = means(z);
        SimpleMatrix variances = variances(z);
        SimpleMatrix preSclShft = new SimpleMatrix(rows, cols);
        SimpleMatrix norm = new SimpleMatrix(rows, cols);

        for (int i = 0; i < cols; i++) {
            SimpleMatrix currCol = z.getColumn(i);
            SimpleMatrix normalizedCol = currCol.minus(means.get(i)).divide(Math.sqrt(variances.get(i) + epsilon));
            preSclShft.setColumn(i, normalizedCol);
            norm.setColumn(i, normalizedCol.scale(scale.get(i)).plus(shift.get(i)));
        }

        this.means = means;
        this.variances = variances;
        // this.runningMeans = runningMeans.scale(momentum).plus(means.scale((1 - momentum)));
        // this.runningVariances = runningVariances.scale(momentum).plus(variances.scale((1 - momentum)));
        this.preNormZ = z.copy();
        this.preScaleShiftZ = preSclShft;

        return norm;
    }

    public void setPreNormZ(SimpleMatrix z) {
        this.preNormZ = z;
    }

    public SimpleMatrix gradientPreBN(SimpleMatrix dLdzHat) {
        // clean this
        int rows = dLdzHat.getNumRows();
        int cols = dLdzHat.getNumCols();

        SimpleMatrix incoming = dLdzHat.copy();
        SimpleMatrix preNormZed = this.preNormZ.copy();
        SimpleMatrix scalingFactor = scale.elementDiv(variances.plus(epsilon).elementPower(0.5));
        SimpleMatrix part2 = new SimpleMatrix(cols, 1);
        SimpleMatrix part4BeforeSum = new SimpleMatrix(rows, cols);
        SimpleMatrix part4Final = new SimpleMatrix(cols, 1);
        SimpleMatrix converted = new SimpleMatrix(rows, cols);
        SimpleMatrix variancesPlusEpsilon = variances.plus(epsilon);

        for (int i = 0; i < rows; i++) {
            part4BeforeSum.setRow(i, incoming.getRow(i).elementMult(preNormZed.getRow(i).minus(means.transpose())));
        }

        for (int j = 0; j < cols; j++) {
            part2.set(j, incoming.getColumn(j).elementSum() / rows);
            part4Final.set(j, part4BeforeSum.transpose().getRow(j).elementSum());
        }

        for (int k = 0; k < rows; k++) {
            SimpleMatrix part3 = preNormZed.getRow(k).minus(means.transpose()).elementDiv(variancesPlusEpsilon.transpose()).elementMult(part4Final.transpose());
            converted.setRow(k, scalingFactor.transpose().elementMult(incoming.getRow(k).minus(part2.transpose()).minus(part3)));
        }

        return converted;
    }
}
