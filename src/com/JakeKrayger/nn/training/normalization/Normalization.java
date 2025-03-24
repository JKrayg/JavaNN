package src.com.JakeKrayger.nn.training.normalization;

import org.ejml.simple.SimpleMatrix;

public abstract class Normalization {
    public abstract boolean isBeforeActivation();
    public abstract SimpleMatrix normalize(SimpleMatrix z);
}
