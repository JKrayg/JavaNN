package src.com.JakeKrayger.nn.activation;

import org.ejml.simple.SimpleMatrix;

public abstract class ActivationFunction {
    public abstract SimpleMatrix execute(SimpleMatrix z);
}
