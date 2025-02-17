public class Node {
    private double value;
    private double weight;
    private double bias;
    private ActivationFunction func;
    private Layer connections;

    public Node(double value, double weight, double bias, ActivationFunction func) {
        this.value = value;
        this.weight = weight;
        this.bias = bias;
        this.func = func;
    }

    public Node(double value, double bias, ActivationFunction func) {
        this.value = value;
        this.bias = bias;
        this.func = func;
    }

    public Node(double value) {
        this.value = value;
    }

    public double getValue() {
        return value;
    }

    public double getWeight() {
        return weight;
    }

    public double getBias() {
        return bias;
    }

    public ActivationFunction getActFunc() {
        return func;
    }

    public Layer getConnections() {
        return connections;
    }

    public void setWeight(double weight) {
        this.weight = weight;
    }
}