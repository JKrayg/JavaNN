import java.util.ArrayList;

public class Dense extends Layer {
    private ActivationFunction func;

    public Dense(int numNodes, String actFunc) {
        super(createNodes(numNodes, actFunc));
        if (actFunc.equals("relu")) {
            this.func = new ReLU();
        } else if (actFunc.equals("sigmoid")) {
            this.func = new Sigmoid();
        } else {
            this.func = new Softmax();
        }
    }

    private static ArrayList<Node> createNodes(int numNodes, String actFunc) {
        ArrayList<Node> nodes = new ArrayList<>();
        for (int i = 0; i < numNodes; i++) {
            if (actFunc.equals("relu")) {
                nodes.add(new HiddenNode(0, 0, new ReLU()));
            } else {
                nodes.add(new HiddenNode(0, 0, new Sigmoid()));
            }
            
        }

        return nodes;
    }

    public String getActFunc() {
        if (this.func instanceof ReLU) {
            return "relu";
        } else if (this.func instanceof Sigmoid) {
            return "sigmoid";
        } else {
            return "softmax";
        }
    }

    public void setWeights(InputLayer in, OutputLayer out, String actFunc) {
        for (Node n: this.getNodes()) {
            n.setWeight(new GlorotInit().initWeight(in, out, actFunc));
        }
    }
}