import java.util.ArrayList;

public class OutputLayer extends Layer {
    public OutputLayer(ArrayList<Node> nodes) {
        super(nodes);
    }

    public OutputLayer(int numNodes) {
        super(createNodes(numNodes));
    }

    public void setWeights(double w) {
        for (Node n: this.getNodes()) {
            n.setWeight(w);
        }
    }

    private static ArrayList<Node> createNodes(int numNodes) {
        ArrayList<Node> nodes = new ArrayList<>();
        for (int k = 0; k < numNodes; k++) {
            nodes.add(new OutputNode(0, 0, new ReLU()));
        }

        return nodes;
    }
}