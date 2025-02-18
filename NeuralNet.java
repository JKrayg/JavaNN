import java.util.ArrayList;

public class NeuralNet {
    private ArrayList<Layer> layers;

    public void addLayer(Layer l) {
        if (this.layers != null) {
            // set forward connections
            for (Node pNode: this.layers.get(this.layers.size() - 1).getNodes()) {
                for (Node nNode: l.getNodes()) {
                    pNode.setForwConnection(nNode);
                }
            }

            // set backward connections
            for (Node nNode: l.getNodes()) {
                for (Node pNode: this.layers.get(this.layers.size() - 1).getNodes()) {
                    nNode.setBackConnection(pNode);
                }
            }
        } else {
            layers = new ArrayList<>();
        }

        this.layers.add(l);
        
    }

    public ArrayList<Layer> getLayers() {
        return layers;
    }

}