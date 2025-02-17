import java.util.ArrayList;
import java.util.Arrays;

public class Main {
    public static void main(String[] args) {
        int numDenseLayers = 4;
        int numDenseNodes = 5;
        int numOutputNodes = 1;
        ArrayList<Node> inNodes = new ArrayList<>();
        ArrayList<Node> outNodes = new ArrayList<>();
        ArrayList<Double> data = new ArrayList<>(Arrays.asList(34.0, 180.0, 74.0));
        InputLayer in;
        ArrayList<Dense> denseLayers = new ArrayList<>();
        OutputLayer out;

        // init input layer
        for (double i: data) {
            inNodes.add(new InputNode(i));
        }
        in = new InputLayer(inNodes);

        // init output layer
        for (int k = 0; k < numOutputNodes; k++) {
            outNodes.add(new OutputNode(0, 0, new ReLU()));
        }
        out = new OutputLayer(outNodes);

        // init weights for output nodes
        for (Node n: out.getNodes()) {
            n.setWeight(new GlorotInit().initWeight(in, out, "relu"));
        }

        // for (Node n: out.getNodes()) {
        //     System.out.println(n.getValue());
        //     System.out.println(n.getWeight());
        //     System.out.println(n.getBias());
        //     System.out.println(n.getActFunc());
        // }

        // init hidden layers
        for (int i = 0; i < numDenseLayers; i++) {
            ArrayList<Node> dNodes = new ArrayList<>();
            for (int j = 0; j < numDenseNodes; j++) {
                dNodes.add(new HiddenNode(0, new GlorotInit().initWeight(in, out, "relu"), 0, new ReLU()));
            }
            Dense d = new Dense(dNodes);
            denseLayers.add(d);
        }

        // for (Layer l: denseLayers) {
        //     System.out.println("Layer------");
        //     for (Node n: l.getNodes()) {
        //         System.out.println(n.getValue());
        //         System.out.println(n.getWeight());
        //         System.out.println(n.getBias());
        //         System.out.println(n.getActFunc());
        //         System.out.println("..............\n");
        //     }
            
        // }

        System.out.println(in.getNodes());
        for (Layer l: denseLayers) {
            System.out.println(l.getNodes());
        }
        
        System.out.println(out.getNodes());

    }
}
