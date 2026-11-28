package fusion;

import java.util.Locale;
import java.util.function.Supplier;
import com.anylogic.engine.Agent;

/** One preview iteration uses the same read-only index-to-scene convention as RL_Training. */
public final class TrainingTopologyPreview {
    private TrainingTopologyPreview() {}
    private static final String PREFIX = "training-layout:";
    private static final String DIRECTORY = "/home/jinx/Faith/experiments/lagrangian_ppo_v1/change_plans/token_attention_001/business_graph_summary_009/variable_topology_training_001/compiled/";
    private static final String[] FILES = {"tree_A.properties", "loop_B.properties"};
    private static final String[] SHA256 = {
        "03828278ea2d57d4664e0f9dd19531f93c5482140ec44d1a878c163ba4ffdba0",
        "701b8ced3ff63d22ba62b0818461ed8fbed48a33e7d0fe0decc5c1dd51565198"
    };

    public static String reference(int index) { return PREFIX + DIRECTORY + FILES[index]; }
    public static String sha(int index) { return SHA256[index]; }
    public static boolean selects(String reference) { return reference != null && reference.startsWith(PREFIX); }

    public static void build(Agent owner, String reference, String expectedSha, Supplier<Agent> createVehicle) {
        TrainingTopology topology = TrainingTopology.load(owner, reference.substring(PREFIX.length()), expectedSha);
        owner.getPresentationShape().add(topology.level());
        for (int i = 0; i < 4; i++) {
            String[] row = topology.setting("load." + i).split(",");
            topology.node(Integer.parseInt(row[0]));
            if (row.length != 4 || !row[1].matches("[01]")) throw new IllegalStateException("Invalid load site row");
        }
        for (int i = 0; i < 2; i++) {
            String[] row = topology.setting("dump." + i).split(",");
            topology.node(Integer.parseInt(row[0]));
            if (row.length != 2 || !row[1].equals(i == 0 ? "ore" : "waste")) {
                throw new IllegalStateException("Invalid dump site role");
            }
        }
        int active = 0;
        for (int i = 0; i < 9; i++) {
            String[] row = topology.setting("vehicle." + i).split(",");
            if (row.length != 4 || !row[0].matches("[01]")) throw new IllegalStateException("Invalid truck row");
            if ("1".equals(row[0])) {
                Agent vehicle = createVehicle.get();
                vehicle.setLevel(topology.level());
                vehicle.setNetwork(topology.network());
                var start = topology.node(Integer.parseInt(row[3]));
                vehicle.setLocation(start);
                if (vehicle.getX() != start.getX() || vehicle.getY() != start.getY()) {
                    throw new IllegalStateException("Truck is outside its configured start node");
                }
                active++;
            }
        }
        if (active < 6 || active > 9) throw new IllegalStateException("Invalid active truck count");
        System.out.printf(Locale.ROOT, "[TOPOLOGY_PREFLIGHT] layout=%s sha256=%s nodes=%d roads=%d active=%d load=4 dump=2%n",
                topology.layoutId(), topology.sceneSha256(), topology.nodeSpecs().size(), topology.roads().size(), active);
    }
}
