package fusion;

import static com.anylogic.engine.TimeUnits.SECOND;

import DRL.grpc.Observation;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.google.gson.JsonPrimitive;
import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;

/** Dispatches from the live AnyLogic observation using the frozen local Actor. */
public final class TopologyPolicyDispatch {
    private static final URI ENDPOINT = URI.create("http://127.0.0.1:50871/decide");
    private static final int OBSERVATION_SIZE = 1519;
    private static final int ACTION_MASK_START = 19;
    private static final int ACTION_COUNT = 4;
    private static final int TRUCK_FEATURE_START = 27;
    private static final int TRUCK_FEATURE_SIZE = 120;
    private static final int TRUCK_SLOTS = 9;
    private static final int ROAD_SLOTS = 16;
    private static final Duration TIMEOUT = Duration.ofSeconds(15);

    private record Decision(long sequence, int truck, double time, int action) {}

    private final Main main;
    private final TrainingTopology topology;
    private final String configSha;
    private final String checkpointSha;
    private final HttpClient client = HttpClient.newBuilder().connectTimeout(TIMEOUT).build();
    private Decision pending;
    private long applied;

    TopologyPolicyDispatch(Main main, TrainingTopology topology, String configSha, String checkpointSha) {
        this.main = main;
        this.topology = topology;
        this.configSha = configSha;
        this.checkpointSha = checkpointSha;
        require(checkpointSha.matches("[0-9a-f]{64}"), "Invalid frozen checkpoint SHA256");
    }

    String checkpointSha() { return checkpointSha; }

    List<String> roadKeys() {
        for (var node : topology.nodeSpecs()) {
            require(main.obsNodeIndex(node.canonicalName()) == node.wireId(),
                    "Active road node disagrees with the pretrained node slot: " + node);
        }
        var keys = new ArrayList<String>();
        for (var road : topology.roads()) {
            String from = main.obsNodeNames[road.fromNodeId()];
            String to = main.obsNodeNames[road.toNodeId()];
            keys.add(from + "_" + to);
            keys.add(to + "_" + from);
        }
        require(!keys.isEmpty() && keys.size() <= ROAD_SLOTS && keys.size() == main.segmentMap.size()
                && new HashSet<>(keys).equals(main.segmentMap.keySet()),
                "Active road slots do not match the instantiated road network");
        return keys;
    }

    int choose(Truck truck, long sequence) {
        require(pending == null && sequence == applied, "An earlier dispatch never committed");
        int dispatchable = main.fleetResilience.dispatchableMask;
        int inSystem = main.fleetResilience.inSystemMask;
        require((dispatchable & ~inSystem) == 0 && (dispatchable & (1 << truck.truckId)) != 0,
                "Dispatching truck is unavailable");
        double now = main.time(SECOND);
        var state = main.buildPhysicalObservation(truck);
        require("dispatch.physical.v3".equals(state.getSchemaId())
                && state.getValuesCount() == OBSERVATION_SIZE, "Unexpected native observation schema");
        checkFleetFields(state, truck.truckId, inSystem);

        var obs = new JsonArray();
        for (float value : state.getValuesList()) {
            require(Float.isFinite(value), "Non-finite native observation");
            obs.add(value);
        }
        int[] legal = new int[ACTION_COUNT];
        var mask = new JsonArray();
        int count = 0;
        for (int i = 0; i < ACTION_COUNT; i++) {
            float encoded = state.getValues(ACTION_MASK_START + i);
            require(encoded == 0 || encoded == 1, "Invalid native action-mask entry");
            legal[i] = (int) encoded;
            count += legal[i];
            mask.add(legal[i]);
        }
        require(count > 0, "The frozen Actor was asked to dispatch without a legal load site");

        var body = new JsonObject();
        body.addProperty("run_index", main.experimentRunIndex);
        body.addProperty("environment_seed", main.experimentSeed);
        body.addProperty("recording_id", main.episodeId);
        body.addProperty("config_sha256", configSha);
        body.addProperty("layout_id", topology.layoutId());
        body.addProperty("decision_sequence", sequence);
        body.addProperty("truck_id", truck.truckId);
        body.addProperty("sim_time_seconds", now);
        body.addProperty("dispatchable_mask", dispatchable);
        body.addProperty("in_system_mask", inSystem);
        body.add("obs", obs);
        body.add("action_mask", mask);

        JsonObject reply = post(body);
        require(configSha.equals(string(reply, "config_sha256"))
                && checkpointSha.equals(string(reply, "checkpoint_sha256"))
                && integer(reply, "decision_sequence") == sequence,
                "Frozen policy replied with mismatched decision identity");
        long action = integer(reply, "action");
        require(action >= 0 && action < ACTION_COUNT && legal[(int) action] == 1,
                "Frozen policy selected a masked action");
        pending = new Decision(sequence, truck.truckId, now, (int) action);
        return pending.action();
    }

    void committed(Truck truck, long sequence) {
        require(pending != null && sequence == pending.sequence() && truck.truckId == pending.truck()
                && main.time(SECOND) == pending.time() && truck.targetLoadSiteId == pending.action()
                && truck.plannedPayloadTons > 0, "Native reservation differs from the Actor decision");
        pending = null;
        applied++;
    }

    void requireComplete(long sequence) {
        require(pending == null && applied > 0 && applied == sequence, "Uncommitted Actor decision at shift end");
    }

    private void checkFleetFields(Observation state, int ego, int inSystem) {
        for (int truck = 0; truck < TRUCK_SLOTS; truck++) {
            int start = TRUCK_FEATURE_START + truck * TRUCK_FEATURE_SIZE;
            require(state.getValues(start) == (truck == ego ? 1 : 0), "Wrong ego truck in native observation");
            if ((inSystem & (1 << truck)) != 0) continue;
            require(state.getValues(start + 15) == 1, "Absent truck is outside the parked phase");
            requireZero(state, start + 4, start + 15);
            requireZero(state, start + 17, start + 41);
            requireZero(state, start + 43, start + TRUCK_FEATURE_SIZE);
        }
    }

    private void requireZero(Observation state, int start, int end) {
        for (int i = start; i < end; i++) {
            require(state.getValues(i) == 0, "Inactive truck has physical observation fields at " + i);
        }
    }

    private JsonObject post(JsonObject body) {
        HttpRequest request = HttpRequest.newBuilder(ENDPOINT).timeout(TIMEOUT)
                .header("Content-Type", "application/json; charset=utf-8")
                .POST(HttpRequest.BodyPublishers.ofString(body.toString())).build();
        try {
            var response = client.send(request, HttpResponse.BodyHandlers.ofString());
            require(response.statusCode() == 200, "Frozen policy RPC returned HTTP " + response.statusCode());
            return JsonParser.parseString(response.body()).getAsJsonObject();
        } catch (InterruptedException error) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("Frozen policy RPC interrupted", error);
        } catch (IOException error) {
            throw new IllegalStateException("Frozen policy RPC unavailable", error);
        }
    }

    private static String string(JsonObject reply, String field) {
        JsonPrimitive value = reply.getAsJsonPrimitive(field);
        require(value != null && value.isString(), "Wrong RPC response field: " + field);
        return value.getAsString();
    }

    private static long integer(JsonObject reply, String field) {
        JsonPrimitive value = reply.getAsJsonPrimitive(field);
        require(value != null && value.isNumber(), "Wrong RPC response field: " + field);
        return value.getAsBigDecimal().longValueExact();
    }

    private static void require(boolean condition, String message) {
        if (!condition) throw new IllegalStateException(message);
    }
}
