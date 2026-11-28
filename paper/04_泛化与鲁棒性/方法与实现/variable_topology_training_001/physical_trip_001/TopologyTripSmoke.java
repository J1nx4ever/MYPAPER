package fusion;

import static com.anylogic.engine.LengthUnits.METER;
import static com.anylogic.engine.SpeedUnits.MPS;
import static com.anylogic.engine.TimeUnits.SECOND;

import com.anylogic.engine.Point;
import com.anylogic.engine.markup.ContinuousShortestPathData;
import com.anylogic.engine.markup.PointNode;
import com.google.gson.Gson;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Random;

/** Runs the production Truck and PML steps for one frozen trip, with no policy RPC. */
public final class TopologyTripSmoke {
    private static final String DIRECTORY = "/home/jinx/Faith/experiments/lagrangian_ppo_v1/change_plans/token_attention_001/business_graph_summary_009/variable_topology_training_001/physical_trip_001/configs/";
    private static final String[] CASE_FILES = {"A.properties", "B.properties"};
    private static final String[] CASE_SHAS = {
        "cd2c7a5df118c1d62a9aec9e1af4f0ea6d02e485febb88dc5ca2eea035fe00ec",
        "d5e4ce04209dc4a290e245467fa9d378a7897a2f99ea1152f6d03a4533eb7791"
    };
    private static final double POSITION_TOLERANCE_M = 0.001;
    private static final double MASS_TOLERANCE_T = 1e-6;

    private final Main main;
    private final Properties config;
    private final String configSha;
    private final TrainingTopology topology;
    private final int truckId;
    private final int loadSlot;
    private final double timeout;
    private final Map<Integer, TrainingTopology.NodeSpec> nodes = new HashMap<>();
    private final Map<String, Integer> wireIds = new HashMap<>();
    private final Map<String, Integer> roadIds = new HashMap<>();
    private final Gson gson = new Gson();
    private final List<String> outbound = new ArrayList<>();
    private final List<String> inbound = new ArrayList<>();
    private PrintWriter events;
    private Truck probe;
    private PathSegment travelling;
    private double departureTime;
    private double lastProgress;
    private boolean sampledMiddle;
    private boolean dispatched;
    private boolean passed;

    public static int caseNumber(int index) {
        if (index < 0 || index >= CASE_FILES.length) throw new IllegalArgumentException("Unexpected case " + index);
        return index + 1;
    }

    public static TopologyTripSmoke open(Main main, int caseNumber) {
        int idx = caseNumber - 1;
        if (idx < 0 || idx >= CASE_FILES.length) throw new IllegalArgumentException("Invalid physical trip case");
        try {
            byte[] bytes = Files.readAllBytes(Path.of(DIRECTORY, CASE_FILES[idx]));
            String sha = HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(bytes));
            if (!CASE_SHAS[idx].equals(sha)) throw new IllegalArgumentException("Physical trip config SHA mismatch");
            Properties config = new Properties();
            config.load(new ByteArrayInputStream(bytes));
            return new TopologyTripSmoke(main, config, sha);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException(e);
        }
    }

    private TopologyTripSmoke(Main main, Properties config, String configSha) {
        this.main = main;
        this.config = config;
        this.configSha = configSha;
        require("topology.physical-trip.v1".equals(config.getProperty("format")), "Wrong config contract");
        truckId = Integer.parseInt(config.getProperty("truck_id"));
        loadSlot = Integer.parseInt(config.getProperty("load_slot"));
        timeout = Double.parseDouble(config.getProperty("stop_seconds"));
        topology = TrainingTopology.load(main, config.getProperty("scene_path"), config.getProperty("scene_sha256"));
        require(topology.layoutId().equals(config.getProperty("layout_id")), "Layout identity mismatch");
        for (var node : topology.nodeSpecs()) {
            nodes.put(node.wireId(), node);
            wireIds.put(node.canonicalName(), node.wireId());
        }
    }

    public void start() {
        require(now() == 0.0 && !main.episodeStarted, "Single trip must start from a fresh Main");
        main.experimentRunIndex = Long.parseLong(config.getProperty("run_index"));
        main.experimentSeed = Long.parseLong(config.getProperty("environment_seed"));
        main.getEngine().setDefaultRandomGenerator(new Random(main.experimentSeed));
        main.getEngine().setStopTime(timeout, SECOND);
        main.numTrucks = 9;
        main.deterministicCheck = true;
        main.dynamic = true;
        main.useLocalDynamicSpeed = false;
        main.enableOnnxDispatch = false;
        main.enableBoundaryEventRecording = false;
        main.enableMaterialEventMemory = false;
        main.enableP0SnapshotAudit = false;
        main.enableWasteTaskLimit = false;
        main.isWarmUp = false;
        main.enableMaterialAudit = true;
        main.materialAuditReady = false;
        main.currentWindowIndex = 0;
        main.waitBreakdownLedger = new WaitBreakdownLedger(main.rlShiftSeconds);

        int disabled = 0;
        for (int i = 0; i < 9; i++) {
            if ("0".equals(row("vehicle", i)[0])) disabled |= 1 << i;
        }
        main.fleetDisabledTruckMask = disabled;
        main.fleetFaultTruckId = -1;
        main.fleetFaultTriggerSeconds = -1;
        main.fleetResilience = new FleetResilienceState(disabled, -1, -1.0);
        require((disabled & (1 << truckId)) == 0, "Probe must be in the configured opening roster");
        installRoadsAndRoutes();
        instantiateSites();
        instantiateFleet();
        main.materialAuditReady = true;
        main.auditMaterialBalances("TOPOLOGY_TRIP_INITIAL");

        try {
            Path directory = Path.of(config.getProperty("event_directory"));
            Files.createDirectories(directory);
            events = new PrintWriter(Files.newBufferedWriter(directory.resolve("events.jsonl"),
                    StandardOpenOption.CREATE_NEW, StandardOpenOption.WRITE));
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        int start = Integer.parseInt(row("vehicle", truckId)[3]);
        require(start == 0 && Integer.parseInt(row("load", loadSlot)[1]) == 1, "Probe contract must be an ore round trip");
        verifyPosition(probe, topology.node(start));
        log("START", Map.of("physical_nodes", nodes.size(), "physical_roads", topology.roads().size(),
                "active_trucks", Integer.bitCount(main.fleetResilience.openingMask), "truck_id", truckId,
                "load_slot", loadSlot));
        probe.deliver("ACTIVATE", probe);
    }

    private void installRoadsAndRoutes() {
        main.network = topology.network();
        main.getPresentationShape().add(topology.level());
        main.roadPointNodeMap.clear();
        main.segmentMap.clear();
        main.routeNodeMap.clear();
        for (var spec : topology.nodeSpecs()) main.roadPointNodeMap.put(spec.canonicalName(), topology.node(spec.wireId()));
        for (var road : topology.roads()) {
            addSegment(road, false);
            addSegment(road, true);
        }
        for (int i = 0; i < 4; i++) {
            int loadNode = Integer.parseInt(row("load", i)[0]);
            addRoute(0, loadNode);
            addRoute(1, loadNode);
            addRoute(loadNode, "1".equals(row("load", i)[1]) ? 0 : 1);
        }
    }

    private void addSegment(TrainingTopology.Road road, boolean reverse) {
        String a = name(road.fromNodeId());
        String b = name(road.toNodeId());
        var segment = new PathSegment(main, reverse ? b : a, reverse ? a : b, a, b,
                reverse ? road.speedToFromMps() : road.speedFromToMps());
        require(Math.abs(segment.length - road.lengthMeters()) <= POSITION_TOLERANCE_M,
                "Existing geometry lookup does not match the native road length: " + segment.key);
        segment.length = road.lengthMeters();
        main.segmentMap.put(segment.key, segment);
        roadIds.put(segment.key, road.index());
    }

    private void addRoute(int start, int target) {
        var route = new ArrayList<String>();
        for (String id : topology.setting("route." + start + "." + target).split(",")) {
            route.add(name(Integer.parseInt(id)));
        }
        require(route.size() > 1 && route.get(0).equals(name(start))
                && route.get(route.size() - 1).equals(name(target)), "Invalid OD endpoints");
        for (int j = 1; j < route.size(); j++) {
            require(roadIds.containsKey(route.get(j - 1) + "_" + route.get(j)), "OD refers to a missing road");
        }
        main.routeNodeMap.put(name(start) + "_" + name(target), route);
    }

    private void instantiateSites() {
        String[] labels = {"HG1", "HG2", "LG", "WASTE"};
        for (int i = 0; i < 4; i++) {
            String[] data = row("load", i);
            PointNode place = topology.node(Integer.parseInt(data[0]));
            LoadSite shovel = main.add_loadSites();
            attach(shovel, place);
            shovel.loadNode = place;
            shovel.gradeHalfRange = Double.parseDouble(data[3]);
            shovel.initLoadSite(i, labels[i], "1".equals(data[1]), Double.parseDouble(data[2]), 30000,
                    0.45, place.getX(), place.getY(), i == 3 ? 15.0 : 13.5, i == 3 ? 2.0 : 2.2);
            main.loadingPointStatsList.get(i).loadingPointName = loadNode(i);
        }
        for (int i = 0; i < 2; i++) {
            String[] data = row("dump", i);
            PointNode place = topology.node(Integer.parseInt(data[0]));
            DumpSite site = main.add_dumpSites();
            attach(site, place);
            site.dumpNode = place;
            boolean ore = "ore".equals(data[1]);
            site.initDumpSite(i, ore ? "Crusher" : "Dump", ore, !ore, ore ? 2.5 : 2.0);
        }
    }

    private void instantiateFleet() {
        for (int i = 0; i < 9; i++) {
            String[] data = row("vehicle", i);
            Truck truck = main.add_trucks();
            boolean active = "1".equals(data[0]);
            int start = Integer.parseInt(data[3]);
            truck.initTruck(i, Double.parseDouble(data[2]));
            truck.truckType = Integer.parseInt(data[1]);
            truck.initialPredecessor = null;
            truck.initialDispatchDone = i != truckId;
            if (active) {
                attach(truck, topology.node(start));
                truck.currentNode = name(start);
                truck.startPoint = truck.currentNode;
                truck.currentBusinessSiteId = truck.currentNode;
            } else {
                truck.setLocation(main.parking);
                truck.currentNode = "";
            }
            truck.setSpeed(i == truckId ? 6.5 : 0.0, MPS);
            main.waitBreakdownLedger.registerTruck(i);
            if (i == truckId) probe = truck;
        }
    }

    private void attach(com.anylogic.engine.Agent agent, PointNode node) {
        agent.setLevel(topology.level());
        agent.setNetwork(topology.network());
        agent.setLocation(node);
    }

    public int dispatch(Truck truck) {
        require(truck == probe && !dispatched, "Single-trip dispatcher activated more than once");
        require(main.getLoadSiteById(loadSlot).getDispatchableTons(truck) > 0.0, "Selected source is not legal");
        dispatched = true;
        return loadSlot;
    }

    public String loadNode(int slot) { return name(Integer.parseInt(row("load", slot)[0])); }
    public String dumpNode(int slot) { return name(Integer.parseInt(row("dump", slot)[0])); }

    public int modelNodeNumber(PointNode node) {
        for (var spec : topology.nodeSpecs()) {
            if (topology.node(spec.wireId()) == node) return main.roadNodeNameToInt(spec.canonicalName());
        }
        throw new IllegalArgumentException("Unknown native route node");
    }

    public void departing(Truck truck, PointNode target) {
        require(truck == probe && travelling == null && truck.currentSegment != null, "Invalid departure");
        PathSegment segment = truck.currentSegment;
        verifyPosition(truck, topology.node(segment.from));
        require(target == topology.node(segment.to), "Departure target is not the active road end");
        var out = new ContinuousShortestPathData();
        main.network.getPathData(new Point(truck.getX(), truck.getY()), new Point(target.getX(), target.getY()), out);
        require(!out.isPlainMovement() && out.getNetwork() == topology.network()
                && Math.abs(main.toLengthUnits(out.distance, METER) - segment.length) <= POSITION_TOLERANCE_M,
                "Native movement would not follow the configured road");
        travelling = segment;
        departureTime = now();
        lastProgress = 0.0;
        sampledMiddle = false;
        truck.setSpeed(Math.min(truck.isLoaded ? 5.0 : 6.5, segment.speedLimit), MPS);
        receipt("DEPART");
    }

    public void sample() {
        if (travelling == null) return;
        PointNode a = topology.node(travelling.from);
        PointNode b = topology.node(travelling.to);
        double dx = b.getX() - a.getX();
        double dy = b.getY() - a.getY();
        double progress = ((probe.getX() - a.getX()) * dx + (probe.getY() - a.getY()) * dy) / (dx * dx + dy * dy);
        double offRoad = main.toLengthUnits(Math.hypot(probe.getX() - a.getX() - progress * dx,
                probe.getY() - a.getY() - progress * dy), METER);
        require(Double.isFinite(progress) && progress >= lastProgress - 1e-7 && progress <= 1 + 1e-7
                && offRoad <= POSITION_TOLERANCE_M, "Vehicle position deviates from current road");
        lastProgress = progress;
        if (!sampledMiddle && progress >= 0.5 && progress < 1.0) {
            receipt("MIDPOINT");
            sampledMiddle = true;
        }
    }

    public void arrived(Truck truck) {
        require(truck == probe && travelling != null && truck.currentSegment == travelling && now() > departureTime
                && sampledMiddle, "Missing native road departure or interior position before arrival");
        verifyPosition(truck, topology.node(travelling.to));
        receipt("ARRIVE");
        (truck.isLoaded ? inbound : outbound).add(travelling.key);
        travelling = null;
    }

    public void finish(Truck truck) {
        var source = main.getLoadSiteById(loadSlot);
        var dump = main.getDumpSiteById(0);
        require(truck == probe && travelling == null && main.completedTrips == 1
                && source.totalServed == 1 && dump.totalServed == 1 && source.incomingTrucks == 0
                && !truck.isLoaded && !truck.isMoving() && truck.currentSegment == null
                && !main.fleetProcessOwns(truck), "Trip or resource release incomplete");
        require(outbound.equals(edgeKeys(name(0), loadNode(loadSlot)))
                && inbound.equals(edgeKeys(loadNode(loadSlot), name(0))), "Native trajectory differs from frozen OD");
        double remaining = 0.0;
        double remainingMetal = 0.0;
        for (double[] batch : dump.crusherFifo) {
            remaining += batch[0];
            remainingMetal += batch[0] * batch[1];
        }
        closeMass(source.producedTons, truck.capacity);
        closeMass(source.producedTons, dump.receivedTons);
        closeMass(source.producedMetalTons, dump.receivedMetalTons);
        closeMass(dump.receivedTons, remaining + main.crusherOreTons);
        closeMass(dump.receivedMetalTons, remainingMetal + main.crusherMetalTons);
        closeMass(source.reservedTons, 0.0);
        closeMass(dump.crusherReservedTons, 0.0);
        main.auditMaterialBalances("TOPOLOGY_TRIP_DONE");
        log("PASS", Map.of("loaded_tons", source.producedTons, "received_tons", dump.receivedTons,
                "received_metal_tons", dump.receivedMetalTons, "crushed_tons", main.crusherOreTons,
                "crushed_metal_tons", main.crusherMetalTons, "fifo_tons", remaining,
                "fifo_metal_tons", remainingMetal, "completed_trips", main.completedTrips));
        passed = true;
        System.out.printf("[TOPOLOGY_TRIP_PASS] layout=%s run=%d ore=%.3f%n",
                topology.layoutId(), main.experimentRunIndex, dump.receivedTons);
        main.getEngine().finish();
    }

    private List<String> edgeKeys(String start, String target) {
        return main.getRouteSegments(start, target).stream().map(segment -> segment.key).toList();
    }

    private void receipt(String event) {
        log(event, Map.of("road", travelling.key, "native_road_index", roadIds.get(travelling.key),
                "from", wireIds.get(travelling.from), "to", wireIds.get(travelling.to), "loaded", probe.isLoaded,
                "x_m", main.toLengthUnits(probe.getX(), METER), "y_m", main.toLengthUnits(probe.getY(), METER),
                "payload_tons", probe.payloadTons));
    }

    private void verifyPosition(Truck truck, PointNode node) {
        require(truck.getNetwork() == topology.network() && truck.getLevel() == topology.level()
                && main.toLengthUnits(Math.hypot(truck.getX() - node.getX(), truck.getY() - node.getY()), METER)
                    <= POSITION_TOLERANCE_M, "Vehicle is outside the configured road node");
    }

    private double now() { return main.time(SECOND); }
    private String[] row(String kind, int index) { return topology.setting(kind + "." + index).split(","); }
    private String name(int id) { return nodes.get(id).canonicalName(); }

    private void log(String event, Map<String, Object> fields) {
        var attributes = new LinkedHashMap<String, Object>();
        attributes.put("event", event);
        attributes.put("time_s", now());
        attributes.put("run_index", main.experimentRunIndex);
        attributes.put("environment_seed", main.experimentSeed);
        attributes.put("recording_id", config.getProperty("recording_id"));
        attributes.put("config_sha256", configSha);
        attributes.put("layout_sha256", topology.sceneSha256());
        attributes.put("layout_id", topology.layoutId());
        attributes.putAll(fields);
        events.println(gson.toJson(attributes));
        require(!events.checkError(), "Could not write single-trip event receipt");
    }

    public void close() {
        if (events == null) return;
        try {
            if (!passed) log("INCOMPLETE", Map.of());
        } finally {
            events.close();
            events = null;
        }
        if (!passed) System.err.println("[TOPOLOGY_TRIP_INCOMPLETE] run=" + main.experimentRunIndex);
    }

    private static void closeMass(double actual, double expected) {
        require(Double.isFinite(actual) && Double.isFinite(expected) && Math.abs(actual - expected) <= MASS_TOLERANCE_T,
                "Material ledger mismatch: " + actual + " versus " + expected);
    }

    private static void require(boolean condition, String message) {
        if (!condition) throw new IllegalStateException(message);
    }
}
