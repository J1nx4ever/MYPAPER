package fusion;

import static com.anylogic.engine.LengthUnits.METER;
import static com.anylogic.engine.SpeedUnits.MPS;
import static com.anylogic.engine.TimeUnits.SECOND;
import static fusion.AcceptanceEventRecorder.fields;

import com.anylogic.engine.Point;
import com.anylogic.engine.markup.ContinuousShortestPathData;
import com.anylogic.engine.markup.PointNode;
import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;
import java.util.Random;

/** Runs the real transport and crusher state machines with a frozen local dispatcher. */
public final class TopologyWitness {
    private static final Path DIRECTORY = Path.of("@CONFIG_DIRECTORY@");
    private static final double POSITION_TOLERANCE_M = 0.001;
    private static final String[] CASES = {"A", "B"};
    private final Main main;
    private final JsonObject config;
    private final String configSha;
    private final String modelSha;
    private final TrainingTopology topology;
    private final Map<Integer, TrainingTopology.NodeSpec> nodes = new HashMap<>();
    private final Map<String, Integer> roadIds = new HashMap<>();
    private final PathSegment[] travelling = new PathSegment[9];
    private final double[] departure = new double[9];
    private final Map<String, Integer> approvedArrivals = new java.util.TreeMap<>();
    private TopologyFeasibilityDispatch dispatcher;
    private long decisionSequence;
    private double previousWait;
    private boolean sealed;

    public static int caseNumber(int index) {
        if (index < 0 || index >= CASES.length) throw new IllegalArgumentException("Unknown witness case");
        return index + 1;
    }

    public static TopologyWitness open(Main main, int number) {
        require(number > 0 && number <= CASES.length, "Unknown witness case");
        try {
            String key = CASES[number - 1];
            byte[] bytes = Files.readAllBytes(DIRECTORY.resolve(key + ".json"));
            String digest = hash(bytes);
            require(Files.readString(DIRECTORY.resolve(key + ".sha256")).trim().equals(digest),
                    "Witness configuration bytes have changed");
            var config = JsonParser.parseString(new String(bytes, java.nio.charset.StandardCharsets.UTF_8))
                    .getAsJsonObject();
            return new TopologyWitness(main, config, digest);
        } catch (IOException error) {
            throw new UncheckedIOException(error);
        }
    }

    private TopologyWitness(Main main, JsonObject config, String digest) throws IOException {
        this.main = main;
        this.config = config;
        configSha = digest;
        require("topology_full_shift_feasibility".equals(text("purpose")), "Wrong witness purpose");
        modelSha = hash(Files.readAllBytes(Path.of(text("model_path"))));
        require(modelSha.equals(text("model_sha256")), "The AnyLogic source file has changed");
        topology = TrainingTopology.load(main, text("layout_path"), text("layout_sha256"));
        require(topology.layoutId().equals(text("layout_id")), "Wrong physical map");
        for (var spec : topology.nodeSpecs()) nodes.put(spec.wireId(), spec);
    }

    public void start() {
        require(now() == 0 && !main.episodeStarted, "Witness must start from a fresh Main");
        var task = config.getAsJsonObject("task");
        main.experimentRunIndex = config.get("run_index").getAsLong();
        main.experimentSeed = config.get("environment_seed").getAsLong();
        main.episodeId = text("recording_id");
        main.getEngine().setDefaultRandomGenerator(new Random(main.experimentSeed));
        main.getEngine().setStopTime(task.get("shift_seconds").getAsInt() + 1.0, SECOND);
        main.numTrucks = 9;
        main.deterministicCheck = true;
        main.dynamic = true;
        main.useLocalDynamicSpeed = false;
        main.useEndogenousCongestion = false;
        main.enableOnnxDispatch = false;
        main.enableBoundaryEventRecording = false;
        main.enableMaterialEventMemory = false;
        main.enableP0SnapshotAudit = false;
        main.enableWasteTaskLimit = true;
        main.targetOreTonsShift = task.get("ore_target_tons").getAsDouble();
        main.targetWasteTonsShift = task.get("waste_target_tons").getAsDouble();
        main.wasteDispatchStopTonsShift = task.get("waste_dispatch_stop_tons").getAsDouble();
        main.cpGradeEpsilon = task.get("grade_epsilon").getAsDouble();
        main.cpTonnageEpsilon = task.get("tonnage_epsilon").getAsDouble();
        main.cpMassBalanceEpsilon = task.get("mass_balance_epsilon").getAsDouble();
        main.isWarmUp = false;
        main.enableMaterialAudit = true;
        main.materialAuditReady = false;
        main.currentWindowIndex = 0;
        main.waitBreakdownLedger = new WaitBreakdownLedger(main.rlShiftSeconds);
        require(main.warmUpTime == 0 && main.shiftDuration == 480.0
                && main.rlShiftSeconds == task.get("shift_seconds").getAsDouble()
                && main.rlWindowSeconds == task.get("window_seconds").getAsDouble()
                && main.gradeLowerBound == task.get("grade_lower").getAsDouble()
                && main.gradeUpperBound == task.get("grade_upper").getAsDouble()
                && !main.acceptanceLongLoadEnabled && main.acceptanceBoundaryMode == 0,
                "Native shift, grade, or timing setup differs from the task");

        int disabled = 0;
        for (int i = 0; i < 9; i++) if ("0".equals(row("vehicle", i)[0])) disabled |= 1 << i;
        main.fleetDisabledTruckMask = disabled;
        main.fleetFaultTruckId = -1;
        main.fleetFaultTriggerSeconds = -1;
        main.fleetResilience = new FleetResilienceState(disabled, -1, -1.0);
        installRoadsAndRoutes();
        instantiateSites();
        instantiateFleet();
        dispatcher = new TopologyFeasibilityDispatch(main);
        main.materialAuditReady = true;
        main.auditMaterialBalances("TOPOLOGY_WITNESS_INITIAL");

        var crusher = main.getDumpSiteById(0);
        var actual = actualScene();
        require(new Gson().toJsonTree(actual).equals(config.get("expected_runtime_scene")),
                "Native startup scene differs from the frozen scene: " + new Gson().toJson(actual));
        var physical = fields("shift_seconds", main.rlShiftSeconds, "window_seconds", main.rlWindowSeconds,
                "grade_lower", main.gradeLowerBound, "grade_upper", main.gradeUpperBound,
                "grade_epsilon", main.cpGradeEpsilon, "tonnage_epsilon", main.cpTonnageEpsilon,
                "mass_balance_epsilon", main.cpMassBalanceEpsilon, "crusher_rate_tph", crusher.crusherRateTph,
                "actual_runtime_scene", actual);
        main.boundaryEventRecorder = new AcceptanceEventRecorder(text("event_directory"), main.episodeId,
                configSha, main.episodeId, "topology-full-shift-v1", main.experimentSeed,
                main.experimentRunIndex, physical);
        for (Truck truck : main.trucks) {
            if (!main.isFleetTruckDisabled(truck.truckId)) truck.deliver("ACTIVATE", truck);
        }
        System.out.printf("[TOPOLOGY_WITNESS_START] %s run=%d%n", topology.layoutId(), main.experimentRunIndex);
    }

    private void instantiateFleet() {
        Truck[] predecessor = new Truck[2];
        for (int i = 0; i < 9; i++) {
            String[] data = row("vehicle", i);
            Truck truck = main.add_trucks();
            int start = Integer.parseInt(data[3]);
            truck.initTruck(i, Double.parseDouble(data[2]));
            truck.truckType = Integer.parseInt(data[1]);
            boolean active = !main.isFleetTruckDisabled(i);
            truck.initialDispatchDone = !active;
            if (active) {
                attach(truck, topology.node(start));
                truck.currentNode = name(start);
                truck.startPoint = truck.currentNode;
                truck.currentBusinessSiteId = truck.currentNode;
                truck.initialPredecessor = predecessor[start];
                predecessor[start] = truck;
            } else {
                truck.initialPredecessor = null;
                truck.setLocation(main.parking);
                truck.currentNode = "";
            }
            truck.setSpeed(active ? 6.5 : 0.0, MPS);
            main.waitBreakdownLedger.registerTruck(i);
        }
    }

    @VERIFIED_BINDER@

    private Map<String, Object> actualScene() {
        List<Object> positions = new ArrayList<>(), roads = new ArrayList<>(), fleet = new ArrayList<>();
        List<Object> loaders = new ArrayList<>(), dumps = new ArrayList<>();
        List<Integer> loaderSlots = new ArrayList<>(), dumpSlots = new ArrayList<>();
        for (var spec : topology.nodeSpecs()) {
            var node = topology.node(spec.wireId());
            positions.add(List.of(spec.wireId(), spec.canonicalName(), millimetres(node.getX()),
                    millimetres(node.getY())));
        }
        for (var road : topology.roads()) {
            int from = road.fromNodeId(), to = road.toNodeId();
            long length = Math.round(1000 * road.nativePath().length(METER));
            for (var edge : List.of(main.segmentMap.get(name(from) + "_" + name(to)),
                    main.segmentMap.get(name(to) + "_" + name(from)))) {
                require(!edge.closed && Math.round(1000 * edge.length) == length, "Native road unavailable");
                roads.add(List.of(edge.from.equals(name(from)) ? from : to,
                        edge.to.equals(name(to)) ? to : from, length, Math.round(1000 * edge.speedLimit)));
            }
        }
        for (Truck truck : main.trucks) {
            boolean active = !main.isFleetTruckDisabled(truck.truckId);
            int start = active ? wireId(truck.currentNode) : -1;
            if (active) verifyPosition(truck, topology.node(start));
            require(!truck.isLoaded && truck.payloadTons == 0 && truck.plannedPayloadTons == 0,
                    "Nonempty opening truck");
            fleet.add(List.of(truck.truckId, active, truck.truckType, Math.round(10 * truck.capacity), start));
        }
        for (int i = 0; i < 4; i++) {
            var site = main.getLoadSiteById(i);
            int point = wireId(site.loadNode);
            require(site.getNetwork() == topology.network() && site.getLevel() == topology.level()
                    && millimetres(site.getX()) == millimetres(site.loadNode.getX())
                    && millimetres(site.getY()) == millimetres(site.loadNode.getY())
                    && site.reservedTons == 0 && site.producedTons == 0,
                    "Invalid physical loader startup");
            for (double grade : site.blockGrades) require(grade == site.grade, "Unexpected grade block");
            loaderSlots.add(site.shovelCapacity);
            loaders.add(List.of(site.siteId, point, site.isOre, Math.round(1_000_000 * site.grade),
                    Math.round(1_000_000 * site.gradeHalfRange), Math.round(10 * site.remainingTons),
                    Math.round(1000 * site.singleServeTime), Math.round(10 * site.bucket),
                    Math.round(10 * site.density), Math.round(10 * site.gradeBlockTons)));
        }
        for (int i = 0; i < 2; i++) {
            var site = main.getDumpSiteById(i);
            require(site.getNetwork() == topology.network() && site.getLevel() == topology.level()
                    && millimetres(site.getX()) == millimetres(site.dumpNode.getX())
                    && millimetres(site.getY()) == millimetres(site.dumpNode.getY())
                    && site.receivedTons == 0 && site.receivedMetalTons == 0
                    && site.crusherStoredTons == 0 && site.crusherReservedTons == 0 && site.crusherFifo.isEmpty(),
                    "Invalid physical dump startup");
            dumpSlots.add(site.serviceCapacity);
            dumps.add(List.of(site.siteId, wireId(site.dumpNode), site.acceptsOre, site.acceptsWaste,
                    Math.round(100 * site.serviceTime)));
        }
        return fields("layout_id", topology.layoutId(), "node_positions", positions, "directed_roads", roads,
                "opening_fleet", fleet, "load_sites", loaders, "dump_sites", dumps,
                "loader_slots", loaderSlots, "dump_slots", dumpSlots,
                "crusher_capacity_tons", Math.round(main.getDumpSiteById(0).crusherBufferCapacityTons),
                "initial_inventory", fields("crusher_fifo_tons", 0, "carried_tons", 0, "reserved_tons", 0),
                "exogenous_config", fields("deterministic", main.deterministicCheck,
                        "local_dynamic_speed", main.useLocalDynamicSpeed,
                        "endogenous_congestion", main.useEndogenousCongestion,
                        "fault_truck_id", main.fleetResilience.faultId,
                        "fault_trigger_seconds", Math.round(main.fleetResilience.triggerSeconds),
                        "empty_speed_mm_s", 6500, "loaded_speed_mm_s", 5000,
                        "ore_target_tons", Math.round(main.targetOreTonsShift),
                        "waste_target_tons", Math.round(main.targetWasteTonsShift),
                        "waste_dispatch_stop_tons", Math.round(main.wasteDispatchStopTonsShift),
                        "waste_upper_tons", Math.round(main.getWasteTaskUpperTons()),
                        "enable_waste_task_limit", main.enableWasteTaskLimit),
                "model_sha256", modelSha);
    }

    public int dispatch(Truck truck) {
        require(!main.isFleetTruckDisabled(truck.truckId) && travelling[truck.truckId] == null,
                "Ineligible vehicle attempted dispatch");
        int action = dispatcher.choose(truck);
        if (action >= 0) {
            require(action < 4 && main.getLoadSiteById(action).getDispatchableTons(truck) > 0,
                    "Frozen dispatcher proposed an invalid load site");
        }
        return action;
    }

    public void applied(Truck truck) {
        dispatcher.committed(truck);
        double wait = main.getCumulativeWaitVehicleSeconds();
        request(wait, false);
        main.boundaryEventRecorder.actionApplied(now(), main.episodeId + "-" + decisionSequence,
                decisionSequence, truck.targetLoadSiteId);
        decisionSequence++;
    }

    private void request(double wait, boolean terminal) {
        require(wait + 1e-8 >= previousWait, "Wait regression");
        main.boundaryEventRecorder.request(now(), main.episodeId + "-" + decisionSequence, decisionSequence,
                wait, Math.max(0, wait - previousWait), terminal);
        previousWait = wait;
    }

    public String loadNode(int slot) { return name(Integer.parseInt(row("load", slot)[0])); }
    public String dumpNode(int slot) { return name(Integer.parseInt(row("dump", slot)[0])); }
    public int modelNodeNumber(PointNode node) { return main.roadNodeNameToInt(name(wireId(node))); }

    public void departing(Truck truck, PointNode target) {
        int id = truck.truckId;
        PathSegment segment = truck.currentSegment;
        require(travelling[id] == null && segment != null && !segment.closed, "Invalid native departure");
        PointNode from = topology.node(segment.from), to = topology.node(segment.to);
        verifyPosition(truck, from);
        require(target == to, "Native departure target differs from road end");
        var path = new ContinuousShortestPathData();
        main.network.getPathData(new Point(from.getX(), from.getY()), new Point(to.getX(), to.getY()), path);
        require(!path.isPlainMovement() && path.getNetwork() == topology.network()
                && Math.abs(main.toLengthUnits(path.distance, METER) - segment.length) <= POSITION_TOLERANCE_M,
                "Movement would bypass the configured road");
        travelling[id] = segment;
        departure[id] = now();
        double speed = Math.min(truck.isLoaded ? 5.0 : 6.5, segment.speedLimit);
        truck.setSpeed(speed, MPS);
    }

    public void arrived(Truck truck) {
        int id = truck.truckId;
        var road = travelling[id];
        require(road != null && road == truck.currentSegment, "Native arrival without matching departure");
        verifyPosition(truck, topology.node(road.to));
        double speed = Math.min(truck.isLoaded ? 5.0 : 6.5, road.speedLimit);
        require(Math.abs(now() - departure[id] - road.length / speed) <= 0.001,
                "Native traversal time differs from segment length and speed");
        approvedArrivals.merge(road.key, 1, Integer::sum);
        travelling[id] = null;
    }

    public void finish(double cumulativeWait) {
        require(now() == main.rlShiftSeconds && decisionSequence > 0 && main.completedTrips > 0
                && !approvedArrivals.isEmpty(), "Full-shift witness did not run");
        var crusher = main.getDumpSiteById(0);
        request(cumulativeWait, true);
        main.boundaryEventRecorder.finishWitness(now(), cumulativeWait, main.crusherOreTons,
                main.crusherMetalTons, main.totalWasteMoved, crusher.crusherStoredTons,
                crusher.auditBufferMetalTons(), approvedArrivals);
        sealed = true;
        main.episodeClosed = true;
        System.out.printf("[TOPOLOGY_WITNESS_COMPLETE] %s run=%d ore=%.3f waste=%.3f%n",
                topology.layoutId(), main.experimentRunIndex, main.crusherOreTons, main.totalWasteMoved);
        main.getEngine().finish();
    }

    public void close() {
        if (!sealed && main.boundaryEventRecorder != null) main.boundaryEventRecorder.closeIncomplete();
    }

    private int wireId(PointNode node) {
        for (var spec : topology.nodeSpecs()) if (topology.node(spec.wireId()) == node) return spec.wireId();
        throw new IllegalArgumentException("Unknown PointNode");
    }

    private int wireId(String node) { return wireId(topology.node(node)); }
    private long millimetres(double modelLength) { return Math.round(1000 * main.toLengthUnits(modelLength, METER)); }
    private void verifyPosition(Truck truck, PointNode node) {
        require(truck.getNetwork() == topology.network() && truck.getLevel() == topology.level()
                && main.toLengthUnits(Math.hypot(truck.getX() - node.getX(), truck.getY() - node.getY()), METER)
                <= POSITION_TOLERANCE_M, "Truck is outside the configured road endpoint");
    }
    private double now() { return main.time(SECOND); }
    private String text(String key) { return config.get(key).getAsString(); }
    private String[] row(String kind, int index) { return topology.setting(kind + "." + index).split(","); }
    private String name(int id) { return nodes.get(id).canonicalName(); }
    private static String hash(byte[] bytes) {
        try {
            return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(bytes));
        } catch (NoSuchAlgorithmException error) {
            throw new IllegalStateException(error);
        }
    }
    private static void require(boolean condition, String message) {
        if (!condition) throw new IllegalStateException(message);
    }
}
