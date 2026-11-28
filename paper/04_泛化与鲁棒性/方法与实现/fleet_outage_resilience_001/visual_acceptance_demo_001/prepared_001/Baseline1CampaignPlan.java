/**
 * Paste the rendered class into the model's Additional Class Baseline1CampaignPlan.
 * The four @@...@@ values are rendered from the frozen one-click plan.
 * Freeform accessors do not poll, claim, connect, or change model state.
 */
public final class Baseline1CampaignPlan {
    private static final String MASTER_PLAN_PATH = "/home/jinx/Faith/experiments/lagrangian_ppo_v1/change_plans/token_attention_001/business_graph_summary_009/fleet_outage_resilience_001/visual_acceptance_demo_001/prepared_001/master_plan.csv";
    private static final String MASTER_PLAN_SHA256 = "eaf7d81502c16638c10b77045aa3c2cd25592b1f01752910b00e787210e06419";
    private static final String CONTROL_DIRECTORY = "/home/jinx/Faith/experiments/lagrangian_ppo_v1/runs/bg128-fleet-outage-visual-demo-20261124-001/control";
    private static final String CAMPAIGN_ID = "bg128-fleet-outage-visual-demo-20261124-001";
    private static final String REVISION = "boundary-events-v1-native-rpc-runtime-scene-v1";
    private static final String[] COLUMNS = {
        "ordinal", "actor_seed", "group", "role", "source_recording_id", "recording_id",
        "run_index", "experiment_seed", "actor_sampling_seed", "event_directory", "event_bundle",
        "python_output", "target_training_update", "checkpoint_update", "initial_mask", "fault_truck_id", "fault_trigger_seconds", "ore_target", "waste_target", "waste_stop", "config_sha256"
    };
    private static final int JOB_COUNT = 1;
    private static final long TIMEOUT_MILLIS = 3600000L;
    private static volatile Plan cached;
    private static final java.util.Set<Long> started = new java.util.HashSet<Long>();

    private Baseline1CampaignPlan() { }

    private static boolean controllerStarted = false;
    /** Only Main startup invokes this; all Freeform accessors remain read-only. */
    private static synchronized void ensureController(Row row) {
        if (controllerStarted) return;
        require(row.ordinal == 0, "one-click startup requires index 0; recovery must be reviewed");
        checkStop(absolute(CONTROL_DIRECTORY, "control directory"));
        try {
            java.nio.file.Path bridge = absolute("/home/jinx/Faith/experiments/lagrangian_ppo_v1/change_plans/token_attention_001/business_graph_summary_009/fleet_outage_resilience_001/visual_acceptance_demo_001/demo_launch.py", "bridge path");
            require(sha256(java.nio.file.Files.readAllBytes(bridge)).equals("3d03e1658c582f04b173d8371df873c284e3daf3b31280eb5df3ef527cab4a7c"),
                "one-click bridge byte SHA256 differs");
            Process process = new ProcessBuilder(
                "/home/jinx/miniconda3/envs/anylogic-env/bin/python", "-B", bridge.toString(),
                "--suite", "/home/jinx/Faith/experiments/lagrangian_ppo_v1/change_plans/token_attention_001/business_graph_summary_009/fleet_outage_resilience_001/visual_acceptance_demo_001/prepared_001/suite.json", "--run-index", Long.toString(row.run),
                "--environment-seed", Long.toString(row.seed), "--recording-id", row.cells[5],
                "--from-anylogic-run")
                .directory(new java.io.File("/home/jinx/Faith/experiments/lagrangian_ppo_v1")).inheritIO().start();
            require(process.waitFor(15L, java.util.concurrent.TimeUnit.MINUTES),
                "controller startup timed out; inspect launch record before any retry");
            require(process.exitValue() == 0,
                "controller startup failed; inspect AnyLogic console; do not resubmit");
            controllerStarted = true;
        } catch (java.io.IOException error) {
            throw new IllegalStateException("Baseline 1 one-click STOP: cannot start fixed Python bridge", error);
        } catch (InterruptedException error) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("Baseline 1 one-click STOP: controller startup interrupted", error);
        }
    }



    private static final int SCENE_LOAD_SITES = 4;
    private static final String SCENARIO_TABLE_PATH = "/home/jinx/Faith/experiments/lagrangian_ppo_v1/change_plans/token_attention_001/business_graph_summary_009/fleet_outage_resilience_001/visual_acceptance_demo_001/prepared_001/scenario_table.csv";
    private static final String SCENARIO_TABLE_SHA256 = "e683304060e0192438225abcc4abeb3c6e6ab6720eb1f449f733998e8a0f05af";
    private static final String SCENARIO_HEADER = "run_index,scenario_id,load0_is_ore,load1_is_ore,load2_is_ore,load3_is_ore,load0_grade,load1_grade,load2_grade,load3_grade,load0_grade_half_range,load1_grade_half_range,load2_grade_half_range,load3_grade_half_range,road_6_speed_limit_mps";

    private static final class Scenario {
        final String id;
        final boolean[] ore = new boolean[SCENE_LOAD_SITES];
        final double[] grades = new double[SCENE_LOAD_SITES];
        final double[] halfRanges = new double[SCENE_LOAD_SITES];
        final double road6Limit;

        Scenario(String[] values) {
            id = values[1];
            require(id.matches("[A-Za-z0-9][A-Za-z0-9._-]*"), "invalid scenario ID");
            int oreSites = 0;
            for (int site = 0; site < SCENE_LOAD_SITES; ++site) {
                String role = values[2 + site];
                require(role.equals("0") || role.equals("1"), "invalid source role");
                ore[site] = role.equals("1");
                if (ore[site]) ++oreSites;
                grades[site] = scenarioNumber(values[2 + SCENE_LOAD_SITES + site]);
                halfRanges[site] = scenarioNumber(values[2 + 2 * SCENE_LOAD_SITES + site]);
                require(halfRanges[site] >= 0.0 && halfRanges[site] <= 1.0,
                    "invalid grade half-range");
                if (ore[site]) {
                    require(grades[site] >= halfRanges[site] && grades[site] <= 1.0 - halfRanges[site],
                        "ore grade support outside [0,1]");
                } else {
                    require(grades[site] == 0.0, "waste source grade must be zero");
                }
            }
            require(oreSites == 3, "exactly three ore sources required");
            road6Limit = scenarioNumber(values[2 + 3 * SCENE_LOAD_SITES]);
            require(road6Limit > 0.0, "invalid scenario road speed");
        }
    }

    private static double scenarioNumber(String text) {
        require(text.matches("[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?"), "invalid scenario numeric syntax");
        double value = Double.parseDouble(text);
        require(Double.isFinite(value), "nonfinite scenario number");
        return value;
    }

    private static java.util.Map<Long, Scenario> readScenarios(Row[] rows) throws java.io.IOException {
        digestValue(SCENARIO_TABLE_SHA256, "scenario table SHA256");
        byte[] bytes = java.nio.file.Files.readAllBytes(absolute(SCENARIO_TABLE_PATH, "scenario table path"));
        require(sha256(bytes).equals(SCENARIO_TABLE_SHA256), "scenario table byte SHA256 differs");
        String csv = java.nio.charset.StandardCharsets.US_ASCII.newDecoder()
            .decode(java.nio.ByteBuffer.wrap(bytes)).toString();
        String[] lines = csv.split("\\n", -1);
        require(lines.length == JOB_COUNT + 2 && lines[JOB_COUNT + 1].isEmpty()
            && lines[0].equals(SCENARIO_HEADER), "scenario table header or row count differs");
        java.util.Map<Long, Scenario> scenes = new java.util.HashMap<>();
        for (int i = 0; i < JOB_COUNT; ++i) {
            String[] values = lines[i + 1].split(",", -1);
            require(values.length == 3 + 3 * SCENE_LOAD_SITES, "invalid scenario column count at ordinal " + i);
            long run = positive(values[0], "scenario run_index");
            require(run == rows[i].run, "scenario run_index differs from master at ordinal " + i);
            require(scenes.put(run, new Scenario(values)) == null, "duplicate scenario run_index");
        }
        return java.util.Collections.unmodifiableMap(scenes);
    }

    private static Scenario scenarioForRun(long runIndex) {
        Scenario scene = plan().scenarios.get(runIndex);
        require(scene != null, "run_index absent from frozen scenario table: " + runIndex);
        return scene;
    }

    private static void requireSceneSite(int siteId) {
        require(siteId >= 0 && siteId < SCENE_LOAD_SITES, "scenario load site must be in [0,3]");
    }

    public static String scenarioId(long runIndex) { return scenarioForRun(runIndex).id; }

    public static boolean loadIsOre(long runIndex, int siteId) {
        requireSceneSite(siteId);
        return scenarioForRun(runIndex).ore[siteId];
    }

    public static double loadGrade(long runIndex, int siteId) {
        requireSceneSite(siteId);
        return scenarioForRun(runIndex).grades[siteId];
    }

    public static double loadGradeHalfRange(long runIndex, int siteId) {
        requireSceneSite(siteId);
        return scenarioForRun(runIndex).halfRanges[siteId];
    }

    /** The scene and this method both use m/s. */
    public static double roadSpeedLimit(long runIndex, String from, String to, double nominal) {
        Scenario scene = scenarioForRun(runIndex);
        require(Double.isFinite(nominal) && nominal > 0.0, "invalid nominal speed limit");
        return "intersection1".equals(from) && "Crusher".equals(to) ? scene.road6Limit : nominal;
    }

    private static final class Row {
        final String[] cells;
        final int ordinal;
        final long run;
        final long seed;
        Row(String[] values) {
            cells = values;
            ordinal = integer(values[0], "ordinal");
            require(values[2].equals("BASELINE") && (values[3].equals("train") || values[3].equals("final")),
                "smoke rows all use the final RPC policy");
            positive(values[1], "actor_seed");
            require(values[2].equals("P0") || values[2].equals("P1") || values[2].equals("P2")
                || values[2].equals("P4") || values[2].equals("BASELINE"), "invalid group");
            require(values[3].equals("calibration") || values[3].equals("t0") || values[3].equals("train")
                || values[3].equals("monitor") || values[3].equals("final"), "invalid role");
            require(values[2].startsWith("P") == values[3].equals("calibration"),
                "native/calibration and BASELINE role combination differs");
            run = positive(values[6], "run_index");
            seed = positive(values[7], "experiment_seed");
            positive(values[8], "actor_sampling_seed");
            java.nio.file.Path events = absolute(values[9], "event_directory");
            java.nio.file.Path bundle = absolute(values[10], "event_bundle");
            require(events.equals(bundle.getParent()), "event_bundle must be a direct child of event_directory");
            absolute(values[11], "python_output");
            require(integer(values[12], "target_training_update") >= 0, "negative target update");
            require(integer(values[13], "checkpoint_update") >= 0, "negative checkpoint update");
        }
    }

    private static final class Plan {
        final Row[] rows;
        final java.util.Map<Long, Row> byRun;
        final java.util.Map<Long, Scenario> scenarios;
        Plan(Row[] rows, java.util.Map<Long, Row> byRun) throws java.io.IOException {
            this.rows = rows;
            this.byRun = byRun;
            this.scenarios = readScenarios(rows);
        }
    }

    private static final class UniqueProperties extends java.util.Properties {
        @Override public synchronized Object put(Object key, Object value) {
            require(!containsKey(key), "duplicate ready property: " + key);
            return super.put(key, value);
        }
    }

    private static void require(boolean condition, String message) {
        if (!condition) throw new IllegalStateException("Baseline 1 one-click STOP: " + message);
    }

    private static int integer(String value, String name) {
        try {
            require(value.matches("0|[1-9][0-9]*"), "invalid integer " + name);
            return Integer.parseInt(value);
        } catch (NumberFormatException error) {
            throw new IllegalStateException("Baseline 1 one-click STOP: integer overflow " + name, error);
        }
    }

    private static long positive(String value, String name) {
        try {
            require(value.matches("[1-9][0-9]*"), "invalid positive integer " + name);
            long result = Long.parseLong(value);
            require(result > 0, "nonpositive " + name);
            return result;
        } catch (NumberFormatException error) {
            throw new IllegalStateException("Baseline 1 one-click STOP: integer overflow " + name, error);
        }
    }

    private static java.nio.file.Path absolute(String value, String name) {
        java.nio.file.Path path = java.nio.file.Paths.get(value);
        require(path.isAbsolute() && path.normalize().toString().equals(value), "absolute normalized " + name + " required");
        return path;
    }

    private static String sha256(byte[] bytes) {
        try {
            byte[] result = java.security.MessageDigest.getInstance("SHA-256").digest(bytes);
            StringBuilder text = new StringBuilder(64);
            for (byte value : result) text.append(String.format(java.util.Locale.ROOT, "%02x", value & 255));
            return text.toString();
        } catch (java.security.NoSuchAlgorithmException error) {
            throw new IllegalStateException("SHA-256 unavailable", error);
        }
    }

    private static void digestValue(String value, String name) {
        require(value != null && value.matches("[0-9a-f]{64}") && !value.matches("0{64}"), "invalid " + name);
    }

    private static Plan plan() {
        Plan result = cached;
        if (result != null) return result;
        synchronized (Baseline1CampaignPlan.class) {
            if (cached != null) return cached;
            digestValue(MASTER_PLAN_SHA256, "master plan SHA256");
            require(!CAMPAIGN_ID.isBlank() && !CAMPAIGN_ID.contains("@@"), "unrendered campaign ID");
            absolute(CONTROL_DIRECTORY, "control directory");
            try {
                byte[] bytes = java.nio.file.Files.readAllBytes(absolute(MASTER_PLAN_PATH, "master plan path"));
                require(sha256(bytes).equals(MASTER_PLAN_SHA256), "master plan byte SHA256 differs");
                String csv = java.nio.charset.StandardCharsets.UTF_8.newDecoder()
                    .decode(java.nio.ByteBuffer.wrap(bytes)).toString();
                require(!csv.contains("\""), "CSV quotes are not permitted");
                String[] lines = csv.split("\\r?\\n", -1);
                int length = lines.length;
                if (length > 0 && lines[length - 1].isEmpty()) length--;
                require(length == JOB_COUNT + 1, "master must contain exactly " + JOB_COUNT + " jobs");
                require(lines[0].equals(String.join(",", COLUMNS)), "master header differs");
                Row[] rows = new Row[JOB_COUNT];
                java.util.Map<Long, Row> byRun = new java.util.HashMap<Long, Row>();

                java.util.Set<Long> environmentSeeds = new java.util.HashSet<>();
                java.util.Set<Long> samplingSeeds = new java.util.HashSet<>();
                for (int i = 0; i < JOB_COUNT; i++) {
                    String[] values = lines[i + 1].split(",", -1);
                    require(values.length == COLUMNS.length, "master column count differs at ordinal " + i);
                    for (String value : values) require(!value.isEmpty() && value.equals(value.trim())
                        && !value.contains("\r") && !value.contains("\n"), "empty or malformed master field");
                    Row row = new Row(values);
                    require(row.ordinal == i, "master ordinal is not contiguous at " + i);
                    require(byRun.put(row.run, row) == null, "duplicate master run_index");

                    environmentSeeds.add(row.seed);
                    samplingSeeds.add(positive(values[8], "actor sampling seed"));
                    rows[i] = row;
                }

                require(java.util.Collections.disjoint(environmentSeeds, samplingSeeds), "environment and sampling seeds overlap");
                cached = new Plan(rows, byRun);
                return cached;
            } catch (java.io.IOException error) {
                throw new IllegalStateException("Baseline 1 one-click STOP: cannot read frozen master", error);
            }
        }
    }

    private static Row row(int index) {
        require(index >= 0 && index < JOB_COUNT, "Freeform index outside frozen plan");
        return plan().rows[index];
    }

    public static long runIndex(int index) { return row(index).run; }
    public static long environmentSeed(int index) { return row(index).seed; }
    public static String recordingId(int index) { return row(index).cells[5]; }
    public static String eventRoot(int index) { return row(index).cells[9]; }

    public static int dispatchPolicy(int index) { row(index); return 3; }
    public static int fleetDisabledTruckMask(int index) { return 511 ^ Integer.parseInt(row(index).cells[14]); }
    public static int faultTruckId(int index) { return Integer.parseInt(row(index).cells[15]); }
    public static double faultTriggerSeconds(int index) { return Double.parseDouble(row(index).cells[16]); }
    public static double oreTarget(int index) { return Double.parseDouble(row(index).cells[17]); }
    public static double wasteTarget(int index) { return Double.parseDouble(row(index).cells[18]); }
    public static double wasteStop(int index) { return Double.parseDouble(row(index).cells[19]); }
    public static boolean materialEnabled(int index) { row(index); return false; }
    public static boolean eventRecordingEnabled(int index) { row(index); return false; }
    public static boolean offlineEnabled(int index) { row(index); return false; }
    public static String offlineSha256(int index) { row(index); return "0000000000000000000000000000000000000000000000000000000000000000"; }
    public static String materialRoot(int index) { return eventRoot(index); }
    public static String phaseId(int index) { return recordingId(index).substring(CAMPAIGN_ID.length() + 1); }
    public static String stageId(int index) { return phaseId(index); }
    public static long actorSamplingSeed(int index) { return Long.parseLong(row(index).cells[8]); }
    public static double waitWeight(int index) { row(index); return 0.2; }
    public static double freeSpeedFactorStd(int index) { row(index); return 0.05; }
    public static boolean preparationOnly() { return false; }
    public static int rpcPort() { return 50051; }
    public static int jobCount() { return JOB_COUNT; }
    public static int frozenPlanJobCount() { return JOB_COUNT; }
    public static int resumeOffset() { return 0; }

    /** Called once at the very beginning of Main On startup, never in Freeform. */
    public static String awaitConfigSha256(long runIndex, long environmentSeed, String recordingId,
                                           String eventRoot, String instrumentationRevision) {
        java.nio.file.Path control = absolute(CONTROL_DIRECTORY, "control directory");
        checkStop(control);
        Row row = plan().byRun.get(runIndex);
        require(row != null, "run_index absent from frozen master");
        require(row.seed == environmentSeed && row.cells[5].equals(recordingId)
            && row.cells[9].equals(eventRoot), "Main parameters differ from frozen master");
        require(REVISION.equals(instrumentationRevision), "instrumentation revision differs");
        synchronized (started) { require(!started.contains(runIndex), "job already started in this classloader"); }
        ensureController(row);
        java.nio.file.Path ready = control.resolve("ready").resolve("run_" + runIndex + ".properties");
        long startedAt = System.nanoTime();
        while (true) {
            checkStop(control);
            if (java.nio.file.Files.exists(ready)) {
                String digest = validateReady(ready, row);
                checkStop(control);
                claim(control, row, digest);
                return digest;
            }
            require((System.nanoTime() - startedAt) / 1000000L < TIMEOUT_MILLIS, "ready wait timed out for run " + runIndex);
            try { Thread.sleep(50L); }
            catch (InterruptedException error) {
                Thread.currentThread().interrupt();
                throw new IllegalStateException("Baseline 1 one-click STOP: ready wait interrupted", error);
            }
        }
    }

    private static void checkStop(java.nio.file.Path control) {
        require(!java.nio.file.Files.exists(control.resolve("STOP.properties")), "STOP.properties exists");
    }

    private static String validateReady(java.nio.file.Path path, Row row) {
        try {
            UniqueProperties properties = new UniqueProperties();
            try (java.io.InputStream input = java.nio.file.Files.newInputStream(path)) { properties.load(input); }
            same(properties, "protocol", "baseline1-one-click-ready-v1");
            same(properties, "campaign_id", CAMPAIGN_ID);
            same(properties, "master_plan_sha256", MASTER_PLAN_SHA256);
            for (int i = 0; i < COLUMNS.length; i++) same(properties, COLUMNS[i], row.cells[i]);
            same(properties, "instrumentation_revision", REVISION);
            String config = properties.getProperty("config_sha256");
            digestValue(config, "ready config SHA256");
            same(properties, "snapshot_sha256", config);
            String snapshot = properties.getProperty("snapshot_path");
            require(snapshot != null && !snapshot.isBlank(), "ready snapshot path missing");
            byte[] bytes = java.nio.file.Files.readAllBytes(absolute(snapshot, "ready snapshot path"));
            require(sha256(bytes).equals(config), "snapshot actual byte SHA256 differs");
            return config;
        } catch (java.io.IOException error) {
            throw new IllegalStateException("Baseline 1 one-click STOP: ready/snapshot cannot be read", error);
        }
    }

    private static void same(java.util.Properties properties, String key, String expected) {
        require(expected.equals(properties.getProperty(key)), "ready property differs: " + key);
    }

    private static void claim(java.nio.file.Path control, Row row, String digest) {
        synchronized (started) {
            require(!started.contains(row.run), "job already started in this classloader");
            try {
                java.nio.file.Path directory = control.resolve("java_claims");
                java.nio.file.Files.createDirectories(directory);
                java.nio.file.Path path = directory.resolve("run_" + row.run + ".once.claim");
                String receipt = "protocol=baseline1-one-click-java-start-v1\ncampaign_id=" + CAMPAIGN_ID
                    + "\nmaster_plan_sha256=" + MASTER_PLAN_SHA256 + "\nordinal=" + row.ordinal
                    + "\nrun_index=" + row.run + "\nsnapshot_sha256=" + digest
                    + "\nconfig_sha256=" + digest + "\n";
                try (java.nio.channels.FileChannel channel = java.nio.channels.FileChannel.open(path,
                        java.nio.file.StandardOpenOption.CREATE_NEW, java.nio.file.StandardOpenOption.WRITE)) {
                    java.nio.ByteBuffer bytes = java.nio.charset.StandardCharsets.US_ASCII.encode(receipt);
                    while (bytes.hasRemaining()) channel.write(bytes);
                    channel.force(true);
                }
                started.add(row.run);
            } catch (java.io.IOException error) {
                throw new IllegalStateException("Baseline 1 one-click STOP: exclusive Java start claim failed; do not resubmit", error);
            }
        }
    }
}
