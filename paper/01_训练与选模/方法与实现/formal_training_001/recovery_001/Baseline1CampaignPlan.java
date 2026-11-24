/**
 * Paste the rendered class into the model's Additional Class Baseline1CampaignPlan.
 * The four @@...@@ values are rendered from the frozen one-click plan.
 * Freeform accessors do not poll, claim, connect, or change model state.
 */
public final class Baseline1CampaignPlan {
    private static final String MASTER_PLAN_PATH = "/home/jinx/Faith/experiments/lagrangian_ppo_v1/change_plans/token_attention_001/business_graph_summary_009/formal_training_001/prepared_001/master_plan.csv";
    private static final String MASTER_PLAN_SHA256 = "0b1dd27e450204f6661b0554e1ffa06d8ab87453e85612b4d0a59b6a1dc1044b";
    private static final String CONTROL_DIRECTORY = "/home/jinx/Faith/experiments/lagrangian_ppo_v1/runs/business-graph-summary-formal-seed1-20261118-001/recovery_001/control";
    private static final String CAMPAIGN_ID = "business-graph-summary-formal-seed1-20261118-001";
    private static final String REVISION = "boundary-events-v1-native-rpc-runtime-scene-v1";
    private static final String[] COLUMNS = {
        "ordinal", "actor_seed", "group", "role", "source_recording_id", "recording_id",
        "run_index", "experiment_seed", "actor_sampling_seed", "event_directory", "event_bundle",
        "python_output", "target_training_update", "checkpoint_update"
    };
    private static final int JOB_COUNT = 512;
    private static final long TIMEOUT_MILLIS = 3600000L;
    private static volatile Plan cached;
    private static final java.util.Set<Long> started = new java.util.HashSet<Long>();

    private Baseline1CampaignPlan() { }

    private static final int RESUME_OFFSET = 1;
    private static final int REMAINING_JOB_COUNT = JOB_COUNT - RESUME_OFFSET;
    private static boolean controllerStarted = false;
    /** Only Main startup invokes this; all Freeform accessors remain read-only. */
    private static synchronized void ensureController(Row row) {
        if (controllerStarted) return;
        require(row.ordinal == RESUME_OFFSET, "recovery startup requires original ordinal 1");
        checkStop(absolute(CONTROL_DIRECTORY, "control directory"));
        try {
            java.nio.file.Path bridge = absolute("/home/jinx/Faith/experiments/lagrangian_ppo_v1/change_plans/token_attention_001/business_graph_summary_009/formal_training_001/recovery_001/bootstrap.py", "bridge path");
            require(sha256(java.nio.file.Files.readAllBytes(bridge)).equals("33909ad25453bb5b7b7f88af51a39de216c53cbc7b559d5ddaf6f8096f5ab6ef"),
                "one-click bridge byte SHA256 differs");
            Process process = new ProcessBuilder(
                "/home/jinx/miniconda3/envs/anylogic-env/bin/python", "-B", bridge.toString(),
                "--recovery", "/home/jinx/Faith/experiments/lagrangian_ppo_v1/change_plans/token_attention_001/business_graph_summary_009/formal_training_001/recovery_001/recovery.json", "--run-index", Long.toString(row.run),
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


    private static final class Row {
        final String[] cells;
        final int ordinal;
        final long run;
        final long seed;
        Row(String[] values) {
            cells = values;
            ordinal = integer(values[0], "ordinal");
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
        Plan(Row[] rows, java.util.Map<Long, Row> byRun) { this.rows = rows; this.byRun = byRun; }
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
                for (int i = 0; i < JOB_COUNT; i++) {
                    String[] values = lines[i + 1].split(",", -1);
                    require(values.length == COLUMNS.length, "master column count differs at ordinal " + i);
                    for (String value : values) require(!value.isEmpty() && value.equals(value.trim())
                        && !value.contains("\r") && !value.contains("\n"), "empty or malformed master field");
                    Row row = new Row(values);
                    require(row.ordinal == i, "master ordinal is not contiguous at " + i);
                    require(byRun.put(row.run, row) == null, "duplicate master run_index");
                    rows[i] = row;
                }
                cached = new Plan(rows, byRun);
                return cached;
            } catch (java.io.IOException error) {
                throw new IllegalStateException("Baseline 1 one-click STOP: cannot read frozen master", error);
            }
        }
    }

    private static Row row(int index) {
        require(index >= 0 && index < REMAINING_JOB_COUNT, "Freeform index outside remaining plan");
        return plan().rows[index + RESUME_OFFSET];
    }

    public static long runIndex(int index) { return row(index).run; }
    public static long environmentSeed(int index) { return row(index).seed; }
    public static String recordingId(int index) { return row(index).cells[5]; }
    public static String eventRoot(int index) { return row(index).cells[9]; }
    public static boolean materialEnabled(int index) { row(index); return true; }
    public static String phaseId(int index) {
        String recording = row(index).cells[5];
        require(recording.startsWith(CAMPAIGN_ID + "-"), "BG128 recording prefix differs");
        return recording.substring(CAMPAIGN_ID.length() + 1);
    }
    public static String materialRoot(int index) { return "/home/jinx/Faith/experiments/lagrangian_ppo_v1/runs/business-graph-summary-formal-seed1-20261118-001_materials" + "/" + phaseId(index); }
    public static String stageId(int index) { return phaseId(index).split("_", 3)[1]; }
    /** Metadata compatibility only. Python uses the fixed combined cost. */
    public static double waitWeight(int index) { row(index); return 0.2; }
    public static long actorSamplingSeed(int index) { return positive(row(index).cells[8], "actor sampling seed"); }
    public static boolean preparationOnly() { return false; }
    public static int jobCount() { return REMAINING_JOB_COUNT; }
    public static int resumeOffset() { return RESUME_OFFSET; }
    public static int frozenPlanJobCount() { return JOB_COUNT; }


    public static int dispatchPolicy(int index) {
        String group = row(index).cells[2];
        return group.equals("BASELINE") ? 3 : Integer.parseInt(group.substring(1));
    }
    public static boolean offlineEnabled(int index) { return row(index).cells[2].startsWith("P"); }
    public static String offlineSha256(int index) {
        String group = row(index).cells[2];
        if (group.equals("P0")) return "e0a0aed980fa1d029ee5cca7cbbe77567753edbdd1b584e802370f1a5d7fd5dd";
        if (group.equals("P1")) return "ceeac2cb9307959da2864ed0582f0ea08c1f6895c047fcf4ee05ee2b7fbd14ab";
        if (group.equals("P2")) return "f7007abd9987745d60d8c9272a1c6481dee533f67c0128cae37cd3fc4c208513";
        if (group.equals("P4")) return "b8adb0493ce4918f4fdd2865592fcfccdc791dda64191912a74882ecdef657e3";
        return "0000000000000000000000000000000000000000000000000000000000000000";
    }
    public static boolean eventRecordingEnabled(int index) { return offlineEnabled(index); }

    /** Called once at the very beginning of Main On startup, never in Freeform. */
    public static String awaitConfigSha256(long runIndex, long environmentSeed, String recordingId,
                                           String eventRoot, String instrumentationRevision) {
        java.nio.file.Path control = absolute(CONTROL_DIRECTORY, "control directory");
        checkStop(control);
        Row row = plan().byRun.get(runIndex);
        require(row != null, "run_index absent from frozen master");
        require(row.ordinal >= RESUME_OFFSET, "already captured original job must not be run again");
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
