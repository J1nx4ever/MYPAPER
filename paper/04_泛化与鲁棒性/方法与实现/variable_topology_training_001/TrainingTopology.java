package fusion;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Properties;
import java.util.Set;
import java.util.TreeMap;

import com.anylogic.engine.Agent;
import com.anylogic.engine.LengthUnits;
import com.anylogic.engine.Scale;
import com.anylogic.engine.markup.Level;
import com.anylogic.engine.markup.MarkupSegmentLine;
import com.anylogic.engine.markup.Network;
import com.anylogic.engine.markup.Path;
import com.anylogic.engine.markup.PointNode;
import com.anylogic.engine.presentation.ShapeDrawMode;

public final class TrainingTopology {
    public record NodeSpec(int wireId, String canonicalName, double xMeters, double yMeters) {}

    public record Road(int index, int fromNodeId, int toNodeId, double speedFromToMps,
            double speedToFromMps, Path nativePath, double lengthMeters) {}

    private record RoadSpec(int index, int fromNodeId, int toNodeId, double speedFromToMps,
            double speedToFromMps) {}

    private final String layoutId;
    private final String sceneSha256;
    private final Properties properties;
    private final Level level;
    private final Network network;
    private final List<NodeSpec> nodeSpecs;
    private final List<Road> roads;
    private final Map<Integer, PointNode> nodes = new HashMap<>();
    private final Map<String, PointNode> namedNodes = new HashMap<>();

    public static TrainingTopology load(Agent owner, String sceneFile, String expectedSha256) {
        Objects.requireNonNull(owner, "owner");
        if (expectedSha256 == null || !expectedSha256.matches("[0-9a-fA-F]{64}")) {
            throw new IllegalArgumentException("A SHA256 of the exact topology scene bytes is required");
        }
        try {
            byte[] bytes = Files.readAllBytes(java.nio.file.Path.of(sceneFile));
            String actualSha256 = HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(bytes));
            if (!actualSha256.equalsIgnoreCase(expectedSha256)) {
                throw new IllegalArgumentException("Topology scene SHA256 mismatch for " + sceneFile
                        + ": expected " + expectedSha256 + ", got " + actualSha256);
            }
            Properties properties = new Properties() {
                @Override
                public synchronized Object put(Object key, Object value) {
                    if (containsKey(key)) throw new IllegalArgumentException("Duplicate scene key " + key);
                    return super.put(key, value);
                }
            };
            properties.load(new ByteArrayInputStream(bytes));
            return new TrainingTopology(owner, properties, actualSha256);
        } catch (IOException e) {
            throw new UncheckedIOException("Could not read topology scene " + sceneFile, e);
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("SHA-256 is unavailable", e);
        }
    }

    private TrainingTopology(Agent owner, Properties properties, String sceneSha256) {
        if (!"dispatch.topology-scene.v1".equals(properties.getProperty("format"))) {
            throw new IllegalArgumentException("Expected topology scene format dispatch.topology-scene.v1");
        }
        this.sceneSha256 = sceneSha256;
        this.properties = properties;
        layoutId = required(properties, "layout_id");

        Map<Integer, NodeSpec> configuredNodes = new TreeMap<>();
        Map<Integer, RoadSpec> configuredRoads = new TreeMap<>();
        Set<String> nodeNames = new HashSet<>();
        for (String key : properties.stringPropertyNames()) {
            if (key.startsWith("node.")) {
                int id = index(key.substring("node.".length()), key);
                String[] fields = fields(properties, key, 3);
                if (!nodeNames.add(fields[0])) {
                    throw new IllegalArgumentException("Duplicate topology node name " + fields[0]);
                }
                configuredNodes.put(id, new NodeSpec(id, fields[0],
                        finite(fields[1], key), finite(fields[2], key)));
            } else if (key.startsWith("road.")) {
                int id = index(key.substring("road.".length()), key);
                String[] fields = fields(properties, key, 4);
                configuredRoads.put(id, new RoadSpec(id, index(fields[0], key), index(fields[1], key),
                        positive(finite(fields[2], key), key), positive(finite(fields[3], key), key)));
            }
        }
        if (configuredNodes.size() < 2 || configuredRoads.isEmpty()) {
            throw new IllegalArgumentException("A topology scene requires at least two nodes and one road");
        }
        validateRoads(configuredNodes, configuredRoads);
        nodeSpecs = List.copyOf(configuredNodes.values());

        network = new Network(owner, "trainingTopologyNetwork");
        level = new Level(owner, "trainingTopologyLevel", ShapeDrawMode.SHAPE_DRAW_2D3D, 0.0);
        Scale scale = owner.getScale();
        for (NodeSpec spec : nodeSpecs) {
            PointNode node = new PointNode();
            node.setPos(coordinate(scale, spec.xMeters()), coordinate(scale, spec.yMeters()), 0.0);
            nodes.put(spec.wireId(), node);
            namedNodes.put(spec.canonicalName(), node);
            network.addAll(node);
        }
        List<Path> nativePaths = new ArrayList<>();
        for (RoadSpec spec : configuredRoads.values()) {
            PointNode from = node(spec.fromNodeId());
            PointNode to = node(spec.toNodeId());
            Path path = new Path();
            path.setBidirectional(true);
            path.setSource(from);
            path.setTarget(to);
            path.addSegment(new MarkupSegmentLine(from.getX(), from.getY(), 0.0, to.getX(), to.getY(), 0.0));
            network.addAll(path);
            nativePaths.add(path);
        }
        level.add(network);
        level.initialize();
        if (network.getNodeCount() != nodeSpecs.size() || network.getPathCount() != nativePaths.size()) {
            throw new IllegalStateException("Native topology does not match the scene counts");
        }

        List<Road> builtRoads = new ArrayList<>();
        for (RoadSpec spec : configuredRoads.values()) {
            Path path = nativePaths.get(spec.index());
            if (path.getSource() != node(spec.fromNodeId()) || path.getTarget() != node(spec.toNodeId())
                    || !path.isBidirectional()) {
                throw new IllegalStateException("Native road " + spec.index() + " does not match the scene endpoints");
            }
            double length = positive(path.length(LengthUnits.METER), "native road " + spec.index());
            double expected = distance(configuredNodes.get(spec.fromNodeId()), configuredNodes.get(spec.toNodeId()));
            if (Math.abs(length - expected) > Math.max(1e-7, expected * 1e-9)) {
                throw new IllegalStateException("Native road " + spec.index() + " has length " + length
                        + " m, expected " + expected + " m");
            }
            builtRoads.add(new Road(spec.index(), spec.fromNodeId(), spec.toNodeId(), spec.speedFromToMps(),
                    spec.speedToFromMps(), path, length));
        }
        roads = List.copyOf(builtRoads);
    }

    public String layoutId() { return layoutId; }
    public String sceneSha256() { return sceneSha256; }
    public Level level() { return level; }
    public Network network() { return network; }
    public List<NodeSpec> nodeSpecs() { return nodeSpecs; }
    public List<Road> roads() { return roads; }
    public String setting(String key) { return required(properties, key); }

    public PointNode node(int wireId) {
        PointNode node = nodes.get(wireId);
        if (node == null) throw new IllegalArgumentException("Unknown topology node ID " + wireId);
        return node;
    }

    public PointNode node(String canonicalName) {
        PointNode node = namedNodes.get(canonicalName);
        if (node == null) throw new IllegalArgumentException("Unknown topology node name " + canonicalName);
        return node;
    }

    private static void validateRoads(Map<Integer, NodeSpec> nodes, Map<Integer, RoadSpec> roads) {
        Set<List<Integer>> endpoints = new HashSet<>();
        int nextIndex = 0;
        for (RoadSpec road : roads.values()) {
            int index = road.index();
            int from = road.fromNodeId();
            int to = road.toNodeId();
            if (index != nextIndex++) throw new IllegalArgumentException("Road indices must be consecutive from 0");
            if (from == to || !nodes.containsKey(from) || !nodes.containsKey(to)) {
                throw new IllegalArgumentException("Invalid endpoint IDs on road " + index);
            }
            if (!endpoints.add(List.of(Math.min(from, to), Math.max(from, to)))) {
                throw new IllegalArgumentException("Duplicate endpoints on road " + index);
            }
            positive(distance(nodes.get(from), nodes.get(to)), "configured road " + index);
        }
    }

    private static double distance(NodeSpec from, NodeSpec to) {
        return Math.hypot(to.xMeters() - from.xMeters(), to.yMeters() - from.yMeters());
    }

    private static double coordinate(Scale scale, double meters) {
        double pixels = scale.toPixels(meters, LengthUnits.METER);
        if (!Double.isFinite(pixels)) throw new IllegalArgumentException("Model coordinate exceeds the scale");
        return pixels;
    }

    private static String required(Properties properties, String key) {
        String value = properties.getProperty(key, "").trim();
        if (value.isEmpty()) throw new IllegalArgumentException("Missing topology scene key " + key);
        return value;
    }

    private static String[] fields(Properties properties, String key, int count) {
        String[] fields = required(properties, key).split(",", -1);
        if (fields.length != count) throw new IllegalArgumentException("Invalid field count for " + key);
        for (int i = 0; i < count; i++) {
            fields[i] = fields[i].trim();
            if (fields[i].isEmpty()) throw new IllegalArgumentException("Empty field in " + key);
        }
        return fields;
    }

    private static int index(String value, String key) {
        if (!value.matches("0|[1-9][0-9]*")) {
            throw new IllegalArgumentException("Expected a nonnegative integer in " + key + ": " + value);
        }
        return Integer.parseInt(value);
    }

    private static double finite(String value, String key) {
        double number = Double.parseDouble(value);
        if (!Double.isFinite(number)) throw new IllegalArgumentException("Nonfinite number in " + key);
        return number;
    }

    private static double positive(double value, String key) {
        if (!(value > 0.0) || !Double.isFinite(value)) {
            throw new IllegalArgumentException("Expected a positive finite number in " + key);
        }
        return value;
    }
}
