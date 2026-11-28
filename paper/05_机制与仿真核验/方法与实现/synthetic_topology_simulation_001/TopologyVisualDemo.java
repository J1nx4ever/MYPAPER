package fusion;

import java.awt.Color;
import java.awt.Font;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Properties;
import java.util.Random;

import com.anylogic.engine.Agent;
import com.anylogic.engine.LengthUnits;
import com.anylogic.engine.SpeedUnits;
import com.anylogic.engine.markup.Level;
import com.anylogic.engine.markup.MarkupSegmentLine;
import com.anylogic.engine.markup.Network;
import com.anylogic.engine.markup.PointNode;
import com.anylogic.engine.presentation.LineStyle;
import com.anylogic.engine.presentation.ShapeDrawMode;
import com.anylogic.engine.presentation.ShapeOval;
import com.anylogic.engine.presentation.ShapeText;
import com.anylogic.engine.presentation.TextAlignment;

public final class TopologyVisualDemo {
    private static final double VIEW_X = 80.0;
    private static final double VIEW_Y = 340.0;
    private static final double PIXELS_PER_UNIT = 2.0;
    private static final Color INK = new Color(24, 49, 71);
    private static final Color BLUE = new Color(0, 105, 175);
    private static final Color ORANGE = new Color(204, 84, 24);
    private static final Color GREEN = new Color(17, 125, 80);
    private static final Color RED = new Color(176, 40, 58);
    private static final Map<String, String> LABELS = Map.of(
            "start", "起点", "junction", "中心路口", "load", "装载点", "dump", "卸载点",
            "west", "西路口", "east", "东路口", "south", "南路口");

    private enum Stage { LOAD, HOLD, DUMP, DONE, FAILED }
    private record XY(double x, double y) {
        double distanceTo(XY other) { return Math.hypot(x - other.x, y - other.y); }
        double displayX() { return VIEW_X + PIXELS_PER_UNIT * x; }
        double displayY() { return VIEW_Y - PIXELS_PER_UNIT * y; }
    }

    private final Agent owner;
    private final Properties config;
    private final String sha;
    private final Map<String, XY> coordinates = new LinkedHashMap<>();
    private final Map<String, PointNode> nodes = new LinkedHashMap<>();
    private final Level level;
    private final Network network;
    private final ShapeOval marker;
    private final ShapeText status;
    private final double speed;
    private final double tolerance;
    private final double stop;
    private final double loadHold;
    private Agent vehicle;
    private PrintWriter events;
    private Stage stage = Stage.LOAD;
    private double dumpDeparture;
    private double nextBreadcrumb;

    public TopologyVisualDemo(Agent owner, String configPath, String expectedSha) {
        this.owner = owner;
        sha = expectedSha;
        config = readConfig(configPath, expectedSha);
        speed = number("speed");
        tolerance = number("arrival_tolerance");
        stop = number("stop_seconds");
        loadHold = number("load_hold_seconds");
        if (speed <= 0 || tolerance <= 0 || stop <= 0 || loadHold < 0) {
            throw new IllegalArgumentException("Invalid simulation timing or tolerance");
        }
        owner.getEngine().setDefaultRandomGenerator(new Random(Long.parseLong(config.getProperty("environment_seed"))));

        level = new Level(owner, "topologyDemoLevel", ShapeDrawMode.SHAPE_DRAW_2D3D, 0.0);
        network = new Network(owner, "topologyDemoNetwork");
        for (String spec : config.getProperty("nodes").split(";")) {
            String[] parts = spec.split(",");
            XY xy = new XY(Double.parseDouble(parts[1]), Double.parseDouble(parts[2]));
            coordinates.put(parts[0], xy);
            PointNode node = new PointNode();
            node.setPos(xy.displayX(), xy.displayY());
            node.setRadius(5.0);
            node.setLineColor(INK);
            nodes.put(parts[0], node);
            network.addAll(node);
        }
        String[] roads = config.getProperty("roads").split(";");
        for (String spec : roads) {
            String[] endpoints = spec.split(",");
            PointNode from = nodes.get(endpoints[0]);
            PointNode to = nodes.get(endpoints[1]);
            com.anylogic.engine.markup.Path road = new com.anylogic.engine.markup.Path();
            road.setBidirectional(true);
            road.setLineColor(new Color(170, 183, 195));
            road.setLineWidth(4.0);
            road.addSegment(new MarkupSegmentLine(from.getX(), from.getY(), 0, to.getX(), to.getY(), 0));
            road.setSource(from);
            road.setTarget(to);
            network.addAll(road);
        }
        level.add(network);
        level.initialize();
        if (network.getNodeCount() != nodes.size() || network.getPathCount() != roads.length) {
            throw new IllegalStateException("Built network differs from the configured topology");
        }
        owner.getPresentationShape().add(level);
        addText(30, 30, String.format("路网 %s   |   %d 节点 · %d 双向道路", config.getProperty("layout"),
                network.getNodeCount(), network.getPathCount()), 22);
        addText(30, 63, "蓝色轨迹 → 装载点     橙色轨迹 → 卸载点", 14);
        for (Map.Entry<String, XY> entry : coordinates.entrySet()) {
            String name = entry.getKey();
            XY xy = entry.getValue();
            boolean top = List.of("load", "west").contains(name);
            addText(xy.displayX() + 9, xy.displayY() + (top ? -20 : 14),
                    String.format(Locale.ROOT, "%s (%.0f, %.0f)", LABELS.get(name), xy.x, xy.y), 12);
        }
        status = addText(30, 407, "准备发车", 16);
        marker = dot(coordinates.get("start"), 8.0, BLUE);
    }

    public void start(Agent probeVehicle, double now) {
        if (vehicle != null || now != 0.0) {
            throw new IllegalStateException("This visual run must start once at simulation time zero");
        }
        vehicle = probeVehicle;
        vehicle.setLevel(level);
        vehicle.setNetwork(network);
        vehicle.setLocation(nodes.get("start"));
        vehicle.setSpeed(owner.toLengthUnits(speed * PIXELS_PER_UNIT, LengthUnits.METER), SpeedUnits.MPS);
        events = openEvents();
        events.println("recording_id,run_index,environment_seed,config_sha256,layout,event,leg,time_s,"
                + "x,y,target_x,target_y,error,nodes,roads");
        depart("load", now);
    }

    public void tick(double now) {
        if (vehicle == null || stage == Stage.DONE || stage == Stage.FAILED) return;
        if (now >= stop) {
            fail(now, "等待超时");
            return;
        }
        if (stage == Stage.HOLD) {
            if (now + 1e-9 >= dumpDeparture) {
                stage = Stage.DUMP;
                depart("dump", now);
            }
            return;
        }
        String leg = activeLeg();
        XY location = vehiclePosition();
        if (distanceFromRoute(location, leg) > tolerance) {
            fail(now, "车辆偏离验收路线");
            return;
        }
        log("sample", leg, now, location);
        marker.setPos(vehicle.getX(), vehicle.getY());
        status.setText(String.format(Locale.ROOT, "%s   t = %.2f s   x = %.2f   y = %.2f",
                stage == Stage.LOAD ? "驶向装载点" : "驶向卸载点", now, location.x, location.y));
        if (now + 1e-9 >= nextBreadcrumb) {
            dot(location, 2.3, stage == Stage.LOAD ? BLUE : ORANGE);
            nextBreadcrumb = now + 1.0;
        }
    }

    public void onArrival(Agent source, double now) {
        if (source != vehicle || stage == Stage.DONE || stage == Stage.FAILED) return;
        if (stage == Stage.HOLD) {
            fail(now, "停留期间收到意外到达");
            return;
        }
        String leg = activeLeg();
        XY location = vehiclePosition();
        if (!(location.distanceTo(coordinates.get(leg)) <= tolerance)) {
            fail(now, "到达回调坐标与站点不符");
            return;
        }
        log("arrival", leg, now, location);
        marker.setPos(vehicle.getX(), vehicle.getY());
        if (stage == Stage.LOAD) {
            stage = Stage.HOLD;
            dumpDeparture = now + loadHold;
            status.setText("装载点坐标通过，等待发往卸载点");
        } else {
            log("pass", "dump", now, location);
            stage = Stage.DONE;
            status.setColor(GREEN);
            status.setText("PASS｜装载点和卸载点的到达坐标通过");
            close(now);
        }
    }

    public void close(double now) {
        if (events == null) return;
        if (stage != Stage.DONE && stage != Stage.FAILED) log("abort", activeLeg(), now, vehiclePosition());
        events.close();
        if (events.checkError()) throw new IllegalStateException("Could not close the event log");
        events = null;
    }

    private void depart(String leg, double now) {
        log("depart", leg, now, vehiclePosition());
        nextBreadcrumb = now;
        status.setText(leg.equals("load") ? "驶向装载点" : "驶向卸载点");
        vehicle.moveTo(nodes.get(leg));
        if (!vehicle.isMoving()) fail(now, "原生网络未建立行程");
    }

    private void fail(double now, String reason) {
        log("fail", activeLeg(), now, vehiclePosition());
        stage = Stage.FAILED;
        status.setColor(RED);
        status.setText("FAIL｜" + reason);
        close(now);
        vehicle.stop();
    }

    private String activeLeg() { return stage == Stage.LOAD || stage == Stage.HOLD ? "load" : "dump"; }

    private XY vehiclePosition() {
        return new XY((vehicle.getX() - VIEW_X) / PIXELS_PER_UNIT, (VIEW_Y - vehicle.getY()) / PIXELS_PER_UNIT);
    }

    private double distanceFromRoute(XY p, String leg) {
        String[] route = config.getProperty("route_" + leg).split(",");
        double best = Double.POSITIVE_INFINITY;
        for (int i = 1; i < route.length; i++) {
            XY a = coordinates.get(route[i - 1]);
            XY b = coordinates.get(route[i]);
            double dx = b.x - a.x;
            double dy = b.y - a.y;
            double t = ((p.x - a.x) * dx + (p.y - a.y) * dy) / (dx * dx + dy * dy);
            t = Math.max(0.0, Math.min(1.0, t));
            best = Math.min(best, p.distanceTo(new XY(a.x + t * dx, a.y + t * dy)));
        }
        return Double.isFinite(best) ? best : Double.POSITIVE_INFINITY;
    }

    private ShapeText addText(double x, double y, String text, int size) {
        ShapeText shape = new ShapeText(true, x, y, 0, INK, text,
                new Font("SansSerif", Font.PLAIN, size), TextAlignment.ALIGNMENT_LEFT);
        owner.getPresentationShape().add(shape);
        return shape;
    }

    private ShapeOval dot(XY point, double radius, Color color) {
        ShapeOval dot = new ShapeOval(true, point.displayX(), point.displayY(), 0, Color.WHITE, color,
                radius, radius, 1.0, LineStyle.LINE_STYLE_SOLID);
        owner.getPresentationShape().add(dot);
        return dot;
    }

    private void log(String event, String leg, double now, XY location) {
        XY target = coordinates.get(leg);
        events.printf(Locale.ROOT, "%s,%s,%s,%s,%s,%s,%s,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%d,%d%n",
                config.getProperty("recording_id"), config.getProperty("run_index"),
                config.getProperty("environment_seed"), sha, config.getProperty("layout"), event, leg,
                now, location.x, location.y, target.x, target.y, location.distanceTo(target),
                network.getNodeCount(), network.getPathCount());
        if (events.checkError()) throw new IllegalStateException("Could not write the event log");
    }

    private PrintWriter openEvents() {
        try {
            Path directory = Path.of(config.getProperty("event_dir"));
            Files.createDirectories(directory);
            return new PrintWriter(Files.newBufferedWriter(directory.resolve("events.csv"),
                    StandardOpenOption.CREATE_NEW, StandardOpenOption.WRITE));
        } catch (IOException e) {
            throw new UncheckedIOException("Cannot open a fresh recording for this Simulation", e);
        }
    }

    private double number(String key) {
        double value = Double.parseDouble(config.getProperty(key));
        if (!Double.isFinite(value)) throw new IllegalArgumentException("Invalid numeric value for " + key);
        return value;
    }

    private static Properties readConfig(String path, String expectedSha) {
        try {
            byte[] bytes = Files.readAllBytes(Path.of(path));
            String actualSha = HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(bytes));
            if (!actualSha.equals(expectedSha)) throw new IllegalArgumentException("Config SHA256 mismatch");
            Properties config = new Properties();
            config.load(new ByteArrayInputStream(bytes));
            return config;
        } catch (IOException e) {
            throw new UncheckedIOException("Could not read the topology config", e);
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException(e);
        }
    }
}
