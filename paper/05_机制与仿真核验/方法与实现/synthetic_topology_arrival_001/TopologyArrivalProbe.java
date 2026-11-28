import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Random;

import com.anylogic.engine.Agent;
import com.anylogic.engine.Engine;
import com.anylogic.engine.Point;
import com.anylogic.engine.Position;
import com.anylogic.engine.markup.ContinuousShortestPathData;
import com.anylogic.engine.markup.Level;
import com.anylogic.engine.markup.MarkupSegmentLine;
import com.anylogic.engine.markup.Network;
import com.anylogic.engine.markup.Path;
import com.anylogic.engine.markup.PointNode;
import com.anylogic.engine.presentation.ShapeDrawMode;
import com.anylogic.engine.presentation.ShapeTopLevelPresentationGroup;

public final class TopologyArrivalProbe {
    public static void main(String[] args) {
        Engine engine = new Engine();
        engine.setDefaultRandomGenerator(new Random(Long.parseLong(args[1])));
        NetworkModel model = new NetworkModel(engine, args);
        System.out.println("run_index,stage,event,time_s,x,y,target_x,target_y,error");
        double timeAtLoad = model.sampleLeg("start", "load", 0.0);
        model.sampleLeg("load", "dump", timeAtLoad);
    }

    private static final class NetworkModel extends Agent {
        private final ShapeTopLevelPresentationGroup presentation;
        private final long runIndex;
        private final double sampleSeconds;
        private final double speed;
        private final double stopSeconds;
        private final double tolerance;
        private final Map<String, PointNode> nodes = new LinkedHashMap<>();
        private final Network network;

        NetworkModel(Engine engine, String[] args) {
            super(engine, null, null);
            runIndex = Long.parseLong(args[0]);
            sampleSeconds = Double.parseDouble(args[2]);
            speed = Double.parseDouble(args[3]);
            stopSeconds = Double.parseDouble(args[4]);
            tolerance = Double.parseDouble(args[5]);

            Level level = new Level(this, "yard", ShapeDrawMode.SHAPE_DRAW_2D3D, 0);
            network = new Network(this, "haulRoads");
            for (String value : args[6].split(";")) {
                String[] fields = value.split(",");
                PointNode node = new PointNode();
                node.setPos(Double.parseDouble(fields[1]), Double.parseDouble(fields[2]), 0.0);
                nodes.put(fields[0], node);
                network.addAll(node);
            }
            for (String value : args[7].split(";")) {
                String[] fields = value.split(",");
                PointNode from = nodes.get(fields[0]);
                PointNode to = nodes.get(fields[1]);
                Path road = new Path();
                road.setBidirectional(true);
                road.addSegment(new MarkupSegmentLine(from.getX(), from.getY(), 0.0, to.getX(), to.getY(), 0.0));
                road.setSource(from);
                road.setTarget(to);
                network.addAll(road);
            }
            level.add(network);
            level.initialize();
            presentation = new ShapeTopLevelPresentationGroup(this, true, 0, 0, 0, 0, level);
            if (network.getNodeCount() != nodes.size() || network.getPathCount() != args[7].split(";").length) {
                throw new AssertionError("native network does not match configured node and road counts");
            }
        }

        @Override
        public ShapeTopLevelPresentationGroup getPresentationShape() {
            return presentation;
        }

        double sampleLeg(String startName, String stage, double startTime) {
            PointNode start = nodes.get(startName);
            PointNode target = nodes.get(stage);
            ContinuousShortestPathData route = new ContinuousShortestPathData();
            network.getPathData(new Point(start.getX(), start.getY()), new Point(target.getX(), target.getY()), route);
            if (route.isPlainMovement() || route.getNetwork() != network
                    || !Double.isFinite(route.distance) || route.distance <= 0.0) {
                throw new AssertionError(stage + " did not route through the configured network");
            }
            double duration = route.distance / speed;
            double endTime = startTime + duration;
            if (endTime > stopSeconds) {
                throw new AssertionError(stage + " exceeded the configured limit");
            }
            Position position = new Position();
            // In SDK 8.9 the offset is the distance remaining to the destination.
            double firstError = record(stage, "depart", startTime,
                    network.getPositionAtOffset(route, route.distance, position), target);
            if (firstError <= tolerance) {
                throw new AssertionError(stage + " was already at the coordinate target at departure");
            }
            for (double elapsed = sampleSeconds; elapsed < duration; elapsed += sampleSeconds) {
                record(stage, "sample", startTime + elapsed,
                        network.getPositionAtOffset(route, route.distance - elapsed * speed, position), target);
            }
            double lastError = record(stage, "arrival", endTime,
                    network.getPositionAtOffset(route, 0.0, position), target);
            if (lastError > tolerance) {
                throw new AssertionError(stage + " did not reach the target coordinates");
            }
            return endTime;
        }

        double record(String stage, String event, double time, Position position, PointNode target) {
            double error = Math.hypot(position.x - target.getX(), position.y - target.getY());
            if (!Double.isFinite(error)) {
                throw new AssertionError("nonfinite vehicle position");
            }
            System.out.printf(Locale.ROOT, "%d,%s,%s,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f%n",
                    runIndex, stage, event, time, position.x, position.y,
                    target.getX(), target.getY(), error);
            return error;
        }
    }
}
