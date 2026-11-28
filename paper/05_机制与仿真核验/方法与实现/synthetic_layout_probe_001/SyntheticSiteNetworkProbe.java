import java.util.Locale;

import com.anylogic.engine.Agent;
import com.anylogic.engine.Engine;
import com.anylogic.engine.Point;
import com.anylogic.engine.markup.ContinuousShortestPathData;
import com.anylogic.engine.markup.Level;
import com.anylogic.engine.markup.MarkupSegmentLine;
import com.anylogic.engine.markup.Network;
import com.anylogic.engine.markup.Path;
import com.anylogic.engine.markup.PointNode;
import com.anylogic.engine.presentation.ShapeDrawMode;
import com.anylogic.engine.presentation.ShapeTopLevelPresentationGroup;

public class SyntheticSiteNetworkProbe {
    private static final class NetworkOwner extends Agent {
        private ShapeTopLevelPresentationGroup presentation;

        NetworkOwner(Engine engine) {
            super(engine, null, null);
        }

        void bindPresentation(Level level) {
            presentation = new ShapeTopLevelPresentationGroup(this, true, 0, 0, 0, 0, level);
        }

        @Override
        public ShapeTopLevelPresentationGroup getPresentationShape() {
            return presentation;
        }
    }

    public static void main(String[] args) {
        checkLayout("A", 100, 0, 180, 0, false, 80);
        checkLayout("B", 200, 100, 400, 300, false, Math.hypot(200, 200));
        checkLayout("C", 200, 100, 400, 300, true, 400);
        System.out.println("PASS: all site bindings, network routes and path lengths match.");
    }

    private static void checkLayout(
            String name, double loadX, double loadY, double dumpX, double dumpY,
            boolean viaJunction, double expectedLength) {
        // Each layout gets its own SDK objects. No experiment is created and no clock is started.
        Engine engine = new Engine();
        NetworkOwner owner = new NetworkOwner(engine);
        PointNode load = node(loadX, loadY);
        PointNode dump = node(dumpX, dumpY);
        Network network = new Network(owner, "syntheticNetwork");

        Path roadToDump;
        Path roadToJunction = null;
        if (viaJunction) {
            PointNode junction = node(loadX, dumpY);
            roadToJunction = connect(load, junction);
            roadToDump = connect(junction, dump);
            network.addAll(load, junction, dump, roadToJunction, roadToDump);
        } else {
            roadToDump = connect(load, dump);
            network.addAll(load, dump, roadToDump);
        }

        Level level = new Level(owner, "syntheticLevel", ShapeDrawMode.SHAPE_DRAW_2D3D, 0);
        level.add(network);
        level.initialize();
        owner.bindPresentation(level);

        Agent loadSite = new Agent(engine, owner, null);
        Agent dumpSite = new Agent(engine, owner, null);
        loadSite.setLocation(load);
        dumpSite.setLocation(dump);
        checkClose(name + " load X", loadSite.getX(), loadX);
        checkClose(name + " load Y", loadSite.getY(), loadY);
        checkClose(name + " dump X", dumpSite.getX(), dumpX);
        checkClose(name + " dump Y", dumpSite.getY(), dumpY);

        ContinuousShortestPathData route = new ContinuousShortestPathData();
        network.getPathData(new Point(loadSite.getX(), loadSite.getY()),
                new Point(dumpSite.getX(), dumpSite.getY()), route);
        if (route.isPlainMovement() || route.getNetwork() != network) {
            throw new AssertionError(name + " did not route through its constructed network");
        }

        double nativeLength = roadToDump.length();
        if (roadToJunction != null) {
            nativeLength += roadToJunction.length();
        }
        checkClose(name + " path geometry", nativeLength, expectedLength);
        checkClose(name + " route distance", route.distance, expectedLength);
        System.out.printf(Locale.ROOT,
                "%s load=(%.0f,%.0f) dump=(%.0f,%.0f) viaJunction=%s geometry=%.5f route=%.5f%n",
                name, loadX, loadY, dumpX, dumpY, viaJunction, nativeLength, route.distance);
    }

    private static PointNode node(double x, double y) {
        PointNode node = new PointNode();
        node.setPos(x, y, 0);
        return node;
    }

    private static Path connect(PointNode from, PointNode to) {
        Path path = new Path();
        path.setBidirectional(true);
        path.addSegment(new MarkupSegmentLine(from.getX(), from.getY(), 0, to.getX(), to.getY(), 0));
        path.setSource(from);
        path.setTarget(to);
        return path;
    }

    private static void checkClose(String field, double actual, double expected) {
        if (!Double.isFinite(actual) || Math.abs(actual - expected) > 1e-7) {
            throw new AssertionError(field + " was " + actual + ", expected " + expected);
        }
    }
}
