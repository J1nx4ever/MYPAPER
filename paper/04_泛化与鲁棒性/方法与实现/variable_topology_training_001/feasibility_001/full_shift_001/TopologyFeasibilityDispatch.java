package fusion;

import com.anylogic.engine.markup.PointNode;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.IdentityHashMap;
import java.util.Map;

import static com.anylogic.engine.TimeUnits.SECOND;

/** A fixed, causal rule for the deterministic A/B shift feasibility check. */
public final class TopologyFeasibilityDispatch {
    private static final double EMPTY_SPEED_MPS = 6.5;
    private static final double LOADED_SPEED_MPS = 5.0;
    private static final double BLEND_TARGET = 0.269;
    private static final double BLEND_SECONDS_PER_TON = 60.0;
    private static final double EXTRA_LOAD_WAIT_WEIGHT = 0.5;
    private static final double PLANNED_WASTE_TONS = 4300.0;
    private static final double WASTE_LOOKAHEAD_SECONDS = 420.0;
    private static final double WASTE_RETURN_BONUS_TONS = 350.0;
    private static final double TON_TOLERANCE = 1e-6;

    private record PendingLoad(Truck truck, double arrival, double duration, long sequence) {}

    private final Main main;
    private final Map<PointNode, String> nodeNames = new IdentityHashMap<>();
    private final PendingLoad[] pendingLoads = new PendingLoad[9];
    private final DumpSite oreDump;
    private final DumpSite wasteDump;
    private final String wasteDumpNode;
    private double committedOreTons;
    private double committedOreMetalTons;
    private long dispatchSequence;

    public TopologyFeasibilityDispatch(Main main) {
        this.main = main;
        for (var entry : main.roadPointNodeMap.entrySet()) {
            nodeNames.put(entry.getValue(), entry.getKey());
        }
        oreDump = main.getDumpSiteById(0);
        wasteDump = main.getDumpSiteById(1);
        if (!main.deterministicCheck || oreDump == null || !oreDump.acceptsOre
                || wasteDump == null || !wasteDump.acceptsWaste || main.loadSites.size() != 4) {
            throw new IllegalStateException("Feasibility dispatch requires the deterministic four-source scene");
        }
        wasteDumpNode = nodeName(wasteDump.dumpNode);
    }

    public int choose(Truck truck) {
        double now = main.time(SECOND);
        double wasteCommitted = main.getWasteTaskCommittedTons();
        double pacedWaste = PLANNED_WASTE_TONS
                * Math.min(1.0, (now + WASTE_LOOKAHEAD_SECONDS) / main.rlShiftSeconds);
        if (wasteDumpNode.equals(truck.currentNode)) pacedWaste += WASTE_RETURN_BONUS_TONS;
        double oreBalance = committedOreMetalTons - BLEND_TARGET * committedOreTons;

        int bestOre = -1;
        double bestOreScore = Double.POSITIVE_INFINITY;
        int legalWaste = -1;
        for (int slot = 0; slot < main.loadSites.size(); slot++) {
            LoadSite site = main.getLoadSiteById(slot);
            double tons = site.getDispatchableTons(truck);
            if (tons <= TON_TOLERANCE) continue;

            String source = nodeName(site.loadNode);
            double arrival = now + travelSeconds(truck.currentNode, source, false);
            double loadStart = expectedLoadStart(site, arrival, now);
            DumpSite destination = site.isOre ? oreDump : wasteDump;
            double finish = loadStart + site.obsMeanLoadMinutes(tons) * 60.0
                    + travelSeconds(source, nodeName(destination.dumpNode), true)
                    + destination.obsMeanDumpMinutes(truck) * 60.0;

            if (!site.isOre) {
                if (wasteCommitted < pacedWaste && wasteCommitted < PLANNED_WASTE_TONS
                        && finish < main.rlShiftSeconds) return slot;
                legalWaste = slot;
                continue;
            }
            double nextBalance = oreBalance + tons * (site.grade - BLEND_TARGET);
            double score = finish - now + EXTRA_LOAD_WAIT_WEIGHT * (loadStart - arrival)
                    + BLEND_SECONDS_PER_TON * Math.abs(nextBalance);
            if (score < bestOreScore) {
                bestOreScore = score;
                bestOre = slot;
            }
        }
        return bestOre >= 0 ? bestOre : legalWaste;
    }

    public void committed(Truck truck) {
        LoadSite site = truck.targetShovel;
        double tons = truck.plannedPayloadTons;
        if (site == null || !Double.isFinite(tons) || tons <= TON_TOLERANCE) {
            throw new IllegalStateException("A dispatch commitment requires a reserved load");
        }
        double arrival = main.time(SECOND) + travelSeconds(truck.currentNode, nodeName(site.loadNode), false);
        pendingLoads[truck.truckId] = new PendingLoad(truck, arrival,
                site.obsMeanLoadMinutes(tons) * 60.0, dispatchSequence++);
        if (site.isOre) {
            committedOreTons += tons;
            committedOreMetalTons += tons * site.grade;
        }
    }

    private double expectedLoadStart(LoadSite site, double arrival, double now) {
        double available = now;
        var queue = new ArrayList<PendingLoad>();
        for (PendingLoad pending : pendingLoads) {
            if (pending == null) continue;
            Truck other = pending.truck();
            if (other.targetShovel != site || other.plannedPayloadTons <= TON_TOLERANCE) continue;
            if ("LOADING".equals(other.executionPhase)) {
                available = Math.max(available, 60.0 * other.loadStartTime + pending.duration());
            } else {
                double expected = other.isWaitingForLoadResource
                        ? 60.0 * other.queueEnterTime : Math.max(now, pending.arrival());
                if (expected <= arrival) {
                    queue.add(new PendingLoad(other, expected, pending.duration(), pending.sequence()));
                }
            }
        }
        queue.sort(Comparator.comparingDouble(PendingLoad::arrival).thenComparingLong(PendingLoad::sequence));
        for (PendingLoad pending : queue) {
            available = Math.max(available, pending.arrival()) + pending.duration();
        }
        return Math.max(available, arrival);
    }

    private String nodeName(PointNode node) {
        String name = nodeNames.get(node);
        if (name == null) throw new IllegalStateException("A station is outside the active network");
        return name;
    }

    private double travelSeconds(String from, String to, boolean loaded) {
        var route = main.routeNodeMap.get(from + "_" + to);
        if (route == null || route.size() < 2) {
            throw new IllegalStateException("Missing dispatch route " + from + " -> " + to);
        }
        // The loaded statechart waits one second before departure and after each road.
        double seconds = loaded ? route.size() : 0.0;
        double speed = loaded ? LOADED_SPEED_MPS : EMPTY_SPEED_MPS;
        for (int i = 1; i < route.size(); i++) {
            var segment = main.segmentMap.get(route.get(i - 1) + "_" + route.get(i));
            if (segment == null || segment.closed || !Double.isFinite(segment.length)
                    || !Double.isFinite(segment.speedLimit) || segment.length <= 0.0 || segment.speedLimit <= 0.0) {
                throw new IllegalStateException("Invalid road on dispatch route " + from + " -> " + to);
            }
            seconds += segment.length / Math.min(speed, segment.speedLimit);
        }
        return seconds;
    }
}
