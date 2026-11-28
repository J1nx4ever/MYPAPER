"""Build the two sequential 50-shift native experiments for the frozen Actor."""

import argparse
from hashlib import sha256
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'physical_trip_001'))
from prepare_trip_alp import named_block, replace

BASE_SHA = 'd80964794ab0f9c70efc8b34e9457e161e380b31e30ad73146938af0077dd071'
BASE_MODEL = 'DRL-PPO-baseline-topologydemo-frozen-policy-scaled'
MODEL = 'DRL-PPO-baseline-topologydemo-frozen-policy-stats50'
BASE_EXPERIMENT = 'Topology_Frozen_Policy_B_Scaled'


def experiment(source, group, number, experiment_id):
    old_id = ET.fromstring(source).findtext('Id')
    source = replace(source, f'<Id>{old_id}</Id>', f'<Id>{experiment_id}</Id>')
    source = replace(source, f'<Name><![CDATA[{BASE_EXPERIMENT}]]>',
                     f'<Name><![CDATA[Topology_Frozen_Stats_{group}50]]>')
    source = replace(source, 'Topology frozen Actor capacity-scaled B',
                     f'Topology frozen Actor 50 independent shifts {group}')
    source = replace(source, '<NumberOfRuns>1</NumberOfRuns>', '<NumberOfRuns>50</NumberOfRuns>')
    return replace(source, '<Code><![CDATA[2]]></Code>',
                   f'<Code><![CDATA[{number} + index]]></Code>')


def stats_witness(source):
    source = replace(source, str(HERE.parent / 'capacity_scaled_001' / 'configs'), str(HERE / 'configs'))
    source = replace(source, '    private static final String[] CASES = {"A", "B"};\n', '')
    start = source.index('    public static int caseNumber(int index) {')
    end = source.index('            byte[] bytes = Files.readAllBytes(', start)
    source = source[:start] + '''    public static TopologyWitness open(Main main, int number) {
        int group = number / 100, slot = number % 100;
        require((group == 1 || group == 2) && slot >= 1 && slot <= 50, "Invalid topology statistics slot");
        String label = group == 1 ? "A" : "B";
        String key = String.format(java.util.Locale.ROOT, "%s_%03d", label, slot);
        try {
''' + source[end:]
    source = replace(source, '            return new TopologyWitness(main, config, digest);', '''            require(config.get("experiment").getAsString().equals("Topology_Frozen_Stats_" + label + "50")
                    && config.get("layout_id").getAsString().equals(group == 1 ? "tree_A" : "loop_B"),
                    "Configuration selects the wrong experiment or layout for " + key);
            return new TopologyWitness(main, config, digest);''')
    source = replace(source, 'main.deterministicCheck = true;', 'main.deterministicCheck = false;')
    source = replace(source, '"topology-full-shift-v1", main.experimentSeed,',
                     '"topology-stats50-v1", main.experimentSeed,')
    return replace(source,
        '            for (double grade : site.blockGrades) require(grade == site.grade, "Unexpected grade block");',
        '''            var gradeCheck = new Random(main.experimentSeed ^ (0x9E3779B97F4A7C15L * (site.siteId + 1L)));
            for (double grade : site.blockGrades) {
                double delta = (2.0 * gradeCheck.nextDouble() - 1.0) * site.gradeHalfRange;
                double expected = site.isOre ? Math.max(0.0, Math.min(1.0, site.grade + delta)) : 0.0;
                require(grade == expected, "Grade block differs from its frozen site and environment seed");
            }''')


def compact_recorder(source):
    source = replace(source, '    private boolean closed = false;', '''    private boolean closed = false;
    private final boolean compactFifo;
    private double[] replayBatch, pendingBatch;
    private double replayRemaining, pendingStart, pendingEnd, pendingAfter, pendingGrade, pendingTime;
    private double pendingSourceTons;
    private int pendingWindow, pendingCount;
    private String pendingId;
    private long sourceCount, groupCount;''')
    source = replace(source, '        this.episodeId = episodeId;',
                     '        this.compactFifo = "topology-stats50-v1".equals(instrumentationRevision);\n'
                     '        this.episodeId = episodeId;')
    emit = '    public synchronized long emit(double now, String type, Map<String, Object> data) {'
    source = replace(source, emit, emit + '''
        flushProcesses();
        return writeEvent(now, type, data);
    }

    private long writeEvent(double now, String type, Map<String, Object> data) {''')
    source = replace(source, '    public void fifoProcess(double now,', '    public synchronized void fifoProcess(double now,')
    source = replace(source, '        emit(now, "FIFO_PROCESS",', '''        if (compactFifo) {
            if (closed || !Double.isFinite(now) || now < lastTime || (pendingBatch != null && now < pendingTime)
                    || !(start < end) || !Double.isFinite(tons) || tons <= 0 || tons > batch[0]
                    || !Double.isFinite(batch[1]) || batch[1] < 0 || batch[1] > 1)
                throw new IllegalStateException("Invalid compact FIFO source slice");
            if (pendingBatch != batch || pendingEnd != start || pendingWindow != window || pendingGrade != batch[1])
                flushProcesses();
            if (pendingBatch == null) {
                if (replayBatch != batch) {
                    replayBatch = batch;
                    replayRemaining = batch[0];
                }
                pendingBatch = batch;
                pendingId = id;
                pendingStart = start;
                pendingWindow = window;
                pendingGrade = batch[1];
                pendingCount = 0;
                pendingSourceTons = 0.0;
            }
            // The recorder callback precedes physical batch subtraction.
            pendingAfter = batch[0] - tons;
            pendingEnd = end;
            pendingTime = now;
            pendingCount++;
            pendingSourceTons += tons;
            sourceCount++;
            return;
        }
        emit(now, "FIFO_PROCESS",''')
    source = replace(source, '    public void fifoDrop(double now, double[] batch) {', '''    private void flushProcesses() {
        if (pendingBatch == null) return;
        // Reconstruct the same residual in readers that subtract one aggregate
        // per group. This also preserves the strict FIFO_DROP numerical tail.
        double tons = replayRemaining - pendingAfter;
        if (!(tons > 0.0) || Math.abs(tons - pendingSourceTons) > 1e-6)
            throw new IllegalStateException("Compact FIFO group mass disagrees with native slices");
        writeEvent(pendingTime, "FIFO_PROCESS", fields("batch_id", pendingId, "interval_start_seconds", pendingStart,
            "interval_end_seconds", pendingEnd, "window_index", pendingWindow, "tons", tons,
            "metal_tons", tons * pendingGrade, "source_slice_count", pendingCount, "source_tons_sum", pendingSourceTons));
        replayRemaining -= tons;
        pendingBatch = null;
        groupCount++;
    }

    public void fifoDrop(double now, double[] batch) {''')
    source = replace(source, '            manifest.put("event_count", sequence);', '''            manifest.put("event_count", sequence);
            if (compactFifo) {
                manifest.put("fifo_source_slice_count", sourceCount);
                manifest.put("fifo_process_event_count", groupCount);
            }''')
    return replace(source, '        try { writer.close(); closed = true; }',
                   '        flushProcesses();\n        try { writer.close(); closed = true; }')


def build(base, destination):
    raw = base.read_bytes()
    if sha256(raw).hexdigest() != BASE_SHA:
        raise ValueError('Expected the sealed capacity-scaled ALP')
    source = raw.decode('utf-8')
    model = ET.fromstring(source).find('Model')
    next_id = 1 + max(int(v.text) for v in model.iter('Id') if v.text and v.text.isdecimal())
    prior = named_block(source, 'ParamVariationExperiment', BASE_EXPERIMENT)
    source = replace(source, prior, experiment(prior, 'A', 101, next_id) + '\n'
                     + experiment(prior, 'B', 201, int(ET.fromstring(prior).findtext('Id'))))
    witness = named_block(source, 'JavaClass', 'TopologyWitness')
    source = replace(source, witness, stats_witness(witness))
    recorder = named_block(source, 'JavaClass', 'AcceptanceEventRecorder')
    source = replace(source, recorder, compact_recorder(recorder))
    source = replace(source, f'<Name><![CDATA[{BASE_MODEL}]]>', f'<Name><![CDATA[{MODEL}]]>')
    model = ET.fromstring(source).find('Model')
    for group, start in [('A', 101), ('B', 201)]:
        item, = (e for e in model.findall('Experiments/ParamVariationExperiment')
                 if e.findtext('Name') == f'Topology_Frozen_Stats_{group}50')
        if (item.findtext('NumberOfRuns') != '50' or item.findtext('AllowParallelEvaluations') != 'false'
                or item.findtext('ReplicationsProperties/UseReplication') != 'false'
                or item.findtext('FreeformParamValue/Expression/Code') != f'{start} + index'
                or item.findtext('ModelTimeProperties/FinalTime') != '28801.0'):
            raise ValueError('Expected 50 sequential independent eight-hour runs per group')
    with destination.open('xb') as stream:
        stream.write(source.encode('utf-8'))
    print(sha256(destination.read_bytes()).hexdigest(), destination)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('base', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    build(args.base, args.destination)
