"""Build the B fine-tuning and checkpoint-evaluation experiments from the sealed B50 model."""

import argparse
from hashlib import sha256
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'physical_trip_001'))
from prepare_trip_alp import named_block, replace

BASE_SHA = 'e3bc2214ba64493d27ad2a0ea5346570ecc1c5fa4df2b2d884528e66465ceccd'
BASE_MODEL = 'DRL-PPO-baseline-topologydemo-frozen-policy-stats50'
MODEL = 'DRL-PPO-baseline-topologydemo-b-finetune'
EXPERIMENTS = (
    *((f'Topology_B_Train{i+1}', 1001 + 32*i, 32) for i in range(4)),
    ('Topology_B_Validate80', 2001, 80),
    ('Topology_B_Test50', 3001, 50),
)


def experiment(template, name, first, count, item_id):
    original_id = ET.fromstring(template).findtext('Id')
    source = replace(template, f'<Id>{original_id}</Id>', f'<Id>{item_id}</Id>')
    source = replace(source, 'Topology_Frozen_Stats_B50', name)
    source = replace(source, 'Topology frozen Actor 50 independent shifts B', name)
    source = replace(source, '<NumberOfRuns>50</NumberOfRuns>', f'<NumberOfRuns>{count}</NumberOfRuns>')
    return replace(source, '<Code><![CDATA[201 + index]]></Code>',
                   f'<Code><![CDATA[{first} + index]]></Code>')


def witness(source):
    source = replace(source, str(HERE.parent / 'paired_50_001' / 'configs'), str(HERE / 'configs'))
    start = source.index('        int group = number / 100, slot = number % 100;')
    end = source.index('        try {', start)
    source = source[:start] + '''        int group = number / 1000, slot = number % 1000;
        require(slot > 0 && ((group == 1 && slot <= 128) || (group == 2 && slot <= 80)
                || (group == 3 && slot <= 50)), "Invalid B fine-tuning slot");
        String phase = group == 1 ? "train" : group == 2 ? "validation" : "test";
        String label = group == 1 ? "T" : group == 2 ? "V" : "E";
        String key = String.format(java.util.Locale.ROOT, "%s_%03d", label, slot);
        int policyUpdate = group == 1 ? (slot - 1) / 32 : group == 2 ? (slot - 1) / 16 : -1;
        String experiment = group == 1 ? "Topology_B_Train" + (policyUpdate + 1)
            : group == 2 ? "Topology_B_Validate80" : "Topology_B_Test50";
''' + source[end:]
    start = source.index('            require(config.get("experiment")')
    end = source.index('            return new TopologyWitness', start)
    source = source[:start] + '''            require(config.get("experiment").getAsString().equals(experiment)
                    && config.get("layout_id").getAsString().equals("loop_B")
                    && config.get("phase").getAsString().equals(phase)
                    && config.get("case").getAsString().equals(key)
                    && config.get("policy_update").getAsBigDecimal().intValueExact() == policyUpdate
                    && config.get("native_case_code").getAsBigDecimal().intValueExact() == number,
                    "Configuration selects the wrong fine-tuning slot");
''' + source[end:]
    source = replace(source, '    private final String configSha;',
                     '    private final String configSha;\n    private final int policyUpdate;\n'
                     '    private final String checkpointSha;')
    source = replace(source, 'require("topology_frozen_policy_trial".equals(text("purpose")), "Wrong witness purpose");',
        '''require("topology_b_finetune".equals(text("purpose")), "Wrong B fine-tuning purpose");
        Path activationPath = Path.of(text("activation_path"));
        var activation = JsonParser.parseString(Files.readString(activationPath)).getAsJsonObject();
        policyUpdate = activation.get("policy_update").getAsBigDecimal().intValueExact();
        checkpointSha = activation.get("checkpoint_sha256").getAsString();
        int expectedUpdate = config.get("policy_update").getAsBigDecimal().intValueExact();
        require(activation.get("run_index").getAsBigDecimal().longValueExact()
                    == config.get("run_index").getAsBigDecimal().longValueExact()
                && configSha.equals(activation.get("config_sha256").getAsString())
                && checkpointSha.matches("[0-9a-f]{64}") && policyUpdate >= 0 && policyUpdate <= 4
                && (policyUpdate == expectedUpdate || (expectedUpdate == -1 && "test".equals(text("phase")))),
                "Model activation differs from the frozen B shift");''')
    source = replace(source, 'dispatcher = new TopologyPolicyDispatch(main, topology, configSha, text("checkpoint_sha256"),',
                     'dispatcher = new TopologyPolicyDispatch(main, topology, configSha, checkpointSha,')
    source = replace(source, '"topology-stats50-v1", main.experimentSeed,',
                     '"topology-b-finetune-v1", main.experimentSeed,')
    source = replace(source, '    public void finish(double cumulativeWait) {', '''    private String writeTrainingTerminal(double wait) {
        var observation = main.buildPhysicalObservation(null);
        require("dispatch.physical.v3".equals(observation.getSchemaId()) && observation.getValuesCount() == 1519
                && observation.getValues(17) == 1, "Invalid physical terminal observation");
        for (int i = 19; i < 23; i++) require(observation.getValues(i) == 0, "Terminal action mask is nonempty");
        var values = new com.google.gson.JsonArray();
        for (float value : observation.getValuesList()) {
            require(Float.isFinite(value), "Nonfinite physical terminal observation");
            values.add(value);
        }
        var payload = fields("run_index", main.experimentRunIndex, "environment_seed", main.experimentSeed,
                "recording_id", main.episodeId, "layout_id", topology.layoutId(), "config_sha256", configSha,
                "checkpoint_sha256", checkpointSha, "policy_update", policyUpdate, "sim_time_seconds", now(),
                "decision_sequence", decisionSequence, "dispatchable_mask", main.fleetResilience.dispatchableMask,
                "in_system_mask", main.fleetResilience.inSystemMask, "cumulative_wait_seconds", wait,
                "ore_tons", main.crusherOreTons, "metal_tons", main.crusherMetalTons,
                "waste_tons", main.totalWasteMoved, "obs", values);
        Path folder = Path.of(text("python_output_directory"));
        Path target = folder.resolve("terminal_observation.json");
        try {
            require(!Files.exists(target), "Terminal evidence already exists");
            byte[] data = (new Gson().toJson(payload) + "\\n").getBytes(java.nio.charset.StandardCharsets.UTF_8);
            Path pending = Files.createTempFile(folder, ".terminal-", ".tmp");
            try {
                Files.write(pending, data);
                Files.move(pending, target, java.nio.file.StandardCopyOption.ATOMIC_MOVE);
            } finally {
                Files.deleteIfExists(pending);
            }
            return hash(data);
        } catch (IOException error) {
            throw new UncheckedIOException("Cannot publish training terminal observation", error);
        }
    }

    public void finish(double cumulativeWait) {''')
    return replace(source, '            request(cumulativeWait, true);',
        '''            request(cumulativeWait, true);
            String terminalSha = "train".equals(text("phase")) ? writeTrainingTerminal(cumulativeWait) : null;
            main.boundaryEventRecorder.bindPolicyUpdate(policyUpdate, terminalSha);''')


def build(base, destination):
    raw = base.read_bytes()
    if sha256(raw).hexdigest() != BASE_SHA:
        raise ValueError('Expected the sealed A/B 50-shift model')
    source = raw.decode()
    model = ET.fromstring(source).find('Model')
    max_id = max(int(node.text) for node in model.iter('Id') if node.text and node.text.isdecimal())
    original_a = named_block(source, 'ParamVariationExperiment', 'Topology_Frozen_Stats_A50')
    original_b = named_block(source, 'ParamVariationExperiment', 'Topology_Frozen_Stats_B50')
    source = replace(source, original_a, '')
    source = replace(source, original_b, '\n'.join(experiment(original_b, *row, max_id+i)
                                                  for i, row in enumerate(EXPERIMENTS, 1)))
    block = named_block(source, 'JavaClass', 'TopologyWitness')
    source = replace(source, block, witness(block))
    block = named_block(source, 'JavaClass', 'AcceptanceEventRecorder')
    recorder = replace(block, 'this.compactFifo = "topology-stats50-v1".equals(instrumentationRevision);',
                       'this.compactFifo = "topology-stats50-v1".equals(instrumentationRevision)\n'
                       '                || "topology-b-finetune-v1".equals(instrumentationRevision);')
    recorder = replace(recorder, '    public synchronized void finishPolicyTrial(',
        '''    public synchronized void bindPolicyUpdate(int update, String terminalSha) {
        if (closed || manifest.containsKey("native_policy_update"))
            throw new IllegalStateException("Duplicate or closed policy binding");
        manifest.put("native_policy_update", update);
        if (terminalSha != null) manifest.put("terminal_observation_sha256", terminalSha);
    }

    public synchronized void finishPolicyTrial(''')
    source = replace(source, block, recorder)
    source = replace(source, f'<Name><![CDATA[{BASE_MODEL}]]>', f'<Name><![CDATA[{MODEL}]]>')
    model = ET.fromstring(source).find('Model')
    for name, first, count in EXPERIMENTS:
        exp, = [e for e in model.findall('Experiments/ParamVariationExperiment') if e.findtext('Name') == name]
        if not (exp.findtext('NumberOfRuns') == str(count) and exp.findtext('AllowParallelEvaluations') == 'false'
                and exp.findtext('ReplicationsProperties/UseReplication') == 'false'
                and exp.findtext('FreeformParamValue/Expression/Code') == f'{first} + index'
                and exp.findtext('ModelTimeProperties/FinalTime') == '28801.0'):
            raise ValueError(f'Invalid B shift mapping for {name}')
    with destination.open('xb') as stream:
        stream.write(source.encode())
    print(sha256(destination.read_bytes()).hexdigest(), destination)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('base', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    build(args.base, args.destination)
