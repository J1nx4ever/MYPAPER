"""Build a separate nine-truck A graph, frozen Full s02/u32 high-grade-loader outage experiment."""
import argparse
from hashlib import sha256
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'physical_trip_001'))
from prepare_trip_alp import named_block, replace

BASE_SHA = '60bedf6c07e6f75058597cfa928faf31e289a47740394d34d62a3657c610211c'
BASE_MODEL = 'DRL-PPO-baseline-topologydemo-b-finetune'
MODEL = 'DRL-PPO-baseline-topologydemo-a-full-u32-high-grade-outage'
EXPERIMENT = 'Topology_A_Full_HighGradeOutage'
CHECKPOINT = '67b7b2c7104079836ccef8f5747e7aebb4e9e5a92dcaac67c94e42412f104f41'


def bridge(code, left, right, replacement):
    if code.count(left) != 1 or right not in code:
        raise ValueError('Base Java source split markers differ')
    a = code.index(left)
    b = code.index(right, a)
    return code[:a] + replacement + code[b:]


def patch_witness(code, count):
    code = replace(code, str(HERE.parent / 'b_finetune_001' / 'configs'), str(HERE / 'configs'))
    code = bridge(code, '        int group = number / 1000', '        try {', f"""
        int slot = number - 4000;
        require(slot >= 1 && slot <= {count}, "Invalid high-grade outage slot");
        String key = String.format(java.util.Locale.ROOT, "O_%03d", slot);
        String phase = "test", experiment = "{EXPERIMENT}";
        int policyUpdate = 32;
""")
    code = bridge(code, '        require("topology_b_finetune".equals', '        modelSha = hash', f'''
        require("topology_a_full_high_grade_outage".equals(text("purpose")), "Wrong intervention purpose");
        policyUpdate = config.get("policy_update").getAsBigDecimal().intValueExact();
        checkpointSha = text("checkpoint_sha256");
        require(policyUpdate == 32 && checkpointSha.equals("{CHECKPOINT}"), "Wrong frozen Full s02/u32");
        readOutage();
''')
    code = replace(code, 'equals("loop_B")', 'equals("tree_A")')
    code = replace(code, 'double alpha = activeCapacity / referenceCapacity;',
                   'require(Math.abs(activeCapacity - 1295.1) <= tolerance, "All nine A trucks must be active");\n'
                   '        double alpha = activeCapacity / referenceCapacity;')
    code = replace(code, '"enable_waste_task_limit", main.enableWasteTaskLimit),',
                   '"enable_waste_task_limit", main.enableWasteTaskLimit, "load_site_outage", physicalOutagePlan()),')
    code = replace(code, '"topology-b-finetune-v1"', '"topology-high-grade-outage-v1"')
    code = replace(code, '        for (Truck truck : main.trucks) {\n            if (!main.isFleetTruckDisabled(truck.truckId))',
                   '        main.evHighGradeStop.restart(stopSec, SECOND);\n'
                   '        main.evHighGradeResume.restart(resumeSec, SECOND);\n'
                   '        for (Truck truck : main.trucks) {\n            if (!main.isFleetTruckDisabled(truck.truckId))')
    code = replace(code, '            int action = dispatcher.choose(truck, decisionSequence);',
                   '            syncOutage();\n            int action = dispatcher.choose(truck, decisionSequence);')
    code = bridge(code, '    private String writeTrainingTerminal(', '    public void finish(double cumulativeWait)',
                  (HERE/'outage_methods.java.inc').read_text())
    return replace(code, '            String terminalSha = "train".equals(text("phase")) ? writeTrainingTerminal(cumulativeWait) : null;\n'
                   '            main.boundaryEventRecorder.bindPolicyUpdate(policyUpdate, terminalSha);',
                   '            main.boundaryEventRecorder.bindPolicyUpdate(policyUpdate, null);\n'
                   '            main.boundaryEventRecorder.bindOutage(publishOutage());')


def patch_recorder(code):
    code = replace(code, '"topology-b-finetune-v1".equals(instrumentationRevision)',
                   '"topology-high-grade-outage-v1".equals(instrumentationRevision)')
    return replace(code, '    public synchronized void bindPolicyUpdate(', '''
    public synchronized long outageCursor() {
        flushProcesses();
        return sequence;
    }

    public synchronized void bindOutage(String sha) {
        if (closed || manifest.containsKey("outage_evidence_sha256") || !sha.matches("[0-9a-f]{64}"))
            throw new IllegalStateException("Invalid or duplicate high-grade loader evidence SHA");
        manifest.put("outage_evidence_sha256",sha);
    }
    public synchronized void bindPolicyUpdate(''')


def build(base, target, count):
    if type(count) is not int or count != 1:
        raise ValueError('Expected exactly one A shift')
    raw = base.read_bytes()
    if sha256(raw).hexdigest() != BASE_SHA:
        raise ValueError('Expected the frozen native B fine-tuning model')
    source = raw.decode()
    model = ET.fromstring(raw).find('Model')
    ceiling = max(int(i.text) for i in model.iter('Id') if i.text and i.text.isdecimal())
    target = target.absolute()
    if target.stem != MODEL or target.suffix != '.alp':
        raise ValueError(f'Use a separate file named {MODEL}.alp')
    for name in (f'Topology_B_Train{i}' for i in range(1,5)):
        source = replace(source, named_block(source,'ParamVariationExperiment',name), '')
    source = replace(source,named_block(source,'ParamVariationExperiment','Topology_B_Validate80'),'')
    test = named_block(source,'ParamVariationExperiment','Topology_B_Test50')
    new = replace(test, 'Topology_B_Test50', EXPERIMENT, count=test.count('Topology_B_Test50'))
    new = replace(new, '<NumberOfRuns>50</NumberOfRuns>', f'<NumberOfRuns>{count}</NumberOfRuns>')
    new = replace(new, '<![CDATA[3001 + index]]>', '<![CDATA[4001 + index]]>')
    source = replace(source, test, new)
    event = named_block(source, 'Event', 'evFleetFault')
    event_id = ET.fromstring(event).findtext('Id')
    events = [event]
    for i, name in enumerate(('evHighGradeStop','evHighGradeResume'),1):
        ev = replace(event,f'<Id>{event_id}</Id>',f'<Id>{ceiling+i}</Id>')
        ev = replace(ev,'<![CDATA[evFleetFault]]>',f'<![CDATA[{name}]]>')
        ev = replace(ev,'<X>500</X><Y>-990</Y>',f'<X>{500+40*i}</X><Y>-1040</Y>')
        ev = replace(ev,'<![CDATA[fleetSync();]]>','<![CDATA[if (topologyWitness != null) topologyWitness.syncOutage();]]>')
        events.append(ev)
    source = replace(source, event, '\n'.join(events))
    for name, patch in [('TopologyWitness', lambda code: patch_witness(code,count)),('AcceptanceEventRecorder',patch_recorder)]:
        block = named_block(source,'JavaClass',name)
        source = replace(source,block,patch(block))
    source = replace(source,f'<Name><![CDATA[{BASE_MODEL}]]>',f'<Name><![CDATA[{MODEL}]]>')
    parsed = ET.fromstring(source).find('Model')
    exp, = [v for v in parsed.findall('Experiments/ParamVariationExperiment') if v.findtext('Name') == EXPERIMENT]
    if exp.findtext('AllowParallelEvaluations') != 'false' or exp.findtext('FreeformParamValue/Expression/Code') != '4001 + index':
        raise ValueError('Expected one native A slot')
    with target.open('xb') as output:
        output.write(source.encode())
    return sha256(target.read_bytes()).hexdigest()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('base', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--count', type=int, required=True)
    opts = parser.parse_args()
    print(build(opts.base,opts.destination,opts.count),opts.destination)
