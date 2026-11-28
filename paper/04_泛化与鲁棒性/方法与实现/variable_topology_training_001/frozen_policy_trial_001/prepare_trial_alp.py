"""Build a frozen-Actor trial sibling of the sealed eight-hour physical witness."""

import argparse
from hashlib import sha256
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
TOPOLOGY = HERE.parent
sys.path.insert(0, str(TOPOLOGY / 'physical_trip_001'))

from prepare_trip_alp import named_block, replace

BASE_MODEL = 'DRL-PPO-baseline-topologydemo-full-shift-witness'
BASE_SHA256 = 'fa209b81f4e9ecc5c147d2e0776327bbefa9cf0a8e04d4352a5b73ca29bce7de'
MODEL = 'DRL-PPO-baseline-topologydemo-frozen-policy-trial'
CASES = ('A', 'B')


def patch_obs_init(source):
    source = replace(source,
        'if (trucks.size()!=9 || loadSites.size()!=4 || dumpSites.size()!=2 || segmentMap.size()!=16)',
        'if (trucks.size()!=9 || loadSites.size()!=4 || dumpSites.size()!=2\n'
        '    || (topologyWitness == null && segmentMap.size()!=16))')
    return replace(source, '''for (String name : obsNodeNames) if(getRoadPointNode(name)==null)
    throw new IllegalStateException("Missing observation node: "+name);
obsSegmentKeys.clear();
obsSegmentKeys.addAll(segmentMap.keySet());
java.util.Collections.sort(obsSegmentKeys);''', '''obsSegmentKeys.clear();
if (topologyWitness != null) {
    obsSegmentKeys.addAll(topologyWitness.observationRoadKeys());
} else {
    for (String name : obsNodeNames) if(getRoadPointNode(name)==null)
        throw new IllegalStateException("Missing observation node: "+name);
    obsSegmentKeys.addAll(segmentMap.keySet());
    java.util.Collections.sort(obsSegmentKeys);
}''')


def patch_obs_collect(source):
    return replace(source, '''    PathSegment s=segmentMap.get(obsSegmentKeys.get(i));
    String p="road."+i+".";''', '''    String p="road."+i+".";
    if (i >= obsSegmentKeys.size()) {
        for (String field : new String[]{"from", "to", "length", "limit", "closed"})
            w.add(p+field, 0);
        continue;
    }
    PathSegment s=segmentMap.get(obsSegmentKeys.get(i));
    if (s == null) throw new IllegalStateException("Missing instantiated observation road");''')


def patch_witness(source):
    old_configs = TOPOLOGY / 'feasibility_001' / 'full_shift_001' / 'configs'
    source = replace(source, str(old_configs), str(HERE / 'configs'))
    source = replace(source, 'topology_full_shift_feasibility', 'topology_frozen_policy_trial')
    source = replace(source, 'with a frozen local dispatcher.', 'with the frozen Actor.')
    source = replace(source, 'private TopologyFeasibilityDispatch dispatcher;',
                     'private TopologyPolicyDispatch dispatcher;')
    source = replace(source, 'dispatcher = new TopologyFeasibilityDispatch(main);',
                     'dispatcher = new TopologyPolicyDispatch(main, topology, configSha, text("checkpoint_sha256"));\n'
                     '        main.obsInit();')
    source = replace(source, 'truck.initialDispatchDone = !active;', 'truck.initialDispatchDone = false;')
    source = replace(source, '    public int dispatch(Truck truck) {',
                     '    public List<String> observationRoadKeys() { return dispatcher.roadKeys(); }\n\n'
                     '    public int dispatch(Truck truck) {')
    source = replace(source, 'int action = dispatcher.choose(truck);',
                     'int action = dispatcher.choose(truck, decisionSequence);')
    source = replace(source, 'Frozen dispatcher proposed an invalid load site',
                     'Frozen Actor proposed an invalid load site')
    source = replace(source, 'dispatcher.committed(truck);', 'dispatcher.committed(truck, decisionSequence);')
    source = replace(source, 'decisionSequence, truck.targetLoadSiteId);',
                     'decisionSequence, truck.targetLoadSiteId, truck.truckId);')
    source = replace(source, '        request(cumulativeWait, true);',
                     '        dispatcher.requireComplete(decisionSequence);\n'
                     '        request(cumulativeWait, true);')
    source = replace(source, 'main.boundaryEventRecorder.finishWitness(',
                     'main.boundaryEventRecorder.finishPolicyTrial(')
    return replace(source, 'crusher.auditBufferMetalTons(), approvedArrivals);',
                   'crusher.auditBufferMetalTons(), approvedArrivals, decisionSequence, dispatcher.checkpointSha());')


def patch_recorder(source):
    source = replace(source, '    public void cutoffExcluded(double now, String operation, int truck) {',
        '''    public void actionApplied(double now, String requestId, long decisionSequence, int action, int truck) {
        emit(now, "ACTION_APPLIED", fields("request_id", requestId, "decision_sequence", decisionSequence,
            "action", action, "truck_id", truck));
    }

    public void cutoffExcluded(double now, String operation, int truck) {''')
    return replace(source, '    public synchronized void closeIncomplete() {',
        '''    public synchronized void finishPolicyTrial(double now, double waiting, double oreTons,
            double metalTons, double wasteTons, double fifoTons, double fifoMetal,
            Map<String, Integer> nativeRoadArrivals, long decisions, String checkpointSha) {
        manifest.put("native_policy_decisions", decisions);
        manifest.put("native_checkpoint_sha256", checkpointSha);
        finishWitness(now, waiting, oreTons, metalTons, wasteTons, fifoTons, fifoMetal, nativeRoadArrivals);
    }

    public synchronized void closeIncomplete() {''')


def patch_experiment(source, case):
    source = replace(source, f'Topology_Full_Shift_{case}', f'Topology_Frozen_Policy_{case}')
    return replace(source, f'Topology complete 8-hour witness {case}', f'Topology frozen Actor 8-hour trial {case}')


def prepare_model(base, destination):
    raw = base.read_bytes()
    if sha256(raw).hexdigest() != BASE_SHA256:
        raise ValueError('The witness ALP must match the sealed and verified source bytes')
    source = raw.decode('utf-8')
    model = ET.fromstring(source).find('Model')
    if model.findtext('Name') != BASE_MODEL:
        raise ValueError('The source must be the full-shift witness ALP')
    java_id = max(int(item.text) for item in model.iter('Id') if item.text and item.text.isdecimal()) + 1

    main = named_block(source, 'ActiveObjectClass', 'Main')
    for function, patch in (('obsInit', patch_obs_init), ('collectPhysicalObservation', patch_obs_collect)):
        block = named_block(main, 'Function', function)
        updated = replace(main, block, patch(block))
        source = replace(source, main, updated)
        main = updated

    for name, patch in (('TopologyWitness', patch_witness), ('AcceptanceEventRecorder', patch_recorder)):
        block = named_block(source, 'JavaClass', name)
        source = replace(source, block, patch(block))

    for case in CASES:
        block = named_block(source, 'ParamVariationExperiment', f'Topology_Full_Shift_{case}')
        source = replace(source, block, patch_experiment(block, case))

    old_dispatch = named_block(source, 'JavaClass', 'TopologyFeasibilityDispatch')
    dispatch = (HERE / 'TopologyPolicyDispatch.java').read_text(encoding='utf-8')
    source = replace(source, old_dispatch,
        f'\t\t<JavaClass>\n\t\t\t<Id>{java_id}</Id>\n'
        '\t\t\t<Name><![CDATA[TopologyPolicyDispatch]]></Name>\n'
        f'\t\t\t<Text><![CDATA[{dispatch.removeprefix("package fusion;").lstrip()}]]></Text>\n'
        '\t\t</JavaClass>')
    source = replace(source, f'<Name><![CDATA[{BASE_MODEL}]]></Name>', f'<Name><![CDATA[{MODEL}]]></Name>')

    model = ET.fromstring(source).find('Model')
    experiments = model.findall('./Experiments/ParamVariationExperiment')
    for number, case in enumerate(CASES, 1):
        experiment, = (value for value in experiments if value.findtext('Name') == f'Topology_Frozen_Policy_{case}')
        if experiment.findtext('NumberOfRuns') != '1' or experiment.findtext('FreeformParamValue/Expression/Code') != str(number):
            raise ValueError(f'Frozen Actor case {case} is not one native eight-hour run')
    with destination.open('x', encoding='utf-8') as stream:
        stream.write(source)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('witness_alp', type=Path)
    parser.add_argument('destination_alp', type=Path)
    args = parser.parse_args()
    prepare_model(args.witness_alp, args.destination_alp)


if __name__ == '__main__':
    main()
