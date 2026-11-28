"""Build the frozen Actor's B experiment with targets locked to the opening fleet."""

import argparse
from hashlib import sha256
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'physical_trip_001'))
from prepare_trip_alp import named_block, replace

BASE_SHA = '8757798c9132c89165746a8f08ee5a4d551745fd1a0b8a68ff363f1be76f0cd8'
BASE_MODEL = 'DRL-PPO-baseline-topologydemo-frozen-policy-trial-b-repair'
MODEL = 'DRL-PPO-baseline-topologydemo-frozen-policy-scaled'
EXPERIMENT = 'Topology_Frozen_Policy_B_Scaled'


def scaled_witness(source):
    source = replace(source, str(HERE.parent / 'repair_001' / 'configs'), str(HERE / 'configs'))
    assign_shift = 'main.targetOreTonsShift = task.get("ore_target_tons").getAsDouble();'
    source = replace(source, assign_shift, assign_shift + '\n        main.targetOreTons2h = main.targetOreTonsShift / 4.0;')
    source = replace(source, '        instantiateFleet();',
                     '        instantiateFleet();\n        verifyOpeningTargets(task);')
    source = replace(source, '    private void instantiateFleet() {', '''    private void verifyOpeningTargets(JsonObject task) {
        double referenceCapacity = 0.0;
        double activeCapacity = 0.0;
        for (Truck truck : main.trucks) {
            referenceCapacity += truck.capacity;
            if (!main.isFleetTruckDisabled(truck.truckId)) activeCapacity += truck.capacity;
        }
        double tolerance = 1e-7;
        require(main.trucks.size() == 9 && Math.abs(referenceCapacity - 1295.1) <= tolerance,
                "Reference truck roster differs from the trained nine-truck fleet");
        double alpha = activeCapacity / referenceCapacity;
        double wasteUpper = main.getWasteTaskUpperTons();
        require(Math.abs(main.targetOreTonsShift - 16000.0 * alpha) <= tolerance
                && Math.abs(main.targetOreTons2h - 4000.0 * alpha) <= tolerance
                && Math.abs(main.targetWasteTonsShift - 4000.0 * alpha) <= tolerance
                && Math.abs(main.wasteDispatchStopTonsShift - 6000.0 * alpha) <= tolerance
                && Math.abs(wasteUpper - (6000.0 * alpha + 193.0)) <= tolerance
                && Math.abs(wasteUpper - task.get("waste_upper_tons").getAsDouble()) <= tolerance,
                "Configured production targets differ from the opening fleet capacity");
    }

    private void instantiateFleet() {''')
    for field in ('main.targetOreTonsShift', 'main.targetWasteTonsShift',
                  'main.wasteDispatchStopTonsShift', 'main.getWasteTaskUpperTons()'):
        source = replace(source, f'Math.round({field})', field)
    return replace(source, '"ore_target_tons", main.targetOreTonsShift,',
                   '"ore_target_tons", main.targetOreTonsShift,\n                        "ore_target_tons_2h", main.targetOreTons2h,')


def build(base, destination):
    raw = base.read_bytes()
    if sha256(raw).hexdigest() != BASE_SHA:
        raise ValueError('Expected the sealed native B repair ALP')
    source = raw.decode('utf-8')
    witness = named_block(source, 'JavaClass', 'TopologyWitness')
    source = replace(source, witness, scaled_witness(witness))
    source = replace(source, f'<Name><![CDATA[{BASE_MODEL}]]>', f'<Name><![CDATA[{MODEL}]]>')
    source = replace(source, '<Name><![CDATA[Topology_Frozen_Policy_B_Repair]]>',
                     f'<Name><![CDATA[{EXPERIMENT}]]>')
    source = replace(source, 'Topology frozen Actor 8-hour repair B', 'Topology frozen Actor capacity-scaled B')
    model = ET.fromstring(source).find('Model')
    experiment, = (item for item in model.findall('Experiments/ParamVariationExperiment')
                   if item.findtext('Name') == EXPERIMENT)
    if (model.findtext('Name') != MODEL or experiment.findtext('NumberOfRuns') != '1'
            or experiment.findtext('FreeformParamValue/Expression/Code') != '2'
            or experiment.findtext('ModelTimeProperties/FinalTime') != '28801.0'):
        raise ValueError('Expected one full-shift scaled B experiment')
    with destination.open('xb') as stream:
        stream.write(source.encode('utf-8'))
    print(sha256(destination.read_bytes()).hexdigest(), destination)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('base', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    build(args.base, args.destination)
