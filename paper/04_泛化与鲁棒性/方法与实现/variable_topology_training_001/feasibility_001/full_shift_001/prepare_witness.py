"""Generate a sibling witness ALP and freeze two single-shift launch identities."""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
TOPOLOGY = HERE.parents[1]
PROJECT = TOPOLOGY.parents[3]
sys.path[:0] = [str(TOPOLOGY), str(TOPOLOGY / 'physical_trip_001'), str(HERE.parent)]

from admit_layouts import canonical
from compile_layouts import compile_layout, load_layouts
from prepare_trip_alp import named_block, replace

MODEL = 'DRL-PPO-baseline-topologydemo-full-shift-witness'
BASE_MODEL = 'DRL-PPO-baseline-topologydemo-physical-trip'
CASE_IDS = {
    'A': ('tree_A', 10054031, 280117723616006),
    'B': ('loop_B', 10054032, 177596069752717),
}
RECORDING_PREFIX = 'site-layout-feasibility-witness-20261127-001'
BINDER_SHA256 = '8bab10fdb7562e355023d2815f203785f50bca00e5a521f74889acd53352bf75'


def java_witness_source():
    source = (HERE / 'TopologyWitness.java').read_text()
    binder_bytes = (TOPOLOGY / 'physical_trip_001' / 'TopologyTripSmoke.java').read_bytes()
    if sha256(binder_bytes).hexdigest() != BINDER_SHA256:
        raise ValueError('The previously verified native binder has changed')
    binder = binder_bytes.decode()
    constructor_methods = binder[binder.index('    private void installRoadsAndRoutes()'):
                                 binder.index('    private void instantiateFleet()')]
    attach = binder[binder.index('    private void attach('):binder.index('    public int dispatch(')]
    return replace(replace(source, '@CONFIG_DIRECTORY@', str(HERE / 'configs')),
                   '    @VERIFIED_BINDER@', (constructor_methods + attach).rstrip())


def experiment(source, item_id, case, number):
    source = re.sub(r'<Id>\d+</Id>', f'<Id>{item_id}</Id>', source, count=1)
    source = replace(source, 'Topology_Physical_Trip_Check', f'Topology_Full_Shift_{case}')
    source = replace(source, '<NumberOfRuns>2</NumberOfRuns>', '<NumberOfRuns>1</NumberOfRuns>')
    source = replace(source, 'TopologyTripSmoke.caseNumber((int) index)', str(number))
    source = replace(source, 'Topology production single-trip check', f'Topology complete 8-hour witness {case}')
    source = replace(source, '<FinalTime><![CDATA[2400.0]]>', '<FinalTime><![CDATA[28801.0]]>')
    return replace(source, '<FinalDate><![CDATA[1776775200000]]>', '<FinalDate><![CDATA[1776801601000]]>')


def prepare_model(base, destination):
    source = base.read_text(encoding='utf-8')
    model = ET.fromstring(source).find('Model')
    if model.findtext('Name') != BASE_MODEL:
        raise ValueError('Base model must be the native single-trip ALP')
    next_id = max(int(item.text) for item in model.iter('Id') if item.text and item.text.isdecimal()) + 1
    main = named_block(source, 'ActiveObjectClass', 'Main')
    main_witness = main.replace('topologyTripCase', 'topologyWitnessCase').replace('topologyTripSmoke', 'topologyWitness')
    main_witness = main_witness.replace('TopologyTripSmoke', 'TopologyWitness')
    main_witness = replace(main_witness, '{ topologyWitness.sample(); return; }', 'return;')
    terminal = named_block(main_witness, 'Function', 'finishEpisodeAndNotify')
    terminal_witness = replace(terminal, 'if (currentDispatchPolicy != 3 && !offlineNativeCollection) {',
                               'if (topologyWitness != null) {\n'
                               '    topologyWitness.finish(cumulativeWait);\n'
                               '    return;\n'
                               '}\n'
                               'if (currentDispatchPolicy != 3 && !offlineNativeCollection) {')
    source = replace(source, main, replace(main_witness, terminal, terminal_witness))

    truck = named_block(source, 'ActiveObjectClass', 'Truck')
    truck_witness = truck.replace('topologyTripSmoke', 'topologyWitness')
    truck_witness = replace(truck_witness,
                            '\nif (main.topologyWitness != null) main.topologyWitness.finish(this);', '')
    truck_witness = replace(truck_witness, '<Condition><![CDATA[main.topologyWitness == null]]></Condition>',
                            '<Condition><![CDATA[true]]></Condition>')
    truck_witness = replace(truck_witness, 'main.auditMaterialBalances("RESERVE_LOAD truck=" + truckId);',
                            'main.auditMaterialBalances("RESERVE_LOAD truck=" + truckId);\n'
                            '    if (main.topologyWitness != null) main.topologyWitness.applied(this);')
    source = replace(source, truck, truck_witness)

    recorder = named_block(source, 'JavaClass', 'AcceptanceEventRecorder')
    recorder_witness = replace(recorder, '    public synchronized void closeIncomplete() {',
        '''    public synchronized void finishWitness(double now, double waiting, double oreTons,
            double metalTons, double wasteTons, double fifoTons, double fifoMetal,
            Map<String, Integer> nativeRoadArrivals) {
        manifest.put("fallback_dispatches", 0);
        manifest.put("invalid_dispatches", 0);
        manifest.put("native_road_arrivals", new LinkedHashMap<>(nativeRoadArrivals));
        finish(now, waiting, oreTons, metalTons, wasteTons, fifoTons, fifoMetal);
    }

    public synchronized void closeIncomplete() {''')
    source = replace(source, recorder, recorder_witness)

    smoke_experiment = named_block(source, 'ParamVariationExperiment', 'Topology_Physical_Trip_Check')
    source = replace(source, smoke_experiment, '\n'.join(
        experiment(smoke_experiment, next_id + number - 1, case, number)
        for number, case in enumerate(CASE_IDS, 1)))
    smoke_helper = named_block(source, 'JavaClass', 'TopologyTripSmoke')
    witness_sources = {'TopologyWitness': java_witness_source(),
                       'TopologyFeasibilityDispatch': (HERE / 'TopologyFeasibilityDispatch.java').read_text()}
    source = replace(source, smoke_helper, '\n'.join(
        f'\t\t<JavaClass>\n\t\t\t<Id>{number}</Id>\n'
        f'\t\t\t<Name><![CDATA[{name}]]></Name>\n'
        f'\t\t\t<Text><![CDATA[{java.removeprefix("package fusion;").lstrip()}]]></Text>\n\t\t</JavaClass>'
        for number, (name, java) in enumerate(witness_sources.items(), next_id + 2)))
    source = replace(source, f'<Name><![CDATA[{BASE_MODEL}]]></Name>', f'<Name><![CDATA[{MODEL}]]></Name>')
    # AnyLogic can add this parameter-name entry when saving the base model.
    source = source.replace('<ParameterName><![CDATA[topologyTripCase]]></ParameterName>',
                            '<ParameterName><![CDATA[topologyWitnessCase]]></ParameterName>')
    if 'topologyTripSmoke' in source or 'topologyTripCase' in source:
        raise ValueError('Stale smoke reference in witness model')
    experiments = ET.fromstring(source).findall('./Model/Experiments/ParamVariationExperiment')
    for number, case in enumerate(CASE_IDS, 1):
        match, = (item for item in experiments if item.findtext('Name') == f'Topology_Full_Shift_{case}')
        assert match.findtext('NumberOfRuns') == '1'
        assert match.findtext('FreeformParamValue/Expression/Code') == str(number)
    with destination.open('x', encoding='utf-8') as stream:
        stream.write(source)


def expected_scene(layout, scene, edges, task, model_sha):
    settings = dict(line.split('=', 1) for line in scene.decode().splitlines())
    nodes = []
    for key, value in settings.items():
        if key.startswith('node.'):
            name, x, y = value.split(',')
            nodes.append([int(key[5:]), name, round(1000 * float(x)), round(1000 * float(y))])
    fleet = []
    for truck in range(9):
        active, category, capacity, start = settings[f'vehicle.{truck}'].split(',')
        fleet.append([truck, active == '1', int(category), round(10 * float(capacity)),
                      int(start) if active == '1' else -1])
    loaders = []
    for site in range(4):
        node, ore, grade, half_range = settings[f'load.{site}'].split(',')
        loaders.append([site, int(node), ore == '1', round(1_000_000 * float(grade)),
                        round(1_000_000 * float(half_range)), 300000, 450,
                        150 if site == 3 else 135, 20 if site == 3 else 22, 5000])
    return {
        'layout_id': layout['id'],
        'node_positions': nodes,
        'directed_roads': [[a, b, math.floor(1000 * length + 0.5), round(1000 * speed)]
                           for a, b, length, speed in edges],
        'opening_fleet': fleet,
        'load_sites': loaders,
        'dump_sites': [[0, 0, True, False, 250], [1, 1, False, True, 200]],
        'loader_slots': task['loader_slots'],
        'dump_slots': task['dump_slots'],
        'crusher_capacity_tons': task['crusher_capacity_tons'],
        'initial_inventory': {'crusher_fifo_tons': 0, 'carried_tons': 0, 'reserved_tons': 0},
        'exogenous_config': {
            'deterministic': True, 'local_dynamic_speed': False, 'endogenous_congestion': False,
            'fault_truck_id': -1, 'fault_trigger_seconds': -1, 'empty_speed_mm_s': 6500,
            'loaded_speed_mm_s': 5000, 'ore_target_tons': task['ore_target_tons'],
            'waste_target_tons': task['waste_target_tons'],
            'waste_dispatch_stop_tons': task['waste_dispatch_stop_tons'],
            'waste_upper_tons': task['waste_upper_tons'], 'enable_waste_task_limit': True,
        },
        'model_sha256': model_sha,
    }


def write_fresh(path, contents):
    if path.exists():
        if path.read_bytes() != contents:
            raise ValueError(f'Frozen configuration already differs at {path}')
        return
    with path.open('xb') as stream:
        stream.write(contents)


def prepare_configs(model_path):
    model_sha = sha256(model_path.read_bytes()).hexdigest()
    task = json.loads((HERE.parent / 'task.json').read_bytes())
    layouts = {value['id']: value for value in load_layouts()}
    configs = HERE / 'configs'
    configs.mkdir(exist_ok=True)
    identities = {}
    for case, (layout_id, run, seed) in CASE_IDS.items():
        layout = layouts[layout_id]
        scene, edges = compile_layout(layout)
        native = expected_scene(layout, scene, edges, task, model_sha)
        layout_path = TOPOLOGY / 'compiled' / f'{layout_id}.properties'
        if layout_path.read_bytes() != scene:
            raise ValueError(f'Compiled layout file {layout_path} does not match input')
        output = PROJECT / 'runs' / f'{RECORDING_PREFIX}-{case}'
        output.mkdir(exist_ok=True)
        config = dict(purpose='topology_full_shift_feasibility', run_index=run, environment_seed=seed,
                      actor_sampling_seed=None, recording_id=f'{RECORDING_PREFIX}-{run}',
                      experiment=f'Topology_Full_Shift_{case}', layout_id=layout_id,
                      layout_path=str(layout_path), layout_sha256=sha256(scene).hexdigest(),
                      task=task, task_sha256=sha256(canonical(task)).hexdigest(), model_path=str(model_path),
                      model_sha256=model_sha, expected_runtime_scene=native,
                      expected_runtime_scene_sha256=sha256(canonical(native)).hexdigest(),
                      event_directory=str(output / 'anylogic_events'),
                      python_output_directory=str(output / 'python'))
        raw = canonical(config)
        digest = sha256(raw).hexdigest()
        write_fresh(configs / f'{case}.json', raw)
        write_fresh(configs / f'{case}.sha256', (digest + '\n').encode())
        identities[case] = {key: config[key] for key in ('experiment', 'run_index', 'environment_seed',
                             'recording_id', 'event_directory', 'python_output_directory')}
        identities[case]['config_sha256'] = digest
        identities[case]['config_path'] = str(configs / f'{case}.json')
    print(json.dumps(identities, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    operations = parser.add_subparsers(dest='mode', required=True)
    model = operations.add_parser('model')
    model.add_argument('source', type=Path)
    model.add_argument('destination', type=Path)
    config = operations.add_parser('configs')
    config.add_argument('installed_model', type=Path)
    args = parser.parse_args()
    if args.mode == 'model':
        prepare_model(args.source, args.destination)
    else:
        prepare_configs(args.installed_model.resolve())


if __name__ == '__main__':
    main()
