"""Render the baseline2 helper using its sealed six-truck source and handshake."""
import json
import re
from pathlib import Path

PLAN = Path(__file__).resolve().parent
TEMPLATE = PLAN.parent / 'fleet_outage_003/prepared_001/Baseline1CampaignPlan.java'
FLEET_FIELDS = ('initial_mask', 'fault_truck_id', 'fault_trigger_seconds', 'ore_target', 'waste_target', 'waste_stop')
MASTER_FIELDS = ('ordinal', 'actor_seed', 'group', 'role', 'source_recording_id', 'recording_id', 'run_index',
                 'experiment_seed', 'actor_sampling_seed', 'event_directory', 'event_bundle', 'python_output',
                 'target_training_update', 'checkpoint_update', *FLEET_FIELDS, 'config_sha256')
SCENE_FIELDS = ('run_index','scenario_id', *(f'load{s}_is_ore' for s in range(4)),
                *(f'load{s}_grade' for s in range(4)), *(f'load{s}_grade_half_range' for s in range(4)),
                'road_6_speed_limit_mps')


def render(suite, scenario_ref, launcher_sha):
    source = TEMPLATE.read_text()
    def replace(old, new):
        nonlocal source
        if source.count(old) != 1:
            raise ValueError('unexpected Java template anchor: ' + old)
        source = source.replace(old, new)
    def constant(name, value):
        old, = re.findall(r'private static final String ' + name + r' = "[^"]*";', source)
        replace(old, 'private static final String ' + name + ' = ' + json.dumps(value) + ';')

    for name, value in dict(MASTER_PLAN_PATH=suite['master_plan']['path'],MASTER_PLAN_SHA256=suite['master_plan']['sha256'],
                            CONTROL_DIRECTORY=suite['control_directory'],CAMPAIGN_ID=suite['campaign_id'],
                            SCENARIO_TABLE_PATH=scenario_ref['path'],SCENARIO_TABLE_SHA256=scenario_ref['sha256']).items():
        constant(name,value)
    replace('private static final int JOB_COUNT = 10;', f'private static final int JOB_COUNT = {len(suite["jobs"])};')
    replace('"target_training_update", "checkpoint_update"', ', '.join(json.dumps(k) for k in MASTER_FIELDS[12:]))
    replace('require(values[2].equals("BASELINE") && values[3].equals("final"),',
            'require(values[2].equals("BASELINE") && (values[3].equals("train") || values[3].equals("final")),')
    # A pair's repeated seeds and the exact paired roles are checked on Python admission.
    replace('require(environmentSeeds.add(row.seed), "environment seed repeats a fleet outage slot");','environmentSeeds.add(row.seed);')
    replace('require(samplingSeeds.add(positive(values[8], "actor sampling seed")),\n                        "Actor sampling seed repeats a fleet outage slot");',
            'samplingSeeds.add(positive(values[8], "actor sampling seed"));')
    source = source.replace(str(TEMPLATE.parent.parent / 'launch.py'), str(PLAN / 'launch.py')).replace(
        str(TEMPLATE.parent / 'suite.json'), suite['suite_path'])
    replace('50d5e9ce9e22a20833f61467ebee626552bfd8db0d15e53991f78b5eb10cacd5', launcher_sha)
    methods = ['public static int fleetDisabledTruckMask(int index) { return 511 ^ Integer.parseInt(row(index).cells[14]); }']
    for column, name, typ in [(15,'faultTruckId','int'), (16,'faultTriggerSeconds','double'),
                              (17,'oreTarget','double'),(18,'wasteTarget','double'),(19,'wasteStop','double')]:
        parser = 'Integer.parseInt' if typ=='int' else 'Double.parseDouble'
        methods.append(f'public static {typ} {name}(int index) {{ return {parser}(row(index).cells[{column}]); }}')
    replace('public static int fleetDisabledTruckMask(int index) { row(index); return 292; }','\n    '.join(methods))
    return source.encode()
