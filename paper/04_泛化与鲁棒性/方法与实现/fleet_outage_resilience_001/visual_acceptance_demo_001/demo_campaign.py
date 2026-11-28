"""Freeze one visual-acceptance shift using the finished dynamic u9 policy."""
import argparse
import json
from pathlib import Path
import os
import sys
from tempfile import TemporaryDirectory

HERE = Path(__file__).resolve().parent
PLAN = HERE.parent
sys.path[:0] = [str(PLAN), str(PLAN/'replay_repair_001')]

import resilience_campaign as base
from repair_campaign import load_suite as load_repair
from resilience_java import MASTER_FIELDS, SCENE_FIELDS, render
from schedule import seed

CAMPAIGN = 'bg128-fleet-outage-visual-demo-20261124-001'
RUN = 10045001
SOURCE_RUN = 10043852
SCHEMA = 'bg128-fleet-outage-visual-demo-v1'
REPAIR = PLAN/'replay_repair_001/prepared_001/suite.json'
REPAIR_SHA = '1ddf3422e5249da3913ca4cbf6f483ba99d7dcc2f8b6970238efcff8ad635258'
IDENTITY_FIELDS = (
    'run_index', 'recording_id', 'experiment_seed', 'actor_sampling_seed', 'config_sha256',
    'config_path', 'python_output', 'event_directory',
)


def anchor():
    source = base.reference(REPAIR)
    if source['sha256'] != REPAIR_SHA:
        raise ValueError('original finished repair suite differs')
    history = load_repair(REPAIR)
    root = Path(history['output'])
    complete = base.reference(root/'suite_complete.json')
    completion = json.loads(base.load_ref(complete))
    if (completion['physical_shifts'] != 992 or completion['updates'] != dict(warmup=4, static=9, dynamic=9)
            or completion['master_plan_sha256'] != history['master_plan']['sha256']):
        raise ValueError('original experiment is not complete')
    scene, = (j for j in history['jobs'] if j['run_index'] == SOURCE_RUN)
    if (scene['arm'], scene['role'], scene['split'], scene['condition'], scene['update'],
            scene['starting_count'], scene['fault_truck_id']) != ('dynamic', 'eval', 'test', 'outage', 9, 9, 6):
        raise ValueError('the anchored outage scene differs')
    checkpoint = root/'models/dynamic/u0009/checkpoint.pt'
    pin = base.reference(checkpoint)
    manifest = base.reference(checkpoint.with_suffix('.pt.manifest.json'))
    seal = base.reference(checkpoint.with_suffix('.pt.seal.json'))
    model = json.loads(base.load_ref(manifest))
    model_seal = json.loads(base.load_ref(seal))
    if (model['update_index'] != 9 or checkpoint.stat().st_size != model['bytes']
            or model['sha256'] != pin['sha256'] or model_seal['sha256'] != pin['sha256']
            or model_seal['manifest_sha256'] != manifest['sha256']):
        raise ValueError('frozen dynamic u9 checkpoint does not match its manifest and seal')
    receipt = base.reference(Path(scene['python_output'])/'receipt.json')
    source_receipt = json.loads(base.load_ref(receipt))
    if (source_receipt['policy_version'] != model['actor_digest']
            or source_receipt['config_sha256'] != scene['config_sha256']
            or not source_receipt['outage']['window_valid']):
        raise ValueError('historical demonstration receipt differs')
    return scene, dict(source_suite=source, source_completion=complete, historical_receipt=receipt,
                       checkpoint=pin, checkpoint_manifest=manifest, checkpoint_seal=seal,
                       actor_digest=model['actor_digest'], protected_core=history['protected_core'])


def demo_job(scene, output):
    kept = (
        'starting_count', 'initial_mask', 'assignment', 'waste_site', 'grades', 'half_ranges', 'ore_target',
        'waste_target', 'waste_stop', 'fault_truck_id', 'fault_trigger_seconds', 'actor_seed', 'checkpoint_update',
        'target_training_update', 'arm', 'update', 'condition',
    )
    environment = seed(CAMPAIGN, 'env')
    recording = f'{CAMPAIGN}-{RUN}'
    events = output/'anylogic_events'/f'run_{RUN}'
    return dict({k: scene[k] for k in kept}, ordinal=0, phase='visual_demo', role='eval', split='visual_demo',
                pair_ordinal=-1, group='BASELINE', run_index=RUN, recording_id=recording,
                source_recording_id=recording, experiment_seed=environment,
                actor_sampling_seed=seed(CAMPAIGN, 'sample'), event_directory=str(events),
                event_bundle=str(events/f'dispatch-production-wait-shortfall-v2-run-{RUN}-seed-{environment}'),
                python_output=str(output/'shifts'/f'run_{RUN}'))


def config(suite, scene, job):
    return dict(schema=SCHEMA, campaign_id=CAMPAIGN, job=job, checkpoint=suite['checkpoint'],
                historical_scene=suite['historical_receipt'], protected_core=suite['protected_core'],
                expected_rpc_config=scene['expected_rpc_config'])


def render_java(suite):
    bridge = HERE/'demo_launch.py'
    old = str(PLAN/'launch.py')
    text = render(suite, suite['scenario_table'], base.reference(bridge)['sha256']).decode()
    if text.count(old) != 1:
        raise ValueError('unexpected frozen Java bridge location')
    return text.replace(old, str(bridge)).encode()


def prepare(directory):
    directory = Path(directory).resolve()
    output = base.ROOT/'runs'/CAMPAIGN
    if directory.exists() or output.exists():
        raise FileExistsError('the one-shift package or its output already exists')
    scene, pins = anchor()
    job = demo_job(scene, output)
    suite = dict(schema=SCHEMA, campaign_id=CAMPAIGN, suite_path=str(directory/'suite.json'),
                 output=str(output), control_directory=str(output/'control'), rpc_port=50051,
                 sources=[base.reference(p) for p in sorted(HERE.glob('*.py'))], **pins)
    with TemporaryDirectory(prefix='.visual-demo-', dir=directory.parent) as temp:
        stage = Path(temp)/'package'
        (stage/'configs').mkdir(parents=True)

        def publish(name, data):
            (stage/name).write_bytes(data)
            return dict(path=str(directory/name), sha256=base.digest(data))

        config_pin = publish(f'configs/run_{RUN}.json', base.encoded(config(suite, scene, job)))
        job = dict(job, config_path=config_pin['path'], config_sha256=config_pin['sha256'])
        suite['jobs'] = [job]
        java_row = dict(job, role='final')
        suite['master_plan'] = publish('master_plan.csv',
                                     base.csv_bytes(MASTER_FIELDS, [[java_row[key] for key in MASTER_FIELDS]]))
        speed = scene['expected_rpc_config']['config.runtime_scene.road.6.speedLimit_mps']
        scene_values = [RUN, f'scene-{RUN}', *[int(r != 'W') for r in job['assignment']],
                        *job['grades'], *job['half_ranges'], speed]
        suite['scenario_table'] = publish('scenario_table.csv', base.csv_bytes(SCENE_FIELDS, [scene_values]))
        suite['identities'] = publish('run_seed.csv',
                                    base.csv_bytes(IDENTITY_FIELDS, [[job[k] for k in IDENTITY_FIELDS]]))
        suite['java_helper'] = publish('Baseline1CampaignPlan.java', render_java(suite))
        import demo_budget
        suite['budget_at_preparation'] = demo_budget.check_fresh(suite)
        seal = publish('suite.json', base.encoded(suite))
        publish('suite.sha256', (seal['sha256']+'\n').encode())
        os.rename(stage, directory)
    return seal


def load_suite(path):
    path = Path(path).resolve(strict=True)
    data = path.read_bytes()
    if base.digest(data) != path.with_name('suite.sha256').read_text().strip():
        raise ValueError('demo suite SHA differs')
    suite = json.loads(data)
    scene, pins = anchor()
    output = base.ROOT/'runs'/CAMPAIGN
    expected = dict(schema=SCHEMA, campaign_id=CAMPAIGN, suite_path=str(path), output=str(output),
                    control_directory=str(output/'control'), rpc_port=50051, **pins)
    if any(suite.get(key) != value for key, value in expected.items()):
        raise ValueError('the demo lineage or paths differ')
    for pin in [*suite['sources'], *[suite[k] for k in ('master_plan','scenario_table','java_helper','identities')]]:
        base.load_ref(pin)
    job, = suite['jobs']
    want = demo_job(scene, output)
    if ({k: v for k, v in job.items() if k not in ('config_sha256', 'config_path')} != want
            or Path(job['config_path']) != path.parent/'configs'/f'run_{RUN}.json'
            or base.load_ref(dict(path=job['config_path'], sha256=job['config_sha256']))
                != base.encoded(config(suite, scene, want))):
        raise ValueError('the demo job or configuration differs')
    job['expected_rpc_config'] = scene['expected_rpc_config']
    return suite


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', required=True, type=Path)
    print(json.dumps(prepare(parser.parse_args().directory), ensure_ascii=False))
