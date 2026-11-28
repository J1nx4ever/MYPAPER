"""Freeze the independent loop_B evaluation using the opening fleet's task."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET


HERE = Path(__file__).resolve().parent
REPAIR = HERE.parent / 'repair_001'
sys.path[:0] = [str(HERE.parent), str(REPAIR)]

import prepare_retry as previous
import prepare_trial as original
from prepare_trial import PROJECT, canonical, require, witness, write_fresh
from opening_targets import scaled_task


CAMPAIGN = 'site-layout-frozen-policy-scaled-20261127-001'
EXPERIMENT = 'Topology_Frozen_Policy_B_Scaled'
RUN = 10054046
RECORDING = f'{CAMPAIGN}-{RUN}'
# Same 48-bit derivation as the fleet training schedule, with an independent campaign namespace.
_SEED_PREFIX = sha256(f'{CAMPAIGN}|loop_B|environment|{RUN}'.encode()).digest()[:8]
SEED = 1 + int.from_bytes(_SEED_PREFIX, 'big') % ((1 << 48) - 1)
OUTPUT = PROJECT / 'runs' / f'{CAMPAIGN}-B'
CONFIG = HERE / 'configs' / 'B.json'
REPORTS = PROJECT / 'reports' / REPAIR.relative_to(PROJECT / 'change_plans')
ARCHIVED_PARENT = REPORTS / 'DRL-PPO-baseline-topologydemo-frozen-policy-trial-b-repair.alp'
PARENT_MODEL_SHA = '8757798c9132c89165746a8f08ee5a4d551745fd1a0b8a68ff363f1be76f0cd8'
TARGET_KEYS = ('ore_target_tons', 'waste_target_tons', 'waste_dispatch_stop_tons', 'waste_upper_tons')
prepared_rows = witness.bind(witness.prepared_rows, HERE=HERE)
check_fresh = witness.bind(witness.check_fresh, prepared_rows=prepared_rows)
suite_job = witness.bind(previous.suite_job, RUN=RUN, SEED=SEED, RECORDING=RECORDING, CONFIG=CONFIG, OUTPUT=OUTPUT)


def physical_reference():
    parent_pin = witness.reference(previous.CONFIG)
    config = json.loads(previous.CONFIG.read_bytes())
    require(previous.CONFIG.with_suffix('.sha256').read_text().strip() == parent_pin['sha256']
            and config['run_index'] == previous.RUN and config['recording_id'] == previous.RECORDING
            and config['layout_id'] == 'loop_B', 'prior B configuration identity differs')
    suite, job = previous.suite_job(config, parent_pin['sha256'])
    status, ledger_pin = witness.ledger_entry(suite, job)
    receipt_pin = witness.reference(Path(job['python_output']) / 'receipt.json')
    require(status == 'COMPLETE' and ledger_pin == receipt_pin, 'prior B physical run has no sealed receipt')
    receipt = json.loads(Path(receipt_pin['path']).read_bytes())
    require(receipt['physical_status'] == 'COMPLETE' and receipt['run_index'] == previous.RUN
            and receipt['config_sha256'] == parent_pin['sha256'], 'prior B receipt differs')
    bound = witness.native_manifest(Path(job['event_directory']), job)
    require(bound is not None, 'prior B native event manifest is missing')
    manifest_path, manifest, manifest_sha = bound
    require(receipt['manifest'] == dict(path=str(manifest_path), sha256=manifest_sha)
            and manifest['complete'] and manifest['terminated'] and not manifest['truncated']
            and manifest['config']['actual_runtime_scene'] == config['expected_runtime_scene'],
            'prior B physical scene or event manifest differs')
    archived = witness.reference(ARCHIVED_PARENT)
    require(archived['sha256'] == PARENT_MODEL_SHA == config['model_sha256']
            == config['expected_runtime_scene']['model_sha256'], 'archived parent model SHA differs')
    require(config['task_sha256'] == sha256(canonical(config['task'])).hexdigest(),
            'prior task does not match its frozen digest')
    ref = dict(purpose='physical_layout_reference_with_unscaled_targets', run_index=previous.RUN,
               task_sha256=config['task_sha256'], environment_seed=job['experiment_seed'],
               recording_id=job['recording_id'], config=parent_pin, receipt=receipt_pin,
               manifest=receipt['manifest'], model=archived)
    return config, ref


def make_config(model):
    parent, ref = physical_reference()
    scene = parent['expected_runtime_scene']
    task = scaled_task(parent['task'], scene['opening_fleet'])
    exogenous = dict(scene['exogenous_config'], **{key: task[key] for key in TARGET_KEYS},
                     ore_target_tons_2h=task['ore_target_tons'] / 4)
    expected = dict(scene, model_sha256=model['sha256'], exogenous_config=exogenous)
    policy = original.checkpoint_metadata()
    for key in ('checkpoint_sha256', 'actor_digest', 'action_mode', 'slot_permutation_seed'):
        require(parent[key] == policy[key], f'{key} differs from the frozen u9 physical reference')
    policy['inference_sources'] += [witness.reference(HERE / filename)
                                    for filename in ('scaled_service.py', 'opening_targets.py')]
    return dict(purpose=original.PURPOSE, experiment=EXPERIMENT, run_index=RUN,
                recording_id=RECORDING, environment_seed=SEED, **policy,
                task=task, task_sha256=sha256(canonical(task)).hexdigest(),
                model_path=model['path'], model_sha256=model['sha256'], expected_runtime_scene=expected,
                expected_runtime_scene_sha256=sha256(canonical(expected)).hexdigest(),
                layout_id=parent['layout_id'], layout_path=parent['layout_path'],
                layout_sha256=parent['layout_sha256'], physical_reference=ref,
                event_directory=str(OUTPUT / 'anylogic_events'), python_output_directory=str(OUTPUT / 'python'))


def load_case(label='B'):
    require(label == 'B', 'scaled evaluation contains only loop_B')
    raw = CONFIG.read_bytes()
    digest = sha256(raw).hexdigest()
    require(CONFIG.with_suffix('.sha256').read_text().strip() == digest, 'scaled config SHA differs')
    config = json.loads(raw)
    require(config == make_config(witness.reference(Path(config['model_path']))),
            'frozen scaled inputs or parent lineage differ')
    suite, job = suite_job(config, digest)
    return config, suite, job


def read_case(case='B'):
    config, _, job = load_case(case)
    layout, = (row for row in witness.load_layouts() if row['id'] == config['layout_id'])
    compiled, edges = witness.compile_layout(layout)
    require(sha256(compiled).hexdigest() == config['layout_sha256']
            and Path(config['layout_path']).read_bytes() == compiled, 'scaled layout bytes differ')
    return config, job['config_sha256'], edges


def prepare(model_path):
    model = witness.reference(model_path.resolve(strict=True))
    experiments = ET.parse(model['path']).findall('./Model/Experiments/ParamVariationExperiment')
    require(model_path.suffix == '.alp' and any(entry.findtext('Name') == EXPERIMENT
            and entry.findtext('FreeformParamValue/Expression/Code') == '2'
            and entry.findtext('NumberOfRuns') == '1' for entry in experiments),
            'expected the independent loop_B scaled experiment')
    config = make_config(model)
    raw = canonical(config)
    digest = sha256(raw).hexdigest()
    suite, job = suite_job(config, digest)
    history = witness.budget_snapshot()
    witness.bind(check_fresh, budget_snapshot=lambda: history)(suite)
    require(not Path(job['event_directory']).exists() and not Path(job['python_output']).exists()
            and not witness.ledger.ledger_path(suite).exists(), 'scaled run already has artifacts')
    fields = ('experiment', 'layout_id', 'run_index', 'environment_seed', 'recording_id',
              'python_output_directory', 'event_directory', 'checkpoint_sha256', 'action_mode', 'task_sha256')
    identity = dict(campaign_id=CAMPAIGN, model=model, registered_at_preparation=history['maximum_total_shifts'],
                    registered_by_preparation=0, cases=[dict(case='B', config_path=str(CONFIG),
                    config_sha256=digest, **{key: config[key] for key in fields})])
    CONFIG.parent.mkdir(exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_fresh(CONFIG, raw)
    write_fresh(CONFIG.with_suffix('.sha256'), (digest + '\n').encode())
    write_fresh(HERE / 'launch_identity.json', canonical(identity))
    return identity


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=Path)
    print(json.dumps(prepare(parser.parse_args().model), indent=2, ensure_ascii=False))
