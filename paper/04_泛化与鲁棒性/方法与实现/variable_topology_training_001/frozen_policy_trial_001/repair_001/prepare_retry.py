"""Freeze a new run for the interrupted loop_B frozen-policy evaluation."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import prepare_trial as original
from prepare_trial import PROJECT, canonical, require, witness, write_fresh


CAMPAIGN = 'site-layout-frozen-policy-trial-20261127-repair01'
EXPERIMENT = 'Topology_Frozen_Policy_B_Repair'
RUN = 10054043
SEED = 279598031114873
RECORDING = f'{CAMPAIGN}-{RUN}'
OUTPUT = PROJECT / 'runs' / f'{CAMPAIGN}-B'
CONFIG = HERE / 'configs' / 'B.json'
ARCHIVED_MODEL = (PROJECT / 'reports/token_attention_001/business_graph_summary_009'
                  / 'variable_topology_training_001/frozen_policy_trial_001'
                  / 'DRL-PPO-baseline-topologydemo-frozen-policy-trial.alp')
prepared_rows = witness.bind(witness.prepared_rows, HERE=HERE)
check_fresh = witness.bind(witness.check_fresh, prepared_rows=prepared_rows)


def historical_case():
    config_path = original.HERE / 'configs' / 'B.json'
    raw = config_path.read_bytes()
    digest = sha256(raw).hexdigest()
    require(config_path.with_suffix('.sha256').read_text().strip() == digest,
            'historical trial configuration SHA differs')
    config = json.loads(raw)
    model = dict(path=config['model_path'], sha256=config['model_sha256'])
    try:
        actual = witness.reference(Path(model['path']))
    except FileNotFoundError:
        actual = None
    if actual != model:
        archived = witness.reference(ARCHIVED_MODEL)
        require(archived['sha256'] == model['sha256'],
                'archived historical trial model SHA differs')
    # Rebuild against the original identity even when its bytes are read from the archive.
    require(config == original.make_case('B', model, original.checkpoint_metadata()),
            'frozen historical trial configuration differs')
    suite, job = original.suite_job('B', config, config_path, digest)
    return config, suite, job


def original_abort():
    config, suite, job = historical_case()
    status, pin = witness.ledger_entry(suite, job)
    require(status == 'UNKNOWN', 'settle original loop_B as UNKNOWN before preparing the retry')
    bound = witness.native_manifest(Path(job['event_directory']), job)
    require(bound is not None, 'aborted loop_B has no bound native manifest')
    path, manifest, digest = bound
    require(not manifest['complete'] and not manifest['terminated'] and manifest['truncated'],
            'original loop_B source is not an interrupted shift')
    manifest_pin = dict(path=str(path), sha256=digest)
    require(pin['manifest'] == manifest_pin, 'archived original B event manifest changed')
    return config, dict(run_index=job['run_index'], environment_seed=job['experiment_seed'],
                        recording_id=job['recording_id'], ledger_status=status,
                        config=witness.reference(suite['suite_path']), event_manifest=manifest_pin,
                        events=witness.reference(path.with_name('events.jsonl')),
                        inference=witness.reference(Path(job['python_output']) / 'inference.jsonl'))


def make_config(model):
    config, previous = original_abort()
    require(SEED not in (config['environment_seed'], config['source_witness']['environment_seed']),
            'retry environment seed reuses a historical run')
    scene = dict(config['expected_runtime_scene'], model_sha256=model['sha256'])
    sources = [*config['inference_sources'], witness.reference(HERE / 'repair_service.py')]
    return dict(config, run_index=RUN, environment_seed=SEED, recording_id=RECORDING,
                experiment=EXPERIMENT, model_path=model['path'], model_sha256=model['sha256'],
                expected_runtime_scene=scene,
                expected_runtime_scene_sha256=sha256(canonical(scene)).hexdigest(),
                event_directory=str(OUTPUT / 'anylogic_events'),
                python_output_directory=str(OUTPUT / 'python'),
                inference_sources=sources, original_aborted_trial=previous)


def suite_job(config, digest):
    job = dict(run_index=RUN, experiment_seed=SEED, recording_id=RECORDING, source_recording_id=RECORDING,
               config_sha256=digest, layout_id=config['layout_id'], actor_seed=0, actor_sampling_seed=0,
               python_output=config['python_output_directory'], event_directory=config['event_directory'])
    suite = dict(campaign_id=OUTPUT.name, output=str(OUTPUT), suite_path=CONFIG, jobs=[job])
    return suite, job


def load_case():
    raw = CONFIG.read_bytes()
    digest = sha256(raw).hexdigest()
    require(CONFIG.with_suffix('.sha256').read_text().strip() == digest, 'retry configuration SHA differs')
    config = json.loads(raw)
    require(config == make_config(witness.reference(Path(config['model_path']))),
            'frozen retry inputs or lineage have changed')
    suite, job = suite_job(config, digest)
    return config, suite, job


def read_case(case='B'):
    """Configuration entry point for the same HTTP server and checkpoint."""
    require(case == 'B', 'repair serves only loop_B')
    config, _, job = load_case()
    layout, = (row for row in witness.load_layouts() if row['id'] == config['layout_id'])
    scene, edges = witness.compile_layout(layout)
    require(sha256(scene).hexdigest() == config['layout_sha256'], 'retry layout bytes differ')
    return config, job['config_sha256'], edges


def prepare(model_path):
    model = witness.reference(model_path.resolve(strict=True))
    require(model_path.suffix == '.alp' and EXPERIMENT.encode() in Path(model['path']).read_bytes(),
            'expected the dedicated native loop_B repair experiment')
    config = make_config(model)
    raw = canonical(config)
    digest = sha256(raw).hexdigest()
    suite, job = suite_job(config, digest)
    history = witness.budget_snapshot()
    witness.bind(check_fresh, budget_snapshot=lambda: history)(suite)
    require(not Path(job['event_directory']).exists() and not Path(job['python_output']).exists()
            and not witness.ledger.ledger_path(suite).exists(), 'retry already has run artifacts')
    identity = dict(campaign_id=CAMPAIGN, model=model, registered_at_preparation=history['maximum_total_shifts'],
                    registered_by_preparation=0, cases=[dict(case='B', config_path=str(CONFIG),
                    config_sha256=digest, **{key: config[key] for key in (
                        'experiment', 'layout_id', 'run_index', 'environment_seed', 'recording_id',
                        'python_output_directory', 'event_directory', 'checkpoint_sha256',
                        'action_mode', 'original_aborted_trial', 'source_witness')})])
    CONFIG.parent.mkdir(exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_fresh(CONFIG, raw)
    write_fresh(CONFIG.with_suffix('.sha256'), (digest + '\n').encode())
    write_fresh(HERE / 'launch_identity.json', canonical(identity))
    return identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.model), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
