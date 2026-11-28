"""Freeze native evaluation identities and inputs for the dynamic BG128 u9 actor."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
WITNESS_DIR = HERE.parent / 'feasibility_001' / 'full_shift_001'
sys.path.insert(0, str(WITNESS_DIR))

import collect_witness as witness
from prepare_witness import write_fresh


PROJECT = witness.PROJECT
CAMPAIGN = 'site-layout-frozen-policy-trial-20261127-001'
PURPOSE = 'topology_frozen_policy_trial'
CASES = {
    'A': ('tree_A', 10054041, 28947842869511),
    'B': ('loop_B', 10054042, 192220921542182),
}
CHECKPOINT = PROJECT / 'runs/bg128-fleet-resilience-20261124-repair01/models/dynamic/u0009/checkpoint.pt'
CHECKPOINT_SHA256 = 'b13e86feb74656def9077e412dbf4443857fba777ecec5e8ac758fd79a191f36'
ACTOR_DIGEST = '16b82688d37c0786d5b1510f0598763feb3d37cef590a68c4d5f05aa8b55e5a6'
POLICY_ID = 'bg128-fleet-resilience-dynamic-u9'
INFERENCE_PORT = 50871
canonical = witness.canonical
require = witness.require


def checkpoint_metadata():
    binary = witness.reference(CHECKPOINT)
    manifest = witness.reference(CHECKPOINT.with_suffix('.pt.manifest.json'))
    seal = witness.reference(CHECKPOINT.with_suffix('.pt.seal.json'))
    metadata = json.loads(Path(manifest['path']).read_bytes())
    sealing = json.loads(Path(seal['path']).read_bytes())
    require(binary['sha256'] == metadata['sha256'] == sealing['sha256'] == CHECKPOINT_SHA256
            and metadata['bytes'] == CHECKPOINT.stat().st_size
            and metadata['actor_digest'] == ACTOR_DIGEST and metadata['update_index'] == 9
            and manifest['sha256'] == sealing['manifest_sha256'], 'dynamic u9 checkpoint seal differs')
    sources = (HERE / 'trial_service.py', HERE / 'frozen_topology_policy.py',
               HERE.parent / 'topology_policy.py',
               HERE.parents[1] / 'fleet_outage_resilience_001' / 'fleet_policy.py')
    return dict(policy_id=POLICY_ID, checkpoint_path=binary['path'], checkpoint_sha256=binary['sha256'],
                checkpoint_manifest=manifest, checkpoint_seal=seal, actor_digest=ACTOR_DIGEST,
                action_mode='argmax', actor_sampling_seed=None, slot_permutation_seed=0,
                inference_host='127.0.0.1', inference_port=INFERENCE_PORT,
                inference_sources=[witness.reference(path) for path in sources])


def admitted_witness(label):
    task = json.loads((WITNESS_DIR.parent / 'task.json').read_bytes())
    layouts = {layout['id']: layout for layout in witness.load_layouts()}
    suite, job, _ = witness.case_config(label, layouts, task)
    config_pin = witness.reference(suite['suite_path'])
    original_config = json.loads(suite['suite_path'].read_bytes())
    receipt_path = Path(job['python_output']) / 'receipt.json'
    status, pin = witness.ledger_entry(suite, job)
    require(status == 'COMPLETE' and witness.reference(receipt_path) == pin,
            f'{label} original witness has no intact COMPLETE receipt')
    receipt = json.loads(receipt_path.read_bytes())
    require(receipt['physical_status'] == 'COMPLETE' and receipt['task_status'] == 'ADMITTED',
            f'{label} original witness did not prove the task feasible')
    manifest = receipt['manifest']
    require(witness.reference(Path(manifest['path'])) == manifest, f'{label} original witness event seal differs')
    source = dict(config=config_pin, receipt=pin, event_manifest=manifest, run_index=job['run_index'],
                  recording_id=job['recording_id'], environment_seed=job['experiment_seed'],
                  events_sha256=receipt['events_sha256'])
    return original_config, source, receipt


def campaign_root(label):
    return PROJECT / 'runs' / f'{CAMPAIGN}-{label}'


def make_case(label, model, policy):
    original, baseline, _ = admitted_witness(label)
    layout, run, seed = CASES[label]
    require(original['layout_id'] == layout and original['environment_seed'] != seed,
            'trial and source witness identities or layout differ')
    runtime_scene = dict(original['expected_runtime_scene'], model_sha256=model['sha256'])
    output = campaign_root(label)
    return dict(original, **policy, purpose=PURPOSE, run_index=run, environment_seed=seed,
                recording_id=f'{CAMPAIGN}-{run}', experiment=f'Topology_Frozen_Policy_{label}',
                model_path=model['path'], model_sha256=model['sha256'],
                expected_runtime_scene=runtime_scene,
                expected_runtime_scene_sha256=sha256(canonical(runtime_scene)).hexdigest(),
                event_directory=str(output / 'anylogic_events'), python_output_directory=str(output / 'python'),
                source_witness=baseline)


def suite_job(label, config, config_path, digest):
    output = campaign_root(label)
    job = dict(run_index=config['run_index'], experiment_seed=config['environment_seed'],
               recording_id=config['recording_id'], source_recording_id=config['recording_id'],
               config_sha256=digest, layout_id=config['layout_id'], actor_seed=0, actor_sampling_seed=0,
               python_output=config['python_output_directory'], event_directory=config['event_directory'])
    suite = dict(campaign_id=output.name, output=str(output), suite_path=config_path, jobs=[job])
    return suite, job


prepared_rows = witness.bind(witness.prepared_rows, HERE=HERE)
check_fresh = witness.bind(witness.check_fresh, prepared_rows=prepared_rows)


def load_case(label):
    config_path = HERE / 'configs' / f'{label}.json'
    raw = config_path.read_bytes()
    digest = sha256(raw).hexdigest()
    require(config_path.with_suffix('.sha256').read_text().strip() == digest, 'trial configuration SHA differs')
    config = json.loads(raw)
    model = witness.reference(Path(config['model_path']))
    require(config == make_case(label, model, checkpoint_metadata()), 'frozen trial configuration differs')
    suite, job = suite_job(label, config, config_path, digest)
    return config, suite, job


def prepare(model_path):
    model = witness.reference(model_path.resolve(strict=True))
    require(model_path.suffix == '.alp', 'expected AnyLogic trial ALP')
    model_bytes = Path(model['path']).read_bytes()
    require(all(f'Topology_Frozen_Policy_{label}'.encode() in model_bytes for label in CASES)
            and sha256(model_bytes).hexdigest() == model['sha256'], 'trial ALP experiments are missing')
    policy = checkpoint_metadata()
    history = witness.budget_snapshot()
    preflight = witness.bind(check_fresh, budget_snapshot=lambda: history)
    identities = []
    for label in CASES:
        config = make_case(label, model, policy)
        raw = canonical(config)
        path = HERE / 'configs' / f'{label}.json'
        digest = sha256(raw).hexdigest()
        suite, job = suite_job(label, config, path, digest)
        preflight(suite)
        require(not Path(job['event_directory']).exists() and not Path(job['python_output']).exists()
                and not witness.ledger.ledger_path(suite).exists(), 'trial already has run artifacts')
        identities.append((label, config, raw, path, digest))

    cases = []
    (HERE / 'configs').mkdir(exist_ok=True)
    for label, config, raw, path, digest in identities:
        campaign_root(label).mkdir(parents=True, exist_ok=True)
        write_fresh(path, raw)
        write_fresh(path.with_suffix('.sha256'), (digest + '\n').encode())
        cases.append(dict(case=label, config_path=str(path), config_sha256=digest, **{
            key: config[key] for key in ('experiment', 'layout_id', 'run_index', 'environment_seed',
                'recording_id', 'python_output_directory', 'event_directory', 'action_mode', 'source_witness')
        }))
    identity = dict(campaign_id=CAMPAIGN, policy=policy, model=model, cases=cases,
                    registered_at_preparation=history['maximum_total_shifts'], registered_by_preparation=0)
    write_fresh(HERE / 'launch_identity.json', canonical(identity))
    return identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.model), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
