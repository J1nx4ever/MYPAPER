"""Freeze two independently seeded cohorts and settle their native runs individually."""

import argparse
from contextlib import contextmanager, nullcontext
import csv
from functools import lru_cache
from hashlib import sha256
import io
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET


HERE = Path(__file__).resolve().parent
TRIAL = HERE.parent
SCALED = TRIAL / 'capacity_scaled_001'
sys.path[:0] = [str(TRIAL), str(SCALED)]
import prepare_trial as original
import prepare_scaled as scaled
from prepare_trial import PROJECT, canonical, require, witness, write_fresh
from opening_targets import scaled_task


CAMPAIGN = 'site-layout-frozen-policy-stats50-20261127-001'
CASES = tuple(f'{group}_{slot:03d}' for group in 'AB' for slot in range(1, 51))
EXPERIMENTS = dict(A='Topology_Frozen_Stats_A50', B='Topology_Frozen_Stats_B50')
OVERVIEW = HERE / 'launch_identity.json'
REPORTS = PROJECT / 'reports' / HERE.relative_to(PROJECT / 'change_plans')
ledger_entry = witness.ledger_entry


def identity(label):
    require(label in CASES, 'unknown topology statistics case')
    group = label[0]
    slot = int(label[2:])
    code = (100 if group == 'A' else 200) + slot
    run = 10054000 + code
    seed_key = f'{CAMPAIGN}|{label}|environment|{run}'.encode()
    seed = 1 + int.from_bytes(sha256(seed_key).digest()[:8], 'big') % ((1 << 48) - 1)
    out = PROJECT / 'runs' / f'{CAMPAIGN}-{label}'
    return dict(case=label, group=group, slot=slot, native_case_code=code, experiment=EXPERIMENTS[group],
                run_index=run, environment_seed=seed, recording_id=f'{CAMPAIGN}-{run}',
                event_directory=str(out / 'anylogic_events'), python_output_directory=str(out / 'python'))


@lru_cache(maxsize=2)
def physical_base(group):
    if group == 'A':
        path = TRIAL / 'configs/A.json'
    else:
        path = SCALED / 'configs/B.json'
    pin = witness.reference(path)
    require(path.with_suffix('.sha256').read_text().strip() == pin['sha256'], 'physical reference config SHA differs')
    config = json.loads(path.read_bytes())
    suite, job = (original.suite_job('A', config, path, pin['sha256']) if group == 'A'
                  else scaled.suite_job(config, pin['sha256']))
    status, receipt_pin = ledger_entry(suite, job)
    output = Path(job['python_output'])
    require(status == 'COMPLETE' and receipt_pin == witness.reference(output / 'receipt.json'),
            'physical reference has no intact COMPLETE receipt')
    receipt = json.loads((output / 'receipt.json').read_bytes())
    bound = witness.native_manifest(Path(job['event_directory']), job)
    require(bound is not None, 'reference native manifest is missing')
    manifest_path, manifest, digest = bound
    scene = config['expected_runtime_scene']
    require(receipt['physical_status'] == 'COMPLETE' and receipt['config_sha256'] == pin['sha256']
            and receipt['manifest'] == dict(path=str(manifest_path), sha256=digest)
            and manifest['complete'] and manifest['terminated'] and not manifest['truncated']
            and manifest['config']['actual_runtime_scene'] == scene
            and scene['model_sha256'] == config['model_sha256'], 'reference receipt or runtime scene differs')
    if group == 'A':
        require(witness.reference(scaled.previous.ARCHIVED_MODEL)['sha256'] == config['model_sha256'],
                'archived A model differs')
    policy_pin = witness.reference(output / 'policy_verification.json')
    audit = json.loads((output / 'policy_verification.json').read_bytes())
    require(audit['status'] == 'VERIFIED' and audit['checkpoint_sha256'] == config['checkpoint_sha256']
            == original.CHECKPOINT_SHA256 and audit['decision_count'] == manifest['native_policy_decisions'],
            'source policy proof differs')
    require(config['task_sha256'] == sha256(canonical(config['task'])).hexdigest(), 'reference task differs')
    ref = dict(role='deterministic_layout_and_workload_reference', run_index=config['run_index'],
               recording_id=config['recording_id'], config=pin, receipt=receipt_pin,
               manifest=receipt['manifest'], policy_verification=policy_pin, model_sha256=config['model_sha256'])
    return scene, config['task'], config['layout_path'], config['layout_sha256'], ref


@lru_cache(maxsize=1)
def policy_metadata():
    policy = original.checkpoint_metadata()
    policy['inference_sources'] += [witness.reference(path) for path in (
        SCALED / 'opening_targets.py', SCALED / 'scaled_service.py', HERE / 'stats_service.py')]
    return policy


def make_config(label, model):
    ident = identity(label)
    group = ident['group']
    scene, task, path, layout_sha, ref = physical_base(group)
    # A supplies the nine-truck nominal task. Generate both outcomes from opening attendance.
    derived = scaled_task(physical_base('A')[1], scene['opening_fleet'])
    require(all(abs(task[key] - derived[key]) <= 1e-9 for key in scaled.TARGET_KEYS),
            'opening fleet scaling differs from reference task')
    exogenous = dict(scene['exogenous_config'], **{key: derived[key] for key in scaled.TARGET_KEYS},
                     deterministic=False, ore_target_tons_2h=derived['ore_target_tons']/4)
    require(not exogenous['local_dynamic_speed'] and not exogenous['endogenous_congestion']
            and exogenous['fault_truck_id'] == -1 and exogenous['fault_trigger_seconds'] == -1,
            'unexpected exogenous reference scenario')
    expected = dict(scene, model_sha256=model['sha256'], exogenous_config=exogenous)
    return dict(purpose=original.PURPOSE, **ident, **policy_metadata(), task=derived,
                task_sha256=sha256(canonical(derived)).hexdigest(), physical_reference=ref,
                layout_id=scene['layout_id'], layout_path=path, layout_sha256=layout_sha,
                model_path=model['path'], model_sha256=model['sha256'], expected_runtime_scene=expected,
                expected_runtime_scene_sha256=sha256(canonical(expected)).hexdigest())


def suite_job(config, digest):
    out = Path(config['python_output_directory']).parent
    job = dict(run_index=config['run_index'], experiment_seed=config['environment_seed'],
               recording_id=config['recording_id'], source_recording_id=config['recording_id'],
               config_sha256=digest, layout_id=config['layout_id'], actor_seed=0, actor_sampling_seed=0,
               python_output=config['python_output_directory'], event_directory=config['event_directory'])
    suite = dict(campaign_id=out.name, output=str(out), suite_path=HERE/'configs'/f"{config['case']}.json",
                 jobs=[job])
    return suite, job


def historical_check(identities, history):
    """Validate disjoint run IDs, recordings, seeds and effective Java states in one historical scan."""
    runs = {row['run_index'] for row in identities}
    recordings = {row['recording_id'] for row in identities}
    seeds = {row['environment_seed'] for row in identities}
    mask = witness.ledger.JAVA_MASK
    states = {x for seed in seeds for x in witness.ledger.java_random_states(seed).values()}
    count = len(identities)
    require(len(runs) == len(recordings) == len(seeds) == count and len(states) == 5*count
            and all(0 < seed <= mask for seed in seeds), 'duplicate batch run, recording, seed or random stream')
    used_campaigns = history['campaigns']
    require(all(f"{CAMPAIGN}-{ident['case']}" not in used_campaigns for ident in identities),
            'requested batch cases already registered')
    prior = witness.bind(witness.ledger.previous.paired10._prior_rows, SUITE=OVERVIEW)(history)
    prior.extend(row['identity'] for row in history['jobs'].values() if isinstance(row.get('identity'), dict))
    for path in sorted(witness.GRAPH.rglob('launch_identity.json')):
        if path == OVERVIEW:
            continue
        entries = json.loads(path.read_bytes()).get('cases', [])
        prior.extend(entries.values() if isinstance(entries, dict) else entries)
    old_environment = set()
    for row in prior:
        require(row.get('run_index') not in runs
                and row.get('recording_id') not in recordings and row.get('source_recording_id') not in recordings,
                'batch run or recording overlaps prior registration or frozen reservation')
        for key in witness.SEED_FIELDS:
            value = row.get(key)
            if type(value) is int and value > 0:
                require(value & mask not in seeds, 'batch seed is already used or frozen')
                if key in ('experiment_seed', 'environment_seed'):
                    old_environment.add(value)
    require(all(states.isdisjoint(witness.ledger.java_random_states(old).values())
                for old in old_environment), 'batch Java random states overlap history')


@lru_cache(maxsize=4)
def _model_pin(path, size, modified_ns):
    return witness.reference(Path(path))


def model_pin(path):
    info = path.stat()
    return _model_pin(str(path), info.st_size, info.st_mtime_ns)


@lru_cache(maxsize=1)
def frozen_index():
    raw = OVERVIEW.read_bytes()
    require(OVERVIEW.with_suffix('.sha256').read_text().strip() == sha256(raw).hexdigest(),
            'batch identity list SHA differs')
    overview = json.loads(raw)
    rows = overview['cases']
    require(overview['campaign_id'] == CAMPAIGN and tuple(row['case'] for row in rows) == CASES,
            'batch index identity or case order differs')
    return {row['case']: row for row in rows}


def load_case(label):
    require(label in CASES, 'unknown frozen statistics case')
    row = frozen_index()[label]
    ident = identity(label)
    require(all(row[key] == value for key, value in ident.items()), 'frozen run identity differs')
    path = HERE / 'configs' / f'{label}.json'
    raw = path.read_bytes()
    digest = sha256(raw).hexdigest()
    require(str(path) == row['config_path'] and path.with_suffix('.sha256').read_text().strip()
            == digest == row['config_sha256'], 'frozen case configuration SHA differs')
    config = json.loads(raw)
    model_path = Path(config['model_path'])
    model = dict(path=str(model_path), sha256=config['model_sha256'])
    archive = model_pin(REPORTS / model_path.name)
    require(archive['sha256'] == model['sha256'], 'archived cohort model SHA differs')
    require(config == make_config(label, model),
            'frozen case, checkpoint or physical inputs differ')
    suite, job = suite_job(config, digest)
    return config, suite, job


@lru_cache(maxsize=2)
def layout_inputs(group):
    scene, _, path, digest, _ = physical_base(group)
    layout, = (row for row in witness.load_layouts() if row['id'] == scene['layout_id'])
    compiled, edges = witness.compile_layout(layout)
    require(sha256(compiled).hexdigest() == digest and Path(path).read_bytes() == compiled,
            'compiled physical layout differs')
    return edges


def read_case(label):
    config, _, job = load_case(label)
    require(model_pin(Path(config['model_path']))['sha256'] == config['model_sha256'],
            'native launch model has changed since preparation')
    return config, job['config_sha256'], layout_inputs(config['group'])


def register_started():
    # Reuse the project's budget lock. Register only slots with a bound native EPISODE_START manifest.
    with witness.ledger._launch_lock(witness.ledger.previous.LOCKS/'budget'):
        pending = []
        for label in CASES:
            ident = identity(label)
            event_dir = Path(ident['event_directory'])
            if not event_dir.is_dir():
                continue
            _, suite, job = load_case(label)
            if witness.ledger.ledger_path(suite).exists():
                ledger_entry(suite, job)
                continue
            try:
                manifest = witness.native_manifest(event_dir, job)
            except witness.EVIDENCE_ERRORS:
                continue   # The collector separately reports invalid or incomplete native evidence.
            if manifest is not None:
                pending.append((ident, suite))
        if not pending:
            return
        historical_check([ident for ident, _ in pending], witness.budget_snapshot())
        # The outer budget lock spans this complete check and every SQLite publication.
        approved_ids = {suite['campaign_id'] for _, suite in pending}
        def checked(suite):
            require(suite['campaign_id'] in approved_ids, 'run not in the checked native batch')
        publish = witness.bind(witness.ledger.register, check_fresh=checked, _launch_lock=lambda path: nullcontext())
        for _, suite in pending:
            publish(suite)


@contextmanager
def settlement_session():
    register_started()
    def settle(suite, job, physical, *, collect=True):
        if collect and physical['physical_status'] in ('COMPLETE', 'UNKNOWN') and not witness.ledger.ledger_path(suite).exists():
            register_started()  # A native manifest may have appeared since the batch check.
            require(witness.ledger.ledger_path(suite).is_file(), 'no identity-bound native evidence to register')
        with witness.ledger._launch_lock(Path(suite['output'])/'collector'):
            return witness.settle(suite, job, physical, collect=collect)
    yield settle


def prepare(model_path):
    model = model_pin(model_path.resolve(strict=True))
    experiments = ET.parse(model['path']).findall('./Model/Experiments/ParamVariationExperiment')
    by_name = {entry.findtext('Name'): entry for entry in experiments}
    require(model_path.suffix == '.alp', 'expected an AnyLogic model')
    for group, name in EXPERIMENTS.items():
        expr = by_name.get(name)
        first_code = identity(f'{group}_001')['native_case_code']
        require(expr is not None and expr.findtext('NumberOfRuns') == '50'
                and expr.findtext('AllowParallelEvaluations') == 'false'
                and expr.findtext('FreeformParamValue/Expression/Code') == f'{first_code} + index',
                f'unexpected iteration count or native case sequence in {name}')
    pins = []
    configs = []
    for label in CASES:
        config = make_config(label, model)
        raw = canonical(config)
        digest = sha256(raw).hexdigest()
        suite, job = suite_job(config, digest)
        require(not Path(job['event_directory']).exists() and not Path(job['python_output']).exists()
                and not witness.ledger.ledger_path(suite).exists(), 'statistics case already has run artifacts')
        row = dict(identity(label), config_path=str(suite['suite_path']), config_sha256=digest,
                   layout_id=config['layout_id'], actor_sampling_seed=None,
                   task_sha256=config['task_sha256'], checkpoint_sha256=config['checkpoint_sha256'])
        pins.append(row)
        configs.append((suite['suite_path'], raw, digest))
    with witness.ledger._launch_lock(witness.ledger.previous.LOCKS/'budget'):
        history = witness.budget_snapshot()
        historical_check(pins, history)
        (HERE/'configs').mkdir(parents=True, exist_ok=True)
        REPORTS.mkdir(parents=True, exist_ok=True)
        model_bytes = Path(model['path']).read_bytes()
        require(sha256(model_bytes).hexdigest() == model['sha256'], 'native model changed during preparation')
        archived_model = REPORTS / Path(model['path']).name
        write_fresh(archived_model, model_bytes)
        # The native recorder creates only its leaf directory.
        for row in pins:
            Path(row['event_directory']).parent.mkdir(parents=True, exist_ok=True)
        for path, raw, digest in configs:
            write_fresh(path, raw)
            write_fresh(path.with_suffix('.sha256'), (digest+'\n').encode())
        table = io.StringIO(newline='')
        writer = csv.DictWriter(table, fieldnames=list(pins[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(pins)
        write_fresh(HERE/'run_index_seed_manifest.csv', table.getvalue().encode())
        overview = dict(campaign_id=CAMPAIGN, model=model, archived_model=witness.reference(archived_model),
                        registered_at_preparation=history['maximum_total_shifts'],
                        registered_by_preparation=0, cases=pins)
        data = canonical(overview)
        write_fresh(OVERVIEW, data)
        write_fresh(OVERVIEW.with_suffix('.sha256'), (sha256(data).hexdigest()+'\n').encode())
    return overview


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=Path)
    plan = prepare(parser.parse_args().model)
    print(json.dumps(dict(campaign_id=CAMPAIGN, cases=len(plan['cases']), identity=str(OVERVIEW),
                          registered_at_preparation=plan['registered_at_preparation'])))
