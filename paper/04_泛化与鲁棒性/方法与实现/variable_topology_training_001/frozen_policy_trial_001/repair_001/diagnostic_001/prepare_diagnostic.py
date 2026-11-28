"""Freeze and account for the partial native loop_B prefix diagnostic."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from prepare_retry import PROJECT, canonical, original_abort, require, witness, write_fresh


CAMPAIGN = 'site-layout-frozen-policy-diagnostic-20261127-001'
RUN = 10054044
SEED = 203918397781720
RECORDING = f'{CAMPAIGN}-{RUN}'
CASE = 'B'
SCRATCH = Path('/tmp/faith-topology-prefix-diagnostic-10054044')
WORKLOAD = 'PARTIAL_NATIVE_DIAGNOSTIC'
CONFIG = HERE / 'configs' / 'B.json'
TAPE = HERE / 'action_tape.jsonl'
OUTPUT = PROJECT / 'runs' / f'{CAMPAIGN}-{CASE}'
FIELDS = ('decision_sequence', 'truck_id', 'sim_time_seconds', 'action', 'observation_sha256',
          'action_mask', 'dispatchable_mask', 'in_system_mask')
prepared_rows = witness.bind(witness.prepared_rows, HERE=HERE)
check_fresh = witness.bind(witness.check_fresh, prepared_rows=prepared_rows)


def identity(job):
    return dict(witness.ledger.identity(job), workload=WORKLOAD)


register = witness.bind(witness.ledger.register, check_fresh=check_fresh, identity=identity)
transition = witness.bind(witness.ledger.transition, identity=identity)


def last_event(path):
    if not path.is_file():
        return 0, None
    with path.open('rb') as stream:
        count, raw = 0, None
        for count, raw in enumerate(stream, 1):
            pass
    return count, json.loads(raw) if raw is not None else None


def source_tape():
    original, source = original_abort()
    raw = Path(source['inference']['path']).read_bytes()
    require(sha256(raw).hexdigest() == source['inference']['sha256'] and raw.endswith(b'\n'),
            'original actor prefix bytes changed')
    rows = [json.loads(line) for line in raw.splitlines()]
    require(len(rows) == 56 and all(
        row['decision_sequence'] == step and row['recording_id'] == source['recording_id']
        and row['run_index'] == source['run_index'] and row['config_sha256'] == source['config']['sha256']
        and row['checkpoint_sha256'] == original['checkpoint_sha256']
        for step, row in enumerate(rows)), 'original B inference prefix identity differs')
    event_path = Path(source['events']['path'])
    count, last = last_event(event_path)
    require(last is not None and count == 15340 and last['event_seq'] == 15339
            and last['event_type'] == 'UNLOAD_COMPLETE'
            and last['sim_time_seconds'] == 8624.941626628355,
            'original B last native event differs')
    tape = b''.join(canonical({key: row[key] for key in FIELDS}) for row in rows)
    return original, tape, dict(
        source=source, action_count=len(rows), action_tape_path=str(TAPE),
        action_tape_sha256=sha256(tape).hexdigest(),
        expected_last_action_seconds=rows[-1]['sim_time_seconds'],
        expected_last_native_event='UNLOAD_COMPLETE',
        expected_last_native_event_seconds=last['sim_time_seconds'],
        expected_last_native_event_sequence=last['event_seq'],
        expected_first_unavailable_decision_sequence=len(rows))


def job_for(config_digest):
    job = dict(run_index=RUN, experiment_seed=SEED, recording_id=RECORDING,
               source_recording_id=RECORDING, actor_seed=0, actor_sampling_seed=0,
               config_sha256=config_digest, event_directory=str(SCRATCH / 'anylogic_events'),
               python_output=str(SCRATCH / 'python'))
    return dict(campaign_id=OUTPUT.name, suite_path=CONFIG, output=str(OUTPUT), jobs=[job]), job


def prepare(model=None, experiment=None):
    original, tape, prefix = source_tape()
    config = dict(original, run_index=RUN, recording_id=RECORDING, environment_seed=SEED,
                  event_directory=str(SCRATCH / 'anylogic_events'),
                  python_output_directory=str(SCRATCH / 'python'),
                  evidence_purpose=WORKLOAD, diagnostic_prefix=prefix)
    if model is not None:
        model = witness.reference(model.resolve(strict=True))
        scene = dict(config['expected_runtime_scene'], model_sha256=model['sha256'])
        config.update(model_path=model['path'], model_sha256=model['sha256'], expected_runtime_scene=scene,
                      expected_runtime_scene_sha256=sha256(canonical(scene)).hexdigest())
    if experiment is not None:
        config['experiment'] = experiment
    raw = canonical(config)
    digest = sha256(raw).hexdigest()
    suite, job = job_for(digest)
    history = witness.budget_snapshot()
    witness.bind(check_fresh, budget_snapshot=lambda: history)(suite)
    require(not Path(job['event_directory']).exists() and not Path(job['python_output']).exists()
            and not witness.ledger.ledger_path(suite).exists(), 'diagnostic already has run artifacts')
    identity = dict(campaign_id=CAMPAIGN, purpose=WORKLOAD,
                    registered_at_preparation=history['maximum_total_shifts'], registered_by_preparation=0,
                    cases=[dict(case=CASE, run_index=RUN, recording_id=RECORDING, environment_seed=SEED,
                                config_path=str(CONFIG), config_sha256=digest, experiment=config['experiment'],
                                python_output_directory=job['python_output'], event_directory=job['event_directory'],
                                diagnostic_prefix=prefix)])
    CONFIG.parent.mkdir(exist_ok=True)
    write_fresh(CONFIG, raw)
    write_fresh(CONFIG.with_suffix('.sha256'), (digest + '\n').encode())
    write_fresh(TAPE, tape)
    write_fresh(HERE / 'launch_identity.json', canonical(identity))
    scratch_configs = SCRATCH / 'configs'
    scratch_configs.mkdir(parents=True, exist_ok=True)
    write_fresh(scratch_configs / CONFIG.name, raw)
    write_fresh(scratch_configs / CONFIG.with_suffix('.sha256').name, (digest + '\n').encode())
    return identity


def started(suite, job):
    bound = witness.native_manifest(Path(job['event_directory']), job)
    require(bound is not None and not bound[1]['complete'] and not bound[1]['terminated']
            and bound[1]['truncated'], 'no bound partial native diagnostic start')
    path, _, digest = bound
    events = path.with_name('events.jsonl')
    with events.open('rb') as stream:
        first = json.loads(stream.readline())
    require(first['event_type'] == 'EPISODE_START' and first['event_seq'] == 0
            and first['sim_time_seconds'] == 0 and first['episode_id'] == RECORDING
            and first['config_sha256'] == job['config_sha256'], 'diagnostic native START differs')
    ledger_path = witness.ledger.ledger_path(suite)
    if not ledger_path.is_file():
        register(suite)
    name, suite_pin, rows = witness.ledger_rows(ledger_path)
    require(name == suite['campaign_id'] and suite_pin == witness.reference(CONFIG)
            and set(rows) == {RUN} and rows[RUN][0] == identity(job), 'diagnostic ledger differs')
    _, status, pin = rows[RUN]
    if status == 'PLANNED':
        transition(suite, job, 'PLANNED', 'CLAIMED')
        status = 'CLAIMED'
    return status, pin, dict(path=str(path), sha256=digest), events


def account(outcome, log, capture=None):
    raw = CONFIG.read_bytes()
    digest = sha256(raw).hexdigest()
    require(CONFIG.with_suffix('.sha256').read_text().strip() == digest, 'diagnostic config SHA differs')
    config = json.loads(raw)
    require(config['evidence_purpose'] == WORKLOAD and config['run_index'] == RUN
            and config['environment_seed'] == SEED and config['recording_id'] == RECORDING
            and witness.reference(TAPE)['sha256'] == config['diagnostic_prefix']['action_tape_sha256'],
            'diagnostic identity or tape differs')
    suite, job = job_for(digest)
    require(Path(job['event_directory']).is_dir(), 'native diagnostic has not started')
    with witness.ledger._launch_lock(OUTPUT / 'collector'):
        status, pin, manifest, path = started(suite, job)
        require(status in ('CLAIMED', 'UNKNOWN'), 'invalid native diagnostic settlement')
        if outcome is None:
            return dict(status=status, workload=WORKLOAD, manifest=manifest)
        count, last = last_event(path)
        require(last is not None and last['event_type'] != 'EPISODE_END',
                'partial diagnostic has a full-shift terminal event')
        result = dict(workload=WORKLOAD, status='UNKNOWN', reason=outcome, manifest=manifest,
                      event_stream=witness.reference(path), event_count=count,
                      last_event=last['event_type'], last_observed_seconds=last['sim_time_seconds'],
                      full_evaluation_completed=False, training_completed=False)
        if log is not None:
            result['native_diagnostic_log'] = witness.reference(log.resolve(strict=True))
        if capture is not None:
            captured = capture.resolve(strict=True)
            request = json.loads(captured.read_bytes())
            prefix = config['diagnostic_prefix']
            require(request['run_index'] == RUN and request['recording_id'] == RECORDING
                    and request['config_sha256'] == digest
                    and request['decision_sequence'] == prefix['expected_first_unavailable_decision_sequence']
                    and request['sim_time_seconds'] == prefix['expected_last_native_event_seconds'],
                    'captured decision differs from the frozen diagnostic boundary')
            result['captured_decision'] = witness.reference(captured)
            result['matched_native_prefix'] = verify_prefix()
        if status == 'UNKNOWN':
            require(json.loads(pin) == result, 'sealed partial diagnostic result differs')
        else:
            transition(suite, job, 'CLAIMED', 'UNKNOWN', result)
        write_fresh(HERE / 'diagnostic_status.json', canonical(result))
        return result


def verify_prefix():
    raw = CONFIG.read_bytes()
    digest = sha256(raw).hexdigest()
    require(CONFIG.with_suffix('.sha256').read_text().strip() == digest, 'diagnostic config SHA differs')
    config = json.loads(raw)
    prefix = config['diagnostic_prefix']
    expected = TAPE.read_bytes()
    require(sha256(expected).hexdigest() == prefix['action_tape_sha256'], 'diagnostic tape bytes differ')
    tape = [json.loads(line) for line in expected.splitlines()]
    _, job = job_for(digest)
    native = witness.native_manifest(Path(job['event_directory']), job)
    require(native is not None and not native[1]['complete'], 'expected only a started partial diagnostic')
    observed = native[0].with_name('events.jsonl')
    matched, applied, last = 0, 0, None
    with observed.open('rb') as stream:
        for sequence, line in enumerate(stream):
            event = json.loads(line)
            require(event['event_seq'] == sequence and event['episode_id'] == RECORDING
                    and event['config_sha256'] == digest, 'native diagnostic event identity differs')
            require(event['event_type'] != 'EPISODE_END', 'unexpected full-shift terminal event')
            last = event
            if event['event_type'] != 'ACTION_APPLIED':
                continue
            data = event['data']
            if applied < len(tape) and matched == applied:
                row = tape[applied]
                if all(row[key] == data[key] for key in ('decision_sequence', 'truck_id', 'action')) and (
                        abs(row['sim_time_seconds'] - event['sim_time_seconds']) <= 1e-6):
                    matched += 1
            applied += 1
    require(last is not None, 'native diagnostic event stream is empty')
    return dict(run_index=RUN, workload=WORKLOAD, expected_actions=len(tape), observed_applied_actions=applied,
                matched_prefix_actions=matched, last_event=last['event_type'],
                last_observed_seconds=last['sim_time_seconds'],
                original_last_event=prefix['expected_last_native_event'],
                original_last_observed_seconds=prefix['expected_last_native_event_seconds'],
                action_tape_sha256=prefix['action_tape_sha256'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    preparation = commands.add_parser('prepare')
    preparation.add_argument('--model', type=Path)
    preparation.add_argument('--experiment')
    commands.add_parser('mark-started')
    commands.add_parser('verify-prefix')
    settle = commands.add_parser('settle')
    settle.add_argument('--outcome', required=True)
    settle.add_argument('--log', type=Path)
    settle.add_argument('--capture', type=Path)
    args = parser.parse_args()
    if args.command == 'prepare':
        result = prepare(args.model, args.experiment)
    elif args.command == 'mark-started':
        result = account(None, None)
    elif args.command == 'verify-prefix':
        result = verify_prefix()
    else:
        result = account(args.outcome, args.log, args.capture)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
