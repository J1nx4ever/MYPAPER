"""Resume the frozen BG128 campaign with one approved physical replacement."""
import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess

from recovery_plan import (
    FAILED, OFFSET, RECOVERY, RECOVERY_CONTROL, RECOVERY_OUTPUT, RECOVERY_REPORT,
    REMAINING, REPLACEMENT, ROOT, check_start, load_recovery,
)
from mixed_plan import CAMPAIGN, CORE, LEDGER, OUTPUT, SUITE, TOTAL, UPDATES, read, reference, require
from recovery_ledger import ReplacementLedger, replacement_registration
from recovery_002.storage_scope import bg128_storage
from transport import per_job_transport, verify_replacement_learner
from business_graph_summary_campaign_v1.health import _process_identity
from business_graph_summary_campaign_v1.launch import _exclusive_json, _launch_lock
from business_graph_summary_campaign_v1.plan_io import encoded
from grade_only_ppo_v2.runtime import _controller_lock
from online_v1_1.operations import _publish

PYTHON = '/home/jinx/miniconda3/envs/anylogic-env/bin/python'
LAUNCH = RECOVERY_OUTPUT / 'launch'
AUTHORITY = LAUNCH / 'run_authority.json'
INTENT = LAUNCH / 'launch_intent.json'
RECEIPT = LAUNCH / 'launch_receipt.json'
LOG = LAUNCH / 'controller.log'
STOP = RECOVERY_CONTROL / 'STOP.properties'
STATUS = RECOVERY_OUTPUT / 'run_status.json'


def _authority(manifest):
    return dict(schema='bg128-continuous-mixed-replacement-authority-v1', campaign_id=CAMPAIGN,
                original_suite=reference(SUITE), replacement_ref=manifest['replacement_config'],
                replacement_run_index=REPLACEMENT, failed_run_index=FAILED, additional_shifts=1,
                authorized_by_anylogic_run=True)


def _intent():
    return dict(recovery=reference(RECOVERY), authority=reference(AUTHORITY),
                command=[PYTHON, '-B', str(Path(__file__).resolve()), '--controller'],
                completed_before=OFFSET, remaining_shifts=REMAINING, additional_shifts=1)


def _write_status(status, **fields):
    data = dict(schema='bg128-mixed-cross-shift-recovery-progress-v1', campaign_id=CAMPAIGN,
                status=status, scheduled_complete_shifts=TOTAL, retained_unknown_run=FAILED,
                updated_at_utc=datetime.now(timezone.utc).isoformat(), **fields)
    _publish(STATUS, encoded(data), replace=True)


def _recheck_complete(state, suite):
    import torch
    from business_graph_summary_campaign_v1.collection import load_admitted_episode
    from mixed_runtime import _exact_verification, _verify_observation_sources

    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(suite['cpu_threads'])
        jobs = suite['jobs'][:OFFSET]
        with bg128_storage(), _exact_verification(jobs):
            for job in jobs:
                evidence = state['jobs'][str(job['run_index'])]
                episode = load_admitted_episode(evidence['receipt'])
                _verify_observation_sources(episode, job)
                require(len(episode['memory'].settlements) == 4, 'completed shift lacks four windows')
    finally:
        torch.set_num_threads(threads)


@contextmanager
def continued_batch(original, configs, replacement, replacement_ref):
    """Keep the original optimizer and make the first batch span its two sealed configs."""
    import mixed_runtime as runtime

    collect = runtime._collect
    preflight = runtime.preflight
    verify = runtime.verify_learner_behavior
    first = configs[0]
    replacement_job, = replacement['jobs']
    first_completed = dict(first, jobs=first['jobs'][:OFFSET])
    first_remaining = dict(first, jobs=first['jobs'][OFFSET+1:])

    def collect_split(suite, cfg, phase_ref, checkpoint, ledger, model, *, actor_ref=None):
        if cfg['phase_id'] != first['phase_id']:
            return collect(suite, cfg, phase_ref, checkpoint, ledger, model, actor_ref=actor_ref)
        require(cfg['jobs'] == first_completed['jobs'] + [replacement_job] + first_remaining['jobs'],
                'first effective batch order differs')
        old, old_refs, shared_actor = collect(suite, first_completed, phase_ref, checkpoint, ledger,
                                               model, actor_ref=actor_ref)
        setup = dict(suite, phase_configs=[*suite['phase_configs'], replacement_ref])
        extra, extra_refs, _ = collect(setup, replacement, replacement_ref, checkpoint, ledger,
                                      model, actor_ref=shared_actor)
        remaining, remaining_refs, _ = collect(suite, first_remaining, phase_ref, checkpoint, ledger,
                                              model, actor_ref=shared_actor)
        return old + extra + remaining, old_refs + extra_refs + remaining_refs, shared_actor

    def checked_preflight(suite, effective):
        require(effective[0]['jobs'] == first_completed['jobs'] + [replacement_job] + first_remaining['jobs']
                and effective[1:] == configs[1:], 'effective runtime phase inventory differs')
        return preflight(original, configs)

    def verify_effective(model, episode, current, job):
        verifier = verify_replacement_learner if job['run_index'] == REPLACEMENT else verify
        return verifier(model, episode, current, job)

    runtime._collect = collect_split
    runtime.preflight = checked_preflight
    runtime.verify_learner_behavior = verify_effective
    try:
        yield
    finally:
        runtime._collect = collect
        runtime.preflight = preflight
        runtime.verify_learner_behavior = verify


def start(manifest, suite, configs, replacement):
    with _launch_lock(LAUNCH):
        require(not any(path.exists() for path in (AUTHORITY, INTENT, RECEIPT, LOG, RECOVERY_CONTROL, STATUS)),
                'existing continuation requires review')
        require(replacement_registration() is None, 'replacement already registered')
        check_start(manifest, suite, configs, replacement)
        execution = configs[0]['execution']
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind((execution['host'], execution['port']))
        _exclusive_json(AUTHORITY, _authority(manifest))
        intent = _intent()
        _exclusive_json(INTENT, intent)
        with LOG.open('xb', buffering=0) as log:
            child = subprocess.Popen(intent['command'], cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True,
                env=dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1'))
        identity = _process_identity(child.pid)
        _exclusive_json(RECEIPT, dict(recovery=intent['recovery'], pid=child.pid,
            process_start_ticks=identity.get('start_ticks'), log=str(LOG)))
    return dict(status='CONTINUOUS_MIXED_REPLACEMENT_CONTROLLER_STARTED', pid=child.pid,
                first=REPLACEMENT, remaining_shifts=REMAINING, maximum_additional_shifts=1)


def controller():
    from mixed_runtime import run

    manifest, suite, configs, replacement = load_recovery()
    require(read(AUTHORITY) == _authority(manifest) and read(INTENT) == _intent(),
            'operator replacement authority or launch intent differs')
    with _controller_lock(OUTPUT):
        require(not STOP.exists() and not STATUS.exists(), 'previous replacement continuation requires review')
        _write_status('STARTING', first_run_index=REPLACEMENT)
        try:
            state = check_start(manifest, suite, configs, replacement)
            _recheck_complete(state, suite)
            archive = RECOVERY_REPORT / 'ledger_before.sqlite'
            archive.parent.mkdir(parents=True, exist_ok=True)
            _publish(archive, LEDGER.read_bytes())
            require(reference(archive)['sha256'] == manifest['ledger_before']['sha256'], 'ledger archive differs')
            ref = manifest['replacement_config']
            ledger = ReplacementLedger(suite, configs, replacement, ref, approval=reference(AUTHORITY))
            effective = dict(configs[0], jobs=configs[0]['jobs'][:OFFSET] + replacement['jobs'] +
                             configs[0]['jobs'][OFFSET + 1:])
            active_jobs = suite['jobs'][:OFFSET] + replacement['jobs'] + suite['jobs'][OFFSET + 1:]
            active = dict(suite, jobs=active_jobs, master_plan=manifest['master'],
                          control_directory=str(RECOVERY_CONTROL))
            with bg128_storage(), per_job_transport(), continued_batch(suite, configs, replacement, ref):
                run(active, [effective, *configs[1:]], ledger)
            state = ledger.snapshot()
            require(Counter(row['status'] for row in state['jobs'].values()) == {'COMPLETE': TOTAL, 'UNKNOWN': 1}
                    and set(state['campaigns'][CAMPAIGN]['commits']) == {str(i) for i in range(1, UPDATES+1)}
                    and state['jobs'][str(FAILED)]['status'] == 'UNKNOWN'
                    and reference(CORE) == suite['protected_core'], 'completion or protected PPO bytes differ')
            _write_status('COMPLETE', completed_shifts=TOTAL, training_updates=UPDATES, additional_shifts=1)
        except BaseException as error:
            RECOVERY_CONTROL.mkdir(parents=True, exist_ok=True)
            if not STOP.exists():
                _publish(STOP, ('reason=' + repr(error).replace('\n',' ') + '\n').encode())
            _write_status('STOP', reason=repr(error))
            raise


def read_status(manifest, suite, configs, replacement):
    ledger = ReplacementLedger(suite, configs, replacement, manifest['replacement_config'])
    state = ledger.snapshot()
    value = dict(campaign_id=CAMPAIGN, status='PREPARED_OFFLINE_ONLY', first_eligible_run=REPLACEMENT,
                 registered_status_counts=dict(Counter(r['status'] for r in state['jobs'].values())),
                 committed_training_updates=len(state['campaigns'][CAMPAIGN]['commits']),
                 replacement_registered=replacement_registration() is not None)
    if STATUS.is_file():
        value.update(read(STATUS))
        current = OUTPUT / 'training_progress.json'
        if current.is_file():
            value['training_progress'] = read(current)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=lambda value: Path(value).resolve(), default=RECOVERY)
    parser.add_argument('--run-index', type=int)
    parser.add_argument('--environment-seed', type=int)
    parser.add_argument('--recording-id')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--status', action='store_true')
    mode.add_argument('--from-anylogic-run', action='store_true')
    mode.add_argument('--controller', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    require(args.suite == RECOVERY, 'unexpected recovery descriptor')
    if args.controller:
        controller()
        return
    manifest, suite, configs, replacement = load_recovery()
    if args.from_anylogic_run:
        job = replacement['jobs'][0]
        require((args.run_index, args.environment_seed, args.recording_id) ==
                (REPLACEMENT, job['experiment_seed'], job['recording_id']), 'unexpected first replacement identity')
        result = start(manifest, suite, configs, replacement)
    elif args.status:
        result = read_status(manifest, suite, configs, replacement)
    else:
        state = check_start(manifest, suite, configs, replacement)
        _recheck_complete(state, suite)
        result = dict(status='PASS_OFFLINE_NO_LISTENER', recovery=reference(RECOVERY), retained_unknown=FAILED,
                      first=REPLACEMENT, remaining_runs=REMAINING, planned_additional_slots=1)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == '__main__':
    main()
