"""Collect frozen s01 GAT1 evaluation against ten settled grade-control shifts."""
from collections import Counter
import fcntl
import json
import os
from pathlib import Path

import numpy as np

from paired_campaign import (
    CAMPAIGN, CONTROL, CORE, COUNT, EXECUTION, LAUNCHER, LEDGER, LOCKS, METHOD, OUTPUT,
    POLICY, PYTHON, REPORT, ROOT, SCHEMA, SUITE, Ledger, _checked_reference, _publish_bytes,
    _publish_json, _require, check_fresh, load_actor, load_anchors, load_reference, load_suite,
    preceding_snapshot, reference,
)
import grade_launch as original
from formal_report import paired_report
from traditional_dispatch_v2.runtime import _verify_anchor
from weak_plan import bind


LANE = LOCKS / f"port-{EXECUTION['port']}.lock"
AUTHORITY = OUTPUT / 'launch' / 'run_authority.json'
INTENT = OUTPUT / 'launch' / 'intent.json'
STATUS = OUTPUT / 'run_status.json'
STOP = CONTROL / 'STOP.properties'
IDENTITY_FIELDS = ('run_index', 'experiment_seed', 'recording_id')
GRADE_POLICY = 'dispatch_flow_grade_control'

status = bind(original.status, STATUS=STATUS, CAMPAIGN=CAMPAIGN, COUNT=COUNT, SUITE=SUITE)
_set_status = bind(original._set_status, STATUS=STATUS, CAMPAIGN=CAMPAIGN, COUNT=COUNT)
_controller_command = bind(original._controller_command, PYTHON=PYTHON, LAUNCHER=LAUNCHER, SUITE=SUITE)
_check_launch_card = bind(original._preflight, OUTPUT=OUTPUT, REPORT=REPORT, SUITE=SUITE, CAMPAIGN=CAMPAIGN,
                          check_fresh=check_fresh, preceding_snapshot=preceding_snapshot)


def _preflight(suite):
    result = _check_launch_card(suite)
    anchors = load_anchors()
    _require(len(anchors) == COUNT, 'ten sealed physical grade-control anchors required')
    for job, anchor in zip(suite['jobs'], anchors, strict=True):
        _check_pair(job, anchor)
    actor, checked_digest = load_actor(suite)
    _require(checked_digest == suite['actor_digest'] and not actor.training and
             all(not parameter.requires_grad for parameter in actor.parameters()), 'Actor is not frozen')
    return dict(result, actor=suite['actor'], actor_digest=checked_digest, grade_anchors_verified=len(anchors))


_from_anylogic = bind(original._from_anylogic, _preflight=_preflight, LANE=LANE, EXECUTION=EXECUTION,
                      AUTHORITY=AUTHORITY, OUTPUT=OUTPUT, SCHEMA=SCHEMA, CAMPAIGN=CAMPAIGN, COUNT=COUNT,
                      INTENT=INTENT, ROOT=ROOT, _controller_command=_controller_command)


def _check_pair(job, anchor):
    _require(job['paired_reference_run_index'] == anchor['run_index']
             and job['experiment_seed'] == anchor['experiment_seed']
             and job['scene_id'] == anchor['scene_id']
             and job['run_index'] != anchor['run_index']
             and job['recording_id'] != anchor['recording_id']
             and job['actor_sampling_seed'] > 0 and anchor['actor_sampling_seed'] == 0
             and job['actor_sampling_seed_consumed'] is True,
             'new physical identity or grade anchor differs from the frozen pair')
    _require(load_reference(job['paired_anchor']) == anchor, 'grade anchor receipt bytes differ from the sealed pair')


def _run_shift(suite, ledger, job, anchor, actor, actor_digest):
    from baseline1_v1.collection import wait_for_java_start
    from business_graph_summary_v1.models import digest
    from python.dispatch_bridge import DispatchCallbackBridge
    from python.dispatch_env import DispatchEnv
    from python.dispatch_service import create_server
    from python.episode_runner import ParameterVariationEpisodeRunner
    from traditional_dispatch_v1.runtime import _publish_ready
    from traditional_dispatch_v2.runtime import _collect

    _check_pair(job, anchor)
    run = job['run_index']
    startup = dict(campaign_id=CAMPAIGN, control_directory=str(CONTROL),
                   master_plan_sha256=suite['master_plan']['sha256'])
    output = Path(job['python_output']) / 'evaluation_receipt.json'
    claim = CONTROL / 'java_claims' / f'run_{run}.once.claim'
    ready = CONTROL / 'ready' / f'run_{run}.properties'
    _require(not any(path.exists() for path in (output, claim, ready, STOP)), 'run identity already has evidence')
    method = load_reference(job['phase_config'])['method']
    timeout = EXECUTION['callback_timeout_seconds']
    bridge = DispatchCallbackBridge(expected_episode_metadata=[(run, job['experiment_seed'])])
    env = DispatchEnv(bridge, ParameterVariationEpisodeRunner(1), timeout)
    server = create_server(bridge, host=EXECUTION['host'], port=EXECUTION['port'])
    claimed = False
    try:
        server.start()
        ledger.transition(job, 'PLANNED', 'CLAIMED')
        claimed = True
        _publish_ready(startup, job)
        wait_for_java_start(startup, job, job['phase_config']['sha256'], timeout)
        # The collector's selector names the common frozen-Actor path "bg128".
        # Artifact, published receipt and ledger retain the actual GAT1 identity.
        result = _collect(env, dict(job, policy_kind='bg128'), method=method, anchor=anchor, actor=actor)
        engineering = result['engineering']
        sampling = engineering['actor_sampling_rng']
        _require(sampling['draws'] == engineering['decisions'] > 0 and
                 sampling['initial_sha256'] != sampling['final_sha256'], 'Actor sampling evidence differs from decisions')
        _require(digest(actor) == actor_digest and not actor.training
                 and all(not parameter.requires_grad for parameter in actor.parameters()),
                 'frozen Actor changed during independent inference')
        _checked_reference(suite['protected_core'])
        _checked_reference(job['paired_anchor'])
        receipt = dict(result, schema=SCHEMA + '-receipt', campaign_id=CAMPAIGN, status='COMPLETE',
            normal_completion=True, completed_shifts=1, optimizer_updates=0, java_start_claim=reference(claim),
            actor=suite['actor'], actor_digest=actor_digest,
            **{key: job[key] for key in (*IDENTITY_FIELDS, 'actor_sampling_seed', 'policy_kind', 'scene_id',
                                       'pair_ordinal', 'paired_anchor', 'paired_reference_run_index', 'phase_config')})
        ref = _publish_json(output, receipt)
        ledger.transition(job, 'CLAIMED', 'COMPLETE', ref)
        claimed = False
        return receipt, ref
    except BaseException as error:
        if claimed:
            ledger.transition(job, 'CLAIMED', 'UNKNOWN', dict(reason=repr(error)))
        raise
    finally:
        env.close()
        server.stop(grace=0).wait()


def _summarize(shifts, anchors, jobs):
    _require(len(shifts) == len(anchors) == len(jobs) == COUNT, 'all ten paired shifts are required')
    _require(len({row['experiment_seed'] for row in shifts}) == COUNT, 'environment seed repeats between slots')
    _require(len({row['actor_sampling_seed'] for row in shifts}) == COUNT, 'Actor seed repeats between slots')
    pairs = []
    for slot, (row, anchor, job) in enumerate(zip(shifts, anchors, jobs, strict=True)):
        _check_pair(job, anchor)
        _require(row['pair_ordinal'] == job['pair_ordinal'] == slot
                 and all(row[key] == job[key] for key in (*IDENTITY_FIELDS, 'actor_sampling_seed', 'phase_config',
                                                          'policy_kind', 'paired_anchor', 'paired_reference_run_index'))
                 and row['normal_completion'] is True and row['optimizer_updates'] == 0,
                 'paired summary received a different or incomplete frozen GAT1 shift')
        _verify_anchor(row, anchor)
        actor, control = row['physical'], anchor['physical']
        pairs.append(dict(pair_ordinal=slot, environment_seed=row['experiment_seed'],
            actor_sampling_seed=row['actor_sampling_seed'], gat1_run=row['run_index'], grade_run=anchor['run_index'],
            gat1_complete=actor['complete_task'], grade_complete=control['complete_task'],
            difference={key: float(actor[key]) - float(control[key]) for key in paired_report.METRICS},
            windows=[dict(window=right['window'], gat1_grade=left['grade'], control_grade=right['grade'],
                          gat1_compliant=left['compliant'], control_compliant=right['compliant'],
                          difference_pp=None if left['grade'] is None or right['grade'] is None
                          else 100.0 * (left['grade'] - right['grade']))
                     for left, right in zip(actor['windows'], control['windows'], strict=True)]))
    groups = {POLICY: shifts, GRADE_POLICY: anchors}
    draws = np.random.default_rng(paired_report.ANALYSIS_SEED).integers(
        0, COUNT, size=(paired_report.BOOTSTRAP_REPLICATES, COUNT))
    contrast = paired_report._matched_comparison(groups, POLICY, GRADE_POLICY, draws)
    contrast['primary'] = True
    return dict(
        evaluation_slots=COUNT, primary_outcome=f'mean_paired_combined_cost_{POLICY}_minus_{GRADE_POLICY}',
        groups={name: paired_report._summarize_arm(records) for name, records in groups.items()},
        engineering_acceptance={name: dict(passed=sum(row['engineering_acceptance']['passed'] for row in records),
                                           shifts=len(records)) for name, records in groups.items()},
        comparison=contrast, pairs=pairs,
        intervals=dict(method='percentile_bootstrap_over_environment_seed_slots',
                       replicates=paired_report.BOOTSTRAP_REPLICATES, seed=paired_report.ANALYSIS_SEED, level=.95),
    )


def _controller(suite, fd):
    _require(os.path.samestat(os.fstat(fd), LANE.stat()), 'inherited lock does not belong to the RPC lane')
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    authority = reference(AUTHORITY)
    intent = json.loads(INTENT.read_bytes())
    _require(not LEDGER.exists() and intent == dict(authority=authority, command=_controller_command(fd)),
             'launch intent or physical registration differs')
    shifts, references = [], []
    ledger = Ledger(suite)
    try:
        _set_status('STARTING', completed_shifts=0)
        actor, actor_digest = load_actor(suite)
        anchors = load_anchors()
        _require(actor_digest == suite['actor_digest'] and len(anchors) == COUNT, 'actor or pair inventory differs')
        for job, anchor in zip(suite['jobs'], anchors, strict=True):
            _check_pair(job, anchor)
        ledger.register(authority)
        for job, anchor in zip(suite['jobs'], anchors, strict=True):
            _set_status('RUNNING', completed_shifts=len(shifts), current_run_index=job['run_index'])
            receipt, ref = _run_shift(suite, ledger, job, anchor, actor, actor_digest)
            shifts.append(receipt)
            references.append(ref)
            print(f"COMPLETE run_index={job['run_index']} {len(shifts)}/{COUNT}", flush=True)
        _, slots = ledger.snapshot()
        _require(Counter(row['status'] for row in slots.values()) == {'COMPLETE': COUNT}
                 and [row['evidence'] for row in slots.values()] == references, 'physical ledger and receipts differ')
        _checked_reference(suite['protected_core'])
        report = _publish_json(REPORT, dict(
            schema=SCHEMA + '-report', campaign_id=CAMPAIGN, status='COMPLETE', physical_shifts=COUNT,
            optimizer_updates=0, actor=suite['actor'], actor_digest=actor_digest, method=METHOD,
            protected_core=reference(CORE), receipts=references, shifts=shifts,
            **_summarize(shifts, anchors, suite['jobs']),
        ))
        result = dict(status='COMPLETE', campaign_id=CAMPAIGN, physical_shifts=COUNT, completed_shifts=COUNT,
                      training_shifts=0, optimizer_updates=0, suite=reference(SUITE), evaluation=report)
        _publish_json(OUTPUT / 'suite_complete.json', result)
        _set_status('COMPLETE', completed_shifts=COUNT, evaluation=report)
        return result
    except BaseException as error:
        if not STOP.exists():
            reason = str(error).replace('\n', ' ').replace('\r', ' ')
            _publish_bytes(STOP, ('reason=' + reason + '\n').encode('ascii', 'backslashreplace'))
        _set_status('STOP', completed_shifts=len(shifts), reason=repr(error))
        raise


main = bind(original.main, SUITE=SUITE, status=status, load_suite=load_suite, _preflight=_preflight,
            _from_anylogic=_from_anylogic, _controller=_controller)

if __name__ == '__main__':
    main()
