"""Read-only history and effective accounting after H2 reservation cancellation.

Only separately approved new campaigns belong to the active ledger. Historical
H2 consumption is reported separately; no historical file is rewritten.
"""
import copy
import csv
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
from types import SimpleNamespace

from baseline1_v1 import budget as legacy
from grade_only_ppo_v1.gate import check_collisions as historical_collisions

ROOT = legacy.ROOT
HISTORY_PATH = legacy.GLOBAL_PATH
GLOBAL_PATH = ROOT / 'runs/active-budget-v2.json'
POLICY_PATH = ROOT / 'change_plans/budget_policy_20261109_001/policy.json'
H2_IDENTITY_REFERENCES = ({
    'path': str(ROOT / 'change_plans/stage8_g1_g4_staged_execution_001/budget_ledgers/job_reservations.csv'),
    'sha256': 'd5d23b2b928ce67ca9e086757d76dff88655b2f4a3bc537c851e798fe19b1ccc'},)
FORMAT = 'active-experiment-budget-v2'
require = legacy.require


def _signature(path):
    path = Path(path).resolve()
    info = path.stat()
    return str(path), info.st_size, info.st_mtime_ns, info.st_ctime_ns


@lru_cache(maxsize=64)
def _file_reference(signature):
    path = Path(signature[0])
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def reference(path):
    return dict(_file_reference(_signature(path)))


def _checked_reference(ref):
    require(isinstance(ref, dict) and set(ref) == {'path', 'sha256'} and
            isinstance(ref['path'], str) and Path(ref['path']).is_absolute() and
            isinstance(ref['sha256'], str) and re.fullmatch('[0-9a-f]{64}', ref['sha256']),
            'absolute budget evidence path/full SHA256 required')
    require(reference(ref['path']) == ref, 'budget evidence byte SHA256 changed')
    return ref


@lru_cache(maxsize=8)
def _history(signature):
    # Invoke the complete read-only historical validator, not its registering
    # constructor. Its _read never creates a lock, changes a file or approves work.
    return legacy.BudgetLedger._read(SimpleNamespace(path=Path(signature[0]), test_only=False))


@lru_cache(maxsize=8)
def _h2_identities(signature):
    identities = []
    with Path(signature[0]).open(newline='') as stream:
        for row in csv.DictReader(stream):
            run = row.get('predecessor_run_index')
            match = re.search(r'/dispatch-production-wait-shortfall-v2-run-([0-9]+)-seed-([0-9]+)$',
                              row.get('event_bundle', ''))
            require(run and match and int(run) == int(match.group(1)), 'historical H2 identity row differs')
            identities.append(dict(run_index=int(run), experiment_seed=int(match.group(2)),
                recording_id=row['recording_id'], event_bundle=row['event_bundle']))
    return identities


def sources(*, history_path=None, policy_path=None):
    history_path = HISTORY_PATH if history_path is None else Path(history_path)
    policy_path = POLICY_PATH if policy_path is None else Path(policy_path)
    history = _history(_signature(history_path))
    history_ref, policy_ref = reference(history_path), reference(policy_path)
    policy = json.loads(Path(policy_path).read_bytes())
    require(policy.get('schema') == 'active-budget-policy-v2' and
            policy.get('status') == 'H2_FUTURE_RESERVATION_CANCELLED' and
            type(policy.get('active_h2_reserved_shifts')) is int and policy['active_h2_reserved_shifts'] == 0 and
            policy.get('source_user_instruction') == 'H2不用预留了' and
            type(policy.get('new_experiment_shifts_approved')) is int and policy['new_experiment_shifts_approved'] == 0 and
            policy.get('legacy_budget') == history_ref,
            'explicit H2 cancellation policy and unchanged historical budget required')
    snapshot = policy.get('historical_h2_snapshot', {})
    require(snapshot.get('historical_snapshot_only') is True and
            snapshot.get('direct_live_ledger_verified') is False,
            'H2 consumption must remain an explicitly unverified historical snapshot')
    for evidence in snapshot.get('evidence', []):
        _checked_reference(evidence)
    identities = []
    refs = H2_IDENTITY_REFERENCES
    require(bool(refs), 'frozen historical H2 identity evidence required')
    for ref in refs:
        _checked_reference(ref)
        identities.extend(_h2_identities(_signature(ref['path'])))
    return history, policy, dict(history=history_ref, policy=policy_ref), identities


def budget_state_reference(*, history_path=None, policy_path=None, active_path=None):
    _, _, refs, _ = sources(history_path=history_path, policy_path=policy_path)
    active = GLOBAL_PATH if active_path is None else Path(active_path)
    return dict(refs, active=reference(active) if active.is_file() else None)


def validate_active(data, *, history, policy, refs, test_only):
    from .gate import check_approval
    require(data.get('format') == FORMAT and data.get('test_only') is test_only and
            data.get('legacy_budget') == refs['history'] and data.get('budget_policy') == refs['policy'] and
            data.get('legacy_reserved_shifts') == data.get('active_h2_reserved_shifts') == 0,
            'active budget schema, scope or immutable source references differ')
    campaigns, jobs = data.get('campaigns'), data.get('jobs')
    require(isinstance(campaigns, dict) and isinstance(jobs, dict), 'active campaign/job inventory required')
    historical_cap = history['additional_reserved_shifts']
    cap = sum(c['cap'] for c in campaigns.values())
    require(data.get('historical_non_h2_reserved_shifts') == historical_cap and
            data.get('new_reserved_shifts') == cap == len(jobs) and
            data.get('additional_reserved_shifts') == data.get('maximum_total_shifts') == historical_cap + cap,
            'effective non-H2 active budget accounting differs')
    planned, batch_ids = {}, set()
    for cid, entry in campaigns.items():
        spec = entry['campaign']
        legacy._hash(entry['campaign_sha256'], 'campaign_sha256')
        legacy._hash(entry['approval_sha256'], 'approval_sha256')
        check_approval(entry['approval'], spec, entry['campaign_sha256'])
        require(cid == spec['campaign_id'] and entry['cap'] == spec['new_shift_cap'] == len(spec['jobs']) == 576 and
                entry['campaign_identity_sha256'] == legacy._digest(spec) and
                entry['approval_identity_sha256'] == legacy._digest(entry['approval']),
                'active registered campaign/approval identity changed')
        check_collisions(spec, history)
        indices = sorted(int(u) for u in entry['commits'])
        require(indices == list(range(1, len(indices) + 1)), 'active committed update sequence differs')
        committed = set()
        for index in indices:
            commit = entry['commits'][str(index)]
            require(commit['commit_sha256'] == legacy._digest({k:v for k,v in commit.items() if k != 'commit_sha256'}) and
                    commit['batch_id'] not in batch_ids and not (set(commit['job_ids']) & committed),
                    'active checkpoint digest or duplicate batch/jobs differs')
            batch_ids.add(commit['batch_id']); committed.update(commit['job_ids'])
            for key in commit['job_ids']:
                require(key in jobs and jobs[key]['campaign_id'] == cid and jobs[key]['status'] == 'COMPLETE' and
                        jobs[key].get('committed_update') == index, 'active committed physical state differs')
        for job in spec['jobs']:
            key = str(job['run_index'])
            require(key not in planned, 'duplicate active canonical job')
            planned[key] = cid, job
    require(set(planned) == set(jobs), 'active global job inventory differs')
    for key, row in jobs.items():
        cid, identity = planned[key]
        require(row['campaign_id'] == cid and row['identity'] == identity and
                row['identity_sha256'] == legacy._digest(identity) and
                row['status'] in ('PLANNED', 'CLAIMED', 'COMPLETE', 'UNKNOWN'), 'active canonical job/state changed')
        if row['status'] != 'PLANNED':
            require(isinstance(row.get('claim'), dict) and row['claim_sha256'] == legacy._digest(row['claim']),
                    'active physical claim receipt changed')
        if row['status'] == 'COMPLETE':
            require(row['completion_sha256'] == legacy._digest(dict(dataset_ref=row['dataset_ref'], receipt=row['receipt'])),
                    'active completion receipt changed')
    return data


def read_active(path, *, history, policy, refs, test_only=False):
    data = json.loads(Path(path).read_bytes())
    checksum = data.pop('document_sha256', None)
    require(checksum == legacy._digest(data), 'active budget checksum changed')
    return validate_active(data, history=history, policy=policy, refs=refs, test_only=test_only)


def budget_snapshot(*, history_path=None, policy_path=None, active_path=None, test_only=False):
    history, policy, refs, identities = sources(history_path=history_path, policy_path=policy_path)
    path = GLOBAL_PATH if active_path is None else Path(active_path)
    active = read_active(path, history=history, policy=policy, refs=refs, test_only=test_only) if path.exists() else None
    result = dict(format=FORMAT, legacy_reserved_shifts=0, active_h2_reserved_shifts=0,
        historical_non_h2_reserved_shifts=history['additional_reserved_shifts'], new_reserved_shifts=0,
        additional_reserved_shifts=history['additional_reserved_shifts'], maximum_total_shifts=history['additional_reserved_shifts'],
        campaigns=dict(history['campaigns']), jobs=dict(history['jobs']), recovery_history=history.get('recovery_history', {}),
        historical_h2_snapshot=copy.deepcopy(policy['historical_h2_snapshot']), historical_h2_identities=identities,
        active_scope='non-H2 approved campaigns only; historical H2 consumption is separate',
        budget_policy=refs['policy'], legacy_budget=refs['history'])
    if active:
        require(set(result['campaigns']).isdisjoint(active['campaigns']) and set(result['jobs']).isdisjoint(active['jobs']),
                'active and historical campaign/job identity collision')
        result['campaigns'].update(active['campaigns']); result['jobs'].update(active['jobs'])
        for key in ('new_reserved_shifts', 'additional_reserved_shifts', 'maximum_total_shifts'):
            result[key] = active[key]
    return result


def check_collisions(campaign, budget):
    historical_collisions(campaign, budget)
    for key in ('run_index', 'experiment_seed', 'actor_sampling_seed', 'recording_id'):
        prior = {row.get(key) for row in budget.get('historical_h2_identities', [])} - {None}
        require(all(job.get(key) not in prior for job in campaign['jobs']), 'historical H2 identity collision: ' + key)
