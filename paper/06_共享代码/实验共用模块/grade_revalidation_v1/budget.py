"""Separate, atomic evaluation capacity; all preceding ledgers remain read only."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

from baseline1_v1 import budget as legacy
from grade_multiobjective_ppo_v1 import budget_policy as prior

ROOT = prior.ROOT
GLOBAL_PATH = ROOT / 'runs/independent-evaluation-budget-v1.json'
FORMAT = 'independent-evaluation-budget-v1'
require = legacy.require


def _sources():
    snapshot = prior.budget_snapshot()
    require(prior.GLOBAL_PATH.is_file(), 'preceding active budget must exist')
    before = prior.reference(prior.GLOBAL_PATH)
    require(snapshot['legacy_reserved_shifts'] == snapshot['active_h2_reserved_shifts'] == 0,
            'cancelled H2 policy required')
    return snapshot, before, snapshot['budget_policy']


def budget_state_reference(path=None):
    """Read ordinary files only; never create a ledger, lock or reservation."""
    _, before, policy = _sources()
    path = GLOBAL_PATH if path is None else Path(path)
    return dict(budget_before=before, budget_policy=policy,
                evaluation_budget=prior.reference(path) if path.is_file() else None)


def budget_snapshot(path=None, *, test_only=False):
    source, _, _ = _sources()
    path = GLOBAL_PATH if path is None else Path(path)
    result = copy.deepcopy(source)
    if not path.exists():
        return result
    view = SimpleNamespace(path=path, test_only=test_only)
    data = BudgetLedger._read(view)
    require(set(result['campaigns']).isdisjoint(data['campaigns']) and
            set(result['jobs']).isdisjoint(data['jobs']), 'evaluation/history identity collision')
    result['campaigns'].update(data['campaigns'])
    result['jobs'].update(data['jobs'])
    result['evaluation_reserved_shifts'] = data['evaluation_reserved_shifts']
    for key in ('additional_reserved_shifts', 'maximum_total_shifts'):
        result[key] = data[key]
    return result


class BudgetLedger(legacy.BudgetLedger):
    """Use the tested physical state machine, while making PPO commits invalid."""
    def __init__(self, path, suite, *, campaign_sha256, approval, approval_sha256,
                 test_only=False):
        from .gate import check_approval, check_sources, SUITE_CONTRACT, RUN_COUNT
        require(suite.get('contract') == SUITE_CONTRACT and
                suite.get('kind') == 'independent_evaluation' and
                suite.get('new_shift_cap') == len(suite.get('jobs', [])) == RUN_COUNT == 64,
                'exact64 fixed-checkpoint independent evaluation required')
        require(all(j.get('partition') == 'validation' and j.get('role') == 'final' and
                    j.get('checkpoint_update') == 8 and j.get('target_training_update') == 0
                    for j in suite['jobs']), 'evaluation slots cannot request optimizer updates')
        legacy._hash(campaign_sha256, 'campaign_sha256')
        legacy._hash(approval_sha256, 'approval_sha256')
        check_approval(approval, suite, campaign_sha256)  # before mkdir or lock
        check_sources(suite['source_pins'])
        require(type(test_only) is bool, 'explicit boolean test_only required')
        self.path = Path(path).resolve()
        self.test_only = test_only
        require(self.path != GLOBAL_PATH.resolve() if test_only else self.path == GLOBAL_PATH.resolve(),
                'production/test evaluation ledger scope mismatch')
        self.lock_path = self.path.with_name(self.path.name + '.lock')
        self.campaign_id = suite['campaign_id']
        source, before, policy = _sources()
        require(suite.get('budget_before') == before and suite.get('budget_policy') == policy and
                suite.get('legacy_reserved_shifts') == suite.get('active_h2_reserved_shifts') == 0,
                'unchanged preceding budget and cancelled H2 policy required')
        entry = dict(campaign=copy.deepcopy(suite), campaign_sha256=campaign_sha256,
                     campaign_identity_sha256=legacy._digest(suite), approval=copy.deepcopy(approval),
                     approval_sha256=approval_sha256, approval_identity_sha256=legacy._digest(approval),
                     cap=64, registered_at=legacy._now(), commits={})
        with self._locked():
            current_source, current_before, current_policy = _sources()
            require(current_source == source and current_before == before and current_policy == policy,
                    'preceding budget changed before atomic evaluation registration')
            if self.path.exists():
                data = self._read()
            else:
                data = dict(format=FORMAT, test_only=test_only, budget_before=before,
                            budget_policy=policy, legacy_reserved_shifts=0, active_h2_reserved_shifts=0,
                            preceding_reserved_shifts=source['maximum_total_shifts'],
                            evaluation_reserved_shifts=0, additional_reserved_shifts=source['maximum_total_shifts'],
                            maximum_total_shifts=source['maximum_total_shifts'], campaigns={}, jobs={}, revision=0)
            if self.campaign_id in data['campaigns']:
                old = data['campaigns'][self.campaign_id]
                require(all(old[k] == entry[k] for k in ('campaign_sha256', 'campaign_identity_sha256',
                            'approval_sha256', 'approval_identity_sha256', 'cap')),
                        'evaluation re-registration differs; no extra capacity granted')
                require(data['maximum_total_shifts'] == suite['maximum_total_if_approved'],
                        'registered evaluation cap changed')
                return
            prior.check_collisions(suite, source)
            prior.check_collisions(suite, dict(source, campaigns=data['campaigns'], jobs=data['jobs']))
            require(suite.get('evaluation_budget_before') ==
                    (prior.reference(self.path) if self.path.is_file() else None),
                    'evaluation ledger changed since plan freeze')
            require(data['maximum_total_shifts'] + 64 == suite['maximum_total_if_approved'],
                    'exact approved64 additional evaluation cap differs')
            data['campaigns'][self.campaign_id] = entry
            for job in suite['jobs']:
                data['jobs'][str(job['run_index'])] = dict(campaign_id=self.campaign_id,
                    identity=copy.deepcopy(job), identity_sha256=legacy._digest(job), status='PLANNED')
            data['evaluation_reserved_shifts'] += 64
            data['additional_reserved_shifts'] += 64
            data['maximum_total_shifts'] += 64
            data['revision'] += 1
            self._write(data)

    def _read(self):
        from .gate import check_approval, SUITE_CONTRACT
        source, before, policy = _sources()
        data = json.loads(self.path.read_bytes())
        checksum = data.pop('document_sha256', None)
        require(checksum == legacy._digest(data), 'evaluation budget checksum changed')
        require(data.get('format') == FORMAT and data.get('test_only') is self.test_only and
                data.get('budget_before') == before and data.get('budget_policy') == policy and
                data.get('legacy_reserved_shifts') == data.get('active_h2_reserved_shifts') == 0,
                'evaluation ledger scope or preceding budget changed')
        campaigns, jobs = data.get('campaigns'), data.get('jobs')
        require(isinstance(campaigns, dict) and isinstance(jobs, dict), 'canonical evaluation inventory required')
        cap = 64 * len(campaigns)
        require(data.get('preceding_reserved_shifts') == source['maximum_total_shifts'] and
                data.get('evaluation_reserved_shifts') == len(jobs) == cap and
                data.get('additional_reserved_shifts') == data.get('maximum_total_shifts') ==
                source['maximum_total_shifts'] + cap, 'evaluation accounting differs')
        planned = {}
        for cid, entry in campaigns.items():
            spec = entry['campaign']
            check_approval(entry['approval'], spec, entry['campaign_sha256'])
            require(spec.get('contract') == SUITE_CONTRACT and spec.get('kind') == 'independent_evaluation' and
                    cid == spec['campaign_id'] and entry['cap'] == spec['new_shift_cap'] == len(spec['jobs']) == 64 and
                    entry.get('commits') == {} and entry['campaign_identity_sha256'] == legacy._digest(spec) and
                    entry['approval_identity_sha256'] == legacy._digest(entry['approval']) and
                    spec.get('budget_before') == before and spec.get('budget_policy') == policy,
                    'evaluation campaign/approval or zero-update identity changed')
            legacy._hash(entry['campaign_sha256'], 'campaign_sha256')
            legacy._hash(entry['approval_sha256'], 'approval_sha256')
            prior.check_collisions(spec, source)
            for job in spec['jobs']:
                key = str(job['run_index'])
                require(key not in planned and job.get('partition') == 'validation' and
                        job.get('role') == 'final' and job.get('target_training_update') == 0 and
                        job.get('checkpoint_update') == 8, 'evaluation canonical job/update changed')
                planned[key] = (cid, job)
        require(set(planned) == set(jobs), 'evaluation physical inventory differs')
        for key, row in jobs.items():
            cid, identity = planned[key]
            require(row['campaign_id'] == cid and row['identity'] == identity and
                    row['identity_sha256'] == legacy._digest(identity) and
                    row['status'] in ('PLANNED', 'CLAIMED', 'COMPLETE', 'UNKNOWN') and
                    'committed_update' not in row, 'evaluation slot state or identity changed')
            if row['status'] != 'PLANNED':
                require(isinstance(row.get('claim'), dict) and
                        row.get('claim_sha256') == legacy._digest(row['claim']),
                        'evaluation claim receipt changed')
            if row['status'] == 'COMPLETE':
                require(row.get('completion_sha256') ==
                        legacy._digest(dict(dataset_ref=row['dataset_ref'], receipt=row['receipt'])),
                        'evaluation completion receipt changed')
        return data

    def commit(self, *args, **kwargs):
        raise ValueError('fixed-checkpoint independent evaluation cannot commit PPO updates')

    @classmethod
    def rebind_zero_action_campaign(cls, *args, **kwargs):
        raise ValueError('evaluation UNKNOWN/CLAIMED slots require explicit recovery, never automatic rebind')
