"""Separate physical PPO curriculum capacity; all preceding ledgers stay read only.

Importing, inspecting a snapshot and preparing a suite never reserve capacity.
The only registering operation is an explicitly approved BudgetLedger constructor.
Physical UNKNOWN/CLAIMED consumption is never automatically resubmitted.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
import uuid

from baseline1_v1 import budget as legacy
from grade_multiobjective_ppo_v1 import budget_policy as prior
from u8_wait_efficiency_v1 import budget as preceding

ROOT = legacy.ROOT
GLOBAL_PATH = ROOT / 'runs/physical-realization-ppo-budget-v1.json'
FORMAT = 'physical-realization-ppo-budget-v1'
CAP = 608
TRAIN_UPDATES = 16
require = legacy.require


def _sources():
    """Merge current validated old ledgers, rejecting unknown unmerged sidecars."""
    known={'baseline1-budget-v1.json','active-budget-v2.json','independent-evaluation-budget-v1.json',
           'sampling-comparison-budget-v1.json','u8-wait-efficiency-budget-v1.json',GLOBAL_PATH.name}
    unknown=[p.name for p in (ROOT/'runs').glob('*budget*.json') if p.name not in known]
    require(not unknown,'unmerged additional budget ledger requires explicit integration: '+str(unknown))
    snapshot=preceding.budget_snapshot()
    require(snapshot['legacy_reserved_shifts']==snapshot['active_h2_reserved_shifts']==0,
            'cancelled H2 future reservation policy required')
    state=preceding.budget_state_reference()
    require(preceding.GLOBAL_PATH.is_file(),'preceding u8 complete ledger required')
    return snapshot,dict(state,u8_wait_budget_before=prior.reference(preceding.GLOBAL_PATH))


def budget_state_reference(path=None):
    """Ordinary file reads only: no mkdir, locks, ledger or reservation."""
    _, state = _sources()
    path = GLOBAL_PATH if path is None else Path(path)
    return dict(state, physical_ppo_budget_before=prior.reference(path) if path.is_file() else None)


def check_collisions(suite, snapshot):
    """Reject old/recovered/H2 runs and seeds, including cross-type seed reuse."""
    require(suite['campaign_id'] not in snapshot.get('campaigns', {}),
            'campaign identity collides with ledger history')
    rows = [row['identity'] for row in snapshot.get('jobs', {}).values()]
    rows.extend(snapshot.get('historical_h2_identities', []))
    for record in snapshot.get('recovery_history', {}).values():
        old = legacy._read_reference(record['old_ledger_snapshot'])
        checksum = old.pop('document_sha256', None)
        require(checksum == legacy._digest(old), 'retired budget checksum differs')
        rows.extend(row['identity'] for row in old['jobs'].values())
    for key in ('run_index', 'recording_id', 'source_recording_id'):
        previous = {row.get(key) for row in rows} - {None}
        require(all(job.get(key) not in previous for job in suite['jobs']),
                'new identity collides with ledger history: ' + key)
    previous_seeds = {row.get(key) for row in rows
                      for key in ('experiment_seed', 'actor_sampling_seed','actor_seed')} - {None}
    require(all(job.get(key) not in previous_seeds for job in suite['jobs']
                for key in ('experiment_seed', 'actor_sampling_seed','actor_seed')),
            'new environment/actor seed collides with ledger history across seed types')


def _check_jobs(suite):
    from .gate import check_jobs
    check_jobs(suite)
    require(suite.get('new_shift_cap')==CAP and len(suite['jobs'])==CAP,'exact608 physical slots required')
    require(sum(j['role']=='train' for j in suite['jobs'])==512 and
            sum(j['role']=='t0' for j in suite['jobs'])==16 and
            sum(j['role']=='monitor' for j in suite['jobs'])==16 and
            sum(j['role']=='final' for j in suite['jobs'])==64,'exact512 training and96 evaluation required')
    for update in range(1,17):
        chosen=[j for j in suite['jobs'] if j['target_training_update']==update]
        stage,local=('production',update) if update<=8 else ('wait01',update-8)
        require(len(chosen)==32 and all(j['role']=='train' and j['partition']=='train' and
            j['stage_id']==stage and j['arm_update']==local for j in chosen),
            'each committed update must use one entire32-shift training phase')
    return suite['jobs']


def check_approval(approval, suite, campaign_sha256):
    """Minimum exact capacity gate, followed by the experiment-specific gate."""
    from .gate import check_approval as strict
    legacy._hash(campaign_sha256, 'campaign_sha256')
    require(isinstance(approval, dict) and approval.get('decision') == 'approved' and
            approval.get('approved_by') == 'user' and approval.get('execution_authorized') is True and
            isinstance(approval.get('source_user_instruction'), str) and
            bool(approval['source_user_instruction'].strip()) and
            approval.get('campaign_id') == suite['campaign_id'] and
            approval.get('campaign_sha256') == campaign_sha256 and
            approval.get('max_new_shifts') == suite.get('new_shift_cap') == CAP and
            approval.get('maximum_total_after_registration') == suite.get('maximum_total_if_approved') and
            approval.get('legacy_reserved_shifts') == approval.get('active_h2_reserved_shifts') == 0,
            'exact608 new shifts and explicit user execution approval required')
    return strict(approval, suite, campaign_sha256)


def _check_state(suite, state, current):
    frozen = suite.get('budget_state_before')
    if frozen is None:  # Keep the flat form usable by existing campaign tooling.
        frozen = {key: suite.get(key) for key in state}
        frozen['physical_ppo_budget_before'] = suite.get('physical_ppo_budget_before')
    require(isinstance(frozen, dict) and set(frozen) == set(state) | {'physical_ppo_budget_before'} and
            all(frozen.get(key) == value for key, value in state.items()),
            'preceding complete budget references or cancelled H2 policy changed')
    if current is not None:
        require(frozen['physical_ppo_budget_before'] == current,
                'physical PPO budget changed since suite freeze')


def budget_snapshot(path=None, *, test_only=False):
    """Read the four historical ledgers and this ledger without mutating any."""
    source, _ = _sources()
    path = GLOBAL_PATH if path is None else Path(path)
    result = copy.deepcopy(source)
    if not path.exists():
        return result
    data = BudgetLedger._read(SimpleNamespace(path=path, test_only=test_only))
    require(set(result['campaigns']).isdisjoint(data['campaigns']) and
            set(result['jobs']).isdisjoint(data['jobs']), 'physical PPO/history identity collision')
    result['campaigns'].update(data['campaigns'])
    result['jobs'].update(data['jobs'])
    for key in ('physical_ppo_reserved_shifts', 'additional_reserved_shifts', 'maximum_total_shifts'):
        result[key] = data[key]
    return result


def _validate_commits(entry, jobs):
    commits = entry.get('commits')
    require(isinstance(commits, dict), 'canonical PPO commit inventory required')
    indices = sorted(int(key) for key in commits)
    require(indices == list(range(1, len(indices) + 1)) and len(indices) <= TRAIN_UPDATES,
            'physical PPO curriculum commit sequence differs')
    used, batches = set(), set()
    for update in indices:
        record = commits[str(update)]
        checksum = record.get('commit_sha256')
        require(checksum == legacy._digest({k: v for k, v in record.items() if k != 'commit_sha256'}),
                'checkpoint commit digest mismatch')
        expected = {key for key, row in jobs.items() if row['campaign_id'] == entry['campaign']['campaign_id']
                    and row['identity']['role'] == 'train' and
                    row['identity']['target_training_update'] == update}
        stage, local = ('production', update) if update <= 8 else ('wait01', update - 8)
        require(record.get('update') == update and len(record.get('job_ids', [])) == 32 and
                set(record['job_ids']) == expected and not (used & expected) and
                record.get('batch_id') not in batches and record.get('stage_id') == stage and
                record.get('arm_update') == local and
                {jobs[key]['identity']['phase_id'] for key in expected} == {record.get('phase_id')},
                'PPO commit must contain one complete unused32-shift training phase')
        used.update(expected)
        batches.add(record['batch_id'])
        for key in expected:
            row = jobs[key]
            require(row['status'] == 'COMPLETE' and row.get('committed_update') == update and
                    record.get('receipt_sha256', {}).get(key) == row['completion_sha256'],
                    'committed physical state/completion differs')
    for key, row in jobs.items():
        if row['campaign_id'] == entry['campaign']['campaign_id']:
            require(('committed_update' in row) == (key in used), 'unregistered committed job')


class BudgetLedger(legacy.BudgetLedger):
    """Atomic physical state machine, exact16 continuation updates, zero eval updates."""
    def __init__(self, path, suite, *, campaign_sha256, approval, approval_sha256, test_only=False):
        from .gate import check_sources
        _check_jobs(suite)
        check_approval(approval, suite, campaign_sha256)
        check_sources(suite['source_pins'])
        legacy._hash(approval_sha256, 'approval_sha256')
        require(type(test_only) is bool, 'explicit boolean test_only required')
        self.path = Path(path).resolve()
        self.test_only = test_only
        require(self.path != GLOBAL_PATH.resolve() if test_only else self.path == GLOBAL_PATH.resolve(),
                'production/test u8 wait ledger scope mismatch')
        self.lock_path = self.path.with_name(self.path.name + '.lock')
        self.campaign_id = suite['campaign_id']
        source, state = _sources()
        _check_state(suite, state, None)
        entry = dict(campaign=copy.deepcopy(suite), campaign_sha256=campaign_sha256,
            campaign_identity_sha256=legacy._digest(suite), approval=copy.deepcopy(approval),
            approval_sha256=approval_sha256, approval_identity_sha256=legacy._digest(approval),
            cap=CAP, registered_at=legacy._now(), commits={})
        with self._locked():
            require(_sources() == (source, state), 'preceding budget changed before atomic registration')
            data = self._read() if self.path.exists() else dict(format=FORMAT, test_only=test_only,
                **state, legacy_reserved_shifts=0, active_h2_reserved_shifts=0,
                preceding_reserved_shifts=source['maximum_total_shifts'], physical_ppo_reserved_shifts=0,
                additional_reserved_shifts=source['maximum_total_shifts'],
                maximum_total_shifts=source['maximum_total_shifts'], campaigns={}, jobs={}, revision=0)
            if self.campaign_id in data['campaigns']:
                old = data['campaigns'][self.campaign_id]
                require(all(old[key] == entry[key] for key in ('campaign_sha256', 'campaign_identity_sha256',
                    'approval_sha256', 'approval_identity_sha256', 'cap')),
                    'physical PPO re-registration differs; no extra capacity granted')
                require(data['maximum_total_shifts'] == suite.get('maximum_total_if_approved'),
                        'registered u8 wait capacity changed')
                return
            frozen_current = prior.reference(self.path) if self.path.is_file() else None
            _check_state(suite, state, frozen_current)
            frozen = suite.get('budget_state_before', suite)
            require(frozen.get('physical_ppo_budget_before') == frozen_current,
                    'physical PPO ledger changed since suite freeze')
            check_collisions(suite, source)
            check_collisions(suite, dict(source, campaigns=data['campaigns'], jobs=data['jobs']))
            require(data['maximum_total_shifts'] + CAP == suite.get('maximum_total_if_approved'),
                    'exact608 additional capacity differs from explicit approval')
            data['campaigns'][self.campaign_id] = entry
            for job in suite['jobs']:
                data['jobs'][str(job['run_index'])] = dict(campaign_id=self.campaign_id,
                    identity=copy.deepcopy(job), identity_sha256=legacy._digest(job), status='PLANNED')
            for key in ('physical_ppo_reserved_shifts', 'additional_reserved_shifts', 'maximum_total_shifts'):
                data[key] += CAP
            data['revision'] += 1
            self._write(data)

    def _read(self):
        source, state = _sources()
        data = json.loads(self.path.read_bytes())
        checksum = data.pop('document_sha256', None)
        require(checksum == legacy._digest(data), 'physical PPO budget checksum changed')
        require(data.get('format') == FORMAT and data.get('test_only') is self.test_only and
                all(data.get(key) == value for key, value in state.items()) and
                data.get('legacy_reserved_shifts') == data.get('active_h2_reserved_shifts') == 0,
                'physical PPO scope or preceding immutable budget changed')
        campaigns, jobs = data.get('campaigns'), data.get('jobs')
        require(isinstance(campaigns, dict) and isinstance(jobs, dict), 'canonical u8 wait inventory required')
        cap = CAP * len(campaigns)
        require(data.get('preceding_reserved_shifts') == source['maximum_total_shifts'] and
                data.get('physical_ppo_reserved_shifts') == len(jobs) == cap and
                data.get('additional_reserved_shifts') == data.get('maximum_total_shifts') ==
                source['maximum_total_shifts'] + cap, 'physical PPO physical capacity accounting differs')
        planned = {}
        for cid, entry in campaigns.items():
            suite = entry['campaign']
            _check_jobs(suite)
            check_approval(entry['approval'], suite, entry['campaign_sha256'])
            legacy._hash(entry['approval_sha256'], 'approval_sha256')
            _check_state(suite, state, None)
            require(cid == suite['campaign_id'] and entry['cap'] == CAP and
                    entry['campaign_identity_sha256'] == legacy._digest(suite) and
                    entry['approval_identity_sha256'] == legacy._digest(entry['approval']),
                    'physical PPO campaign/approval identity changed')
            check_collisions(suite, source)
            for job in suite['jobs']:
                key = str(job['run_index'])
                require(key not in planned, 'duplicate u8 wait canonical job')
                planned[key] = cid, job
        require(set(planned) == set(jobs), 'physical PPO physical job inventory differs')
        for key, row in jobs.items():
            cid, identity = planned[key]
            require(row['campaign_id'] == cid and row['identity'] == identity and
                    row['identity_sha256'] == legacy._digest(identity) and
                    row['status'] in ('PLANNED', 'CLAIMED', 'COMPLETE', 'UNKNOWN'),
                    'physical PPO physical state/identity differs')
            if row['status'] != 'PLANNED':
                require(isinstance(row.get('claim'), dict) and row.get('claim_sha256') == legacy._digest(row['claim']),
                        'physical PPO claim receipt changed')
            if row['status'] == 'COMPLETE':
                require(row.get('completion_sha256') == legacy._digest(dict(dataset_ref=row['dataset_ref'],
                    receipt=row['receipt'])), 'physical PPO completion receipt changed')
                _check_receipt(identity, row['receipt'])
            if identity['role'] != 'train':
                require('committed_update' not in row, 'independent evaluation cannot enter PPO')
        for entry in campaigns.values():
            _validate_commits(entry, jobs)
        return data

    def snapshot(self):
        return copy.deepcopy(self._read())

    def registration(self):
        data = self._read()
        entry = data['campaigns'][self.campaign_id]
        return dict(protocol='physical-realization-ppo-budget-registration-v1', campaign_id=self.campaign_id,
            **{key: entry[key] for key in ('campaign_sha256', 'campaign_identity_sha256',
                'approval_sha256', 'approval_identity_sha256', 'cap')},
            preceding_reserved_shifts=data['preceding_reserved_shifts'],
            maximum_total_shifts=data['maximum_total_shifts'])

    def claim(self, job):
        with self._locked():
            data=self._read();_,row=self._job(data,job)
            commits=data['campaigns'][self.campaign_id]['commits']
            require(row['status']=='PLANNED','slot already claimed/complete/unknown; never automatically resubmit')
            previous=[r for r in data['jobs'].values() if r['campaign_id']==self.campaign_id and
                      r['identity']['ordinal']<job['ordinal']]
            require(all(r['status']=='COMPLETE' for r in previous),'all previous physical slots must complete before next ordinal')
            expected=(job['target_training_update']-1 if job['role']=='train' else
                      {'t0':0,'monitor':8,'final':16}[job['role']])
            require(len(commits)==expected,'stage/evaluation requires its exact frozen predecessor update')
            row.update(status='CLAIMED',claim=dict(token=uuid.uuid4().hex,claimed_at=legacy._now(),
                campaign_id=self.campaign_id,identity_sha256=row['identity_sha256']))
            row['claim_sha256']=legacy._digest(row['claim']);data['revision']+=1;self._write(data)
            return copy.deepcopy(row)

    def complete(self, job, dataset_ref, receipt):
        _check_receipt(job, receipt)
        return super().complete(job, dataset_ref, receipt)

    def commit(self, update, jobs, checkpoint_ref, batch_id):
        require(type(update) is int and 1 <= update <= TRAIN_UPDATES, 'positive1..16 committed update required')
        require(isinstance(jobs, list) and len(jobs) == 32, 'entire32-shift training phase required')
        require(isinstance(batch_id, str) and bool(batch_id.strip()), 'explicit batch_id required')
        checkpoint = legacy._reference(checkpoint_ref)
        with self._locked():
            data = self._read()
            campaign = data['campaigns'][self.campaign_id]
            require(update == len(campaign['commits']) + 1, 'committed update must be next uncommitted update')
            chosen = [self._job(data, job) for job in jobs]
            keys = [key for key, _ in chosen]
            expected = {key for key, row in data['jobs'].items() if row['campaign_id'] == self.campaign_id and
                row['identity']['role'] == 'train' and row['identity']['target_training_update'] == update}
            require(len(set(keys)) == 32 and set(keys) == expected,
                    'only the entire frozen32-shift train phase may enter PPO; no independent evaluation')
            for _, row in chosen:
                require(row['status'] == 'COMPLETE' and 'committed_update' not in row,
                        'training jobs must be complete and not previously committed')
                _check_receipt(row['identity'], row['receipt'])
                legacy._reference(row['dataset_ref'])
            require(not any(c['batch_id'] == batch_id for entry in data['campaigns'].values()
                for c in entry['commits'].values()), 'batch_id already committed globally')
            stage, local = ('production', update) if update <= 8 else ('wait01', update - 8)
            commit = dict(update=update, stage_id=stage, arm_update=local, phase_id=jobs[0]['phase_id'],
                job_ids=sorted(keys), checkpoint_ref=checkpoint, batch_id=batch_id,
                receipt_sha256={key: row['completion_sha256'] for key, row in chosen}, committed_at=legacy._now())
            commit['commit_sha256'] = legacy._digest(commit)
            campaign['commits'][str(update)] = commit
            for _, row in chosen:
                row['committed_update'] = update
            data['revision'] += 1
            self._write(data)
            return copy.deepcopy(commit)

    @classmethod
    def rebind_zero_action_campaign(cls, *args, **kwargs):
        raise ValueError('physical PPO UNKNOWN/CLAIMED slots require explicit recovery, never automatic rebind')


def _check_receipt(job, receipt):
    require(isinstance(receipt, dict) and receipt.get('normal_completion') is True and
            receipt.get('source_kind') == 'real_anylogic_rpc' and
            not any(receipt.get(key, False) for key in ('truncated', 'abnormal_interruption', 'sampler_truncated')),
            'normal complete real-AnyLogic-RPC shift receipt required')
    for key in ('run_index', 'experiment_seed', 'role', 'partition', 'stage_id', 'phase_id'):
        require(receipt.get(key) == job.get(key), 'completion receipt identity differs: ' + key)
    if job['role'] == 'train':
        require(job['partition'] == 'train' and receipt.get('evaluation_only', False) is False and
                receipt.get('ppo_trainable', True) is True, 'independent evaluation source cannot enter PPO')
    else:
        require(job['partition']=='validation' and receipt.get('evaluation_only') is True and
                receipt.get('ppo_trainable') is False,'independent evaluation flags and partition required')
