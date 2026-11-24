"""Additional Baseline 1 slots; old H2's 11,280 reserved shifts remain intact.

Only approved, immutable campaigns can register. Physical jobs and committed
updates are separate: completed persistent data may be loaded after a restart,
whereas CLAIMED/UNKNOWN jobs never become fresh slots automatically.
"""
from contextlib import contextmanager
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import uuid

ROOT = Path(__file__).resolve().parents[1]
GLOBAL_PATH = ROOT / 'runs' / 'baseline1-budget-v1.json'
FORMAT = 'baseline1-global-budget-v1'
LEGACY_RESERVED = 11280
ROLES = {'calibration', 'train', 't0', 'monitor', 'final', 'evaluation', 'final_blind'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode()


def _digest(value):
    return hashlib.sha256(_bytes(value)).hexdigest()


def _hash(value, name):
    require(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None,
            name + ' must be a full SHA256')


def _now():
    return datetime.now(timezone.utc).isoformat()


def _positive_integer(value):
    return type(value) is int and value > 0


def _validate_campaign(campaign, campaign_sha256, approval, approval_sha256):
    _hash(campaign_sha256, 'campaign_sha256')
    _hash(approval_sha256, 'approval_sha256')
    require(isinstance(campaign, dict) and isinstance(approval, dict), 'explicit campaign and approval required')
    _bytes(campaign)
    _bytes(approval)
    require(isinstance(campaign.get('campaign_id'), str) and campaign['campaign_id'].strip(), 'campaign_id required')
    require(campaign.get('kind') in ('calibration', 'single_seed_trial'), 'unsupported campaign kind')
    require(campaign.get('legacy_reserved_shifts') == LEGACY_RESERVED,
            'old H2 reservation must remain 11280')
    jobs = campaign.get('jobs')
    require(isinstance(jobs, list) and bool(jobs), 'explicit nonempty campaign jobs required')
    require(_positive_integer(campaign.get('new_shift_cap')) and campaign['new_shift_cap'] == len(jobs),
            'new_shift_cap must equal the frozen physical job count')
    seen = set()
    for job in jobs:
        require(isinstance(job, dict) and _positive_integer(job.get('run_index')) and
                _positive_integer(job.get('experiment_seed')), 'invalid explicit run_index/experiment_seed')
        require(job.get('role') in ROLES and isinstance(job.get('phase_id'), str) and job['phase_id'].strip(),
                'explicit supported role and phase_id required')
        require((job['role'] == 'calibration') == (campaign['kind'] == 'calibration'),
                'job role differs from campaign scope')
        require(job['run_index'] not in seen, 'duplicate run_index in campaign')
        seen.add(job['run_index'])
        for key in ('target_training_update', 'checkpoint_update'):
            if key in job:
                require(type(job[key]) is int and job[key] >= 0, 'invalid job update index')
    require(approval.get('decision') == 'approved' and
            approval.get('campaign_id') == campaign['campaign_id'] and
            approval.get('campaign_sha256') == campaign_sha256, 'approval scope/hash mismatch or missing approval')
    require(_positive_integer(approval.get('max_new_shifts')) and
            approval['max_new_shifts'] >= len(jobs), 'approval does not cover new physical cap')
    require(approval.get('legacy_reserved_shifts') == LEGACY_RESERVED and
            approval.get('accounting') == 'additional_to_h2_reserved',
            'approval must preserve H2 and explicitly approve additional slots')
    if 'kind' in approval:
        require(approval['kind'] == campaign['kind'], 'approval kind mismatch')


def _reference(value):
    require(isinstance(value, dict) and set(value) >= {'path', 'sha256'}, 'artifact path/SHA256 required')
    _hash(value['sha256'], 'artifact sha256')
    path = Path(value['path'])
    require(path.is_absolute() and path.is_file(), 'persistent absolute artifact file required')
    require(hashlib.sha256(path.read_bytes()).hexdigest() == value['sha256'], 'artifact byte hash mismatch')
    return dict(path=str(path.resolve()), sha256=value['sha256'])


def _read_reference(value):
    return json.loads(Path(_reference(value)['path']).read_bytes())


def _check_recovery(old, new, authorization_ref, evidence_ref, snapshot_ref):
    """A reviewed zero-action failure can transfer a reservation, never add one."""
    authorization = _read_reference(authorization_ref)
    evidence = _read_reference(evidence_ref)
    snapshot = _read_reference(snapshot_ref)
    checksum = snapshot.pop('document_sha256', None)
    require(checksum == _digest(snapshot), 'recovery snapshot checksum mismatch')
    cid, new_cid = old['campaign']['campaign_id'], new['campaign']['campaign_id']
    require(authorization.get('protocol') == 'baseline1-zero-action-rebind-authorization-v1'
            and authorization.get('decision') == 'approved' and authorization.get('approved_by') == 'user'
            and authorization.get('old_campaign_id') == cid and authorization.get('new_campaign_id') == new_cid
            and authorization.get('original_shift_cap') == old['cap']
            and authorization.get('additional_full_shift_budget') == 0
            and bool(authorization.get('source_user_instruction')), 'explicit scoped recovery authorization required')
    require(old['cap'] == new['cap'] and old['campaign']['kind'] == new['campaign']['kind'] == 'calibration'
            and not old['commits'] and not new['commits'], 'only unchanged calibration capacity can transfer')
    require(snapshot['campaigns'].get(cid) == old, 'recovery snapshot old campaign differs')
    old_jobs = {k: v for k, v in snapshot['jobs'].items() if v['campaign_id'] == cid}
    require(len(old_jobs) == old['cap'] and sum(v['status'] in ('CLAIMED', 'UNKNOWN') for v in old_jobs.values()) == 1
            and all(v['status'] in ('PLANNED', 'CLAIMED', 'UNKNOWN') for v in old_jobs.values()),
            'recovery requires exactly one failed startup and no completed shift')
    require(evidence.get('protocol') == 'baseline1-zero-action-startup-evidence-v1'
            and evidence.get('classification') == 'PRE_ACTION_STARTUP_FAILURE'
            and evidence.get('old_campaign_id') == cid
            and evidence.get('old_ledger_snapshot') == snapshot_ref
            and evidence.get('old_controller_exited') is True
            and evidence.get('failure_before_recorder') is True
            and evidence.get('failure_before_truck_activation') is True
            and evidence.get('physical_shift_started') is False,
            'explicit pre-action failure and controller-exit evidence required')
    for key in ('accepted_rpc_callbacks', 'action_responses', 'production_actions', 'physical_event_records'):
        require(type(evidence.get(key)) is int and evidence[key] == 0, 'nonzero or unknown recovery activity: ' + key)
    proofs = evidence.get('proof_refs', {})
    require(set(proofs) >= {'failure_diagnosis', 'controller_exit', 'aborted', 'java_claim', 'stop'},
            'complete recovery proof references required')
    for ref in proofs.values():
        _reference(ref)
    aborted = _read_reference(proofs['aborted'])
    require(aborted.get('completed_episodes') == 0 and aborted.get('transitions') == 0
            and aborted.get('error') == 'timed out waiting for AnyLogic callback',
            'aborted first callback evidence differs')
    approval = new['approval']
    require(approval.get('approval_basis') == 'transfer_existing_reservation'
            and approval.get('inherited_campaign_id') == cid
            and approval.get('inherited_campaign_sha256') == old['campaign_sha256']
            and approval.get('inherited_approval_sha256') == old['approval_sha256']
            and approval.get('recovery_authorization_ref') == authorization_ref,
            'new approval must inherit the exact existing reservation')
    for key in ('run_index', 'experiment_seed', 'actor_sampling_seed', 'recording_id', 'event_directory', 'python_output'):
        prior = {j.get(key) for j in old['campaign']['jobs']}
        fresh = {j.get(key) for j in new['campaign']['jobs']}
        require(None not in prior and None not in fresh and prior.isdisjoint(fresh), 'recovery identity reused: ' + key)
    return old_jobs


class BudgetLedger:
    def __init__(self, path, campaign, *, campaign_sha256, approval, approval_sha256, test_only=False):
        # Authorization is checked before mkdir, lock creation or any ledger write.
        _validate_campaign(campaign, campaign_sha256, approval, approval_sha256)
        require(type(test_only) is bool, 'explicit boolean test_only required')
        self.path = Path(path).resolve()
        require(test_only or self.path == GLOBAL_PATH.resolve(), 'production ledger must use the shared global path')
        require(not test_only or self.path != GLOBAL_PATH.resolve(), 'test ledger cannot use production path')
        self.lock_path = self.path.with_name(self.path.name + '.lock')
        self.campaign_id = campaign['campaign_id']
        self.test_only = test_only
        spec = copy.deepcopy(campaign)
        entry = dict(campaign=spec, campaign_sha256=campaign_sha256,
            campaign_identity_sha256=_digest(spec), approval=copy.deepcopy(approval),
            approval_sha256=approval_sha256, approval_identity_sha256=_digest(approval),
            cap=spec['new_shift_cap'], registered_at=_now(), commits={})
        with self._locked():
            data = self._read() if self.path.exists() else dict(format=FORMAT, test_only=test_only,
                legacy_reserved_shifts=LEGACY_RESERVED, additional_reserved_shifts=0,
                maximum_total_shifts=LEGACY_RESERVED, campaigns={}, jobs={}, revision=0)
            require(self.campaign_id not in data.get('recovery_history', {}),
                    'retired failed campaign cannot register again')
            if self.campaign_id in data['campaigns']:
                old = data['campaigns'][self.campaign_id]
                require(all(old[k] == entry[k] for k in ('campaign_sha256', 'campaign_identity_sha256',
                    'approval_sha256', 'approval_identity_sha256', 'cap')),
                    'campaign re-registration differs; no extra capacity is granted')
                return
            for job in spec['jobs']:
                require(str(job['run_index']) not in data['jobs'], 'global run_index already reserved')
                for history in data.get('recovery_history', {}).values():
                    prior = _read_reference(history['old_ledger_snapshot'])
                    require(str(job['run_index']) not in prior['jobs'], 'retired run_index cannot be reused')
            data['campaigns'][self.campaign_id] = entry
            for job in spec['jobs']:
                data['jobs'][str(job['run_index'])] = dict(campaign_id=self.campaign_id,
                    identity=copy.deepcopy(job), identity_sha256=_digest(job), status='PLANNED')
            data['additional_reserved_shifts'] += entry['cap']
            data['maximum_total_shifts'] = LEGACY_RESERVED + data['additional_reserved_shifts']
            data['revision'] += 1
            self._write(data)

    @contextmanager
    def _locked(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open('a+b') as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def _read(self):
        data = json.loads(self.path.read_text())
        checksum = data.pop('document_sha256', None)
        require(checksum == _digest(data), 'ledger checksum mismatch; stop for reconciliation')
        require(data.get('format') == FORMAT and data.get('test_only') is self.test_only,
                'ledger format/test scope mismatch')
        require(data['legacy_reserved_shifts'] == LEGACY_RESERVED, 'legacy H2 reservation changed')
        history = data.get('recovery_history', {})
        require(isinstance(history, dict) and len(history) <= 1, 'repeated recovery needs a new reviewed protocol')
        for cid, record in history.items():
            require(record.get('record_sha256') == _digest({k: v for k, v in record.items() if k != 'record_sha256'}),
                    'recovery history checksum mismatch')
            require(cid not in data['campaigns'] and record['new_campaign_id'] in data['campaigns'],
                    'retired recovery campaign is active or successor missing')
            snapshot = _read_reference(record['old_ledger_snapshot'])
            old = snapshot['campaigns'][cid]
            old_jobs = _check_recovery(old, data['campaigns'][record['new_campaign_id']],
                record['authorization'], record['evidence'], record['old_ledger_snapshot'])
            require(set(old_jobs).isdisjoint(data['jobs']) and record['transferred_capacity'] == old['cap'],
                    'retired job reused or transferred capacity differs')
        cap = sum(c['cap'] for c in data['campaigns'].values())
        require(data['additional_reserved_shifts'] == cap == len(data['jobs']) and
                data['maximum_total_shifts'] == LEGACY_RESERVED + cap, 'global additional budget accounting mismatch')
        planned = {}
        for cid, campaign in data['campaigns'].items():
            spec = campaign['campaign']
            _validate_campaign(spec, campaign['campaign_sha256'], campaign['approval'], campaign['approval_sha256'])
            require(cid == spec['campaign_id'] and campaign['cap'] == spec['new_shift_cap'] and
                    campaign['campaign_identity_sha256'] == _digest(spec) and
                    campaign['approval_identity_sha256'] == _digest(campaign['approval']), 'registered campaign changed')
            committed_ids, batch_ids = set(), set()
            indices = sorted(int(u) for u in campaign['commits'])
            require(indices == list(range(1, len(indices) + 1)), 'committed update sequence differs')
            for index in indices:
                commit = campaign['commits'][str(index)]
                require(commit['commit_sha256'] == _digest({k: v for k, v in commit.items() if k != 'commit_sha256'}),
                        'checkpoint commit digest mismatch')
                require(commit['batch_id'] not in batch_ids and not (set(commit['job_ids']) & committed_ids),
                        'duplicate committed batch/jobs')
                batch_ids.add(commit['batch_id'])
                committed_ids.update(commit['job_ids'])
                for key in commit['job_ids']:
                    require(key in data['jobs'] and data['jobs'][key]['campaign_id'] == cid and
                            data['jobs'][key]['status'] == 'COMPLETE' and
                            data['jobs'][key].get('committed_update') == index, 'committed physical state differs')
            for job in spec['jobs']:
                key = str(job['run_index'])
                require(key not in planned, 'global duplicate canonical job')
                planned[key] = (cid, job)
        require(set(planned) == set(data['jobs']), 'global job inventory differs')
        for key, row in data['jobs'].items():
            cid, identity = planned[key]
            require(row['campaign_id'] == cid and row['identity'] == identity and
                    row['identity_sha256'] == _digest(identity), 'canonical job identity changed')
            require(row['status'] in ('PLANNED', 'CLAIMED', 'COMPLETE', 'UNKNOWN'), 'unsupported physical status')
            if row['status'] != 'PLANNED':
                require(isinstance(row.get('claim'), dict) and
                        row['claim_sha256'] == _digest(row['claim']), 'physical claim receipt changed')
            if row['status'] == 'COMPLETE':
                require(row['completion_sha256'] == _digest(dict(dataset_ref=row['dataset_ref'], receipt=row['receipt'])),
                        'completion receipt changed')
        return data

    @classmethod
    def rebind_zero_action_campaign(cls, path, *, old_campaign_id, expected_ledger_sha256,
            old_campaign_sha256, old_approval_sha256, new_campaign, new_campaign_sha256,
            new_approval, new_approval_sha256, recovery_authorization_ref, evidence_ref,
            old_ledger_snapshot_ref, test_only=False):
        """Explicit operator recovery; ordinary resume/claim never calls this API."""
        _validate_campaign(new_campaign, new_campaign_sha256, new_approval, new_approval_sha256)
        ledger = object.__new__(cls)
        ledger.path = Path(path).resolve()
        ledger.lock_path = ledger.path.with_name(ledger.path.name + '.lock')
        ledger.test_only = test_only
        ledger.campaign_id = new_campaign['campaign_id']
        require(type(test_only) is bool and
                (ledger.path != GLOBAL_PATH.resolve() if test_only else ledger.path == GLOBAL_PATH.resolve()),
                'recovery production/test ledger scope mismatch')
        with ledger._locked():
            require(hashlib.sha256(ledger.path.read_bytes()).hexdigest() == expected_ledger_sha256,
                    'ledger changed since recovery review')
            data = ledger._read()
            require(not data.get('recovery_history'), 'recovery already applied; never rebind twice')
            require(old_campaign_id in data['campaigns'] and ledger.campaign_id not in data['campaigns'],
                    'old campaign missing or recovery successor already registered')
            old = data['campaigns'][old_campaign_id]
            require(old['campaign_sha256'] == old_campaign_sha256 and old['approval_sha256'] == old_approval_sha256,
                    'old campaign/approval changed')
            require(_reference(old_ledger_snapshot_ref)['sha256'] == expected_ledger_sha256,
                    'snapshot must preserve exact pre-recovery ledger bytes')
            spec = copy.deepcopy(new_campaign)
            entry = dict(campaign=spec, campaign_sha256=new_campaign_sha256,
                campaign_identity_sha256=_digest(spec), approval=copy.deepcopy(new_approval),
                approval_sha256=new_approval_sha256, approval_identity_sha256=_digest(new_approval),
                cap=spec['new_shift_cap'], registered_at=_now(), commits={})
            prior_jobs = _check_recovery(old, entry, recovery_authorization_ref, evidence_ref, old_ledger_snapshot_ref)
            remaining_keys = set(data['jobs']) - set(prior_jobs)
            require(all(str(j['run_index']) not in data['jobs'] for j in spec['jobs']),
                    'recovery run_index already reserved')
            record = dict(old_campaign_id=old_campaign_id, new_campaign_id=ledger.campaign_id,
                old_ledger_snapshot=copy.deepcopy(old_ledger_snapshot_ref), authorization=copy.deepcopy(recovery_authorization_ref),
                evidence=copy.deepcopy(evidence_ref), transferred_capacity=old['cap'], recovered_at=_now())
            record['record_sha256'] = _digest(record)
            data['recovery_history'] = {old_campaign_id: record}
            data['campaigns'].pop(old_campaign_id)
            data['campaigns'][ledger.campaign_id] = entry
            data['jobs'] = {k: v for k, v in data['jobs'].items() if k in remaining_keys}
            for job in spec['jobs']:
                data['jobs'][str(job['run_index'])] = dict(campaign_id=ledger.campaign_id,
                    identity=copy.deepcopy(job), identity_sha256=_digest(job), status='PLANNED')
            data['revision'] += 1
            ledger._write(data)
            return ledger.snapshot_unlocked_checked()

    def snapshot_unlocked_checked(self):
        return copy.deepcopy(self._read())

    def _write(self, data):
        document = copy.deepcopy(data)
        document['document_sha256'] = _digest(data)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile('wb', dir=self.path.parent,
                    prefix='.' + self.path.name + '.', delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(_bytes(document) + b'\n')
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            temporary = None
            descriptor = os.open(str(self.path.parent), os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    def _job(self, data, job):
        require(isinstance(job, dict) and _positive_integer(job.get('run_index')), 'full canonical job required')
        key = str(job['run_index'])
        require(key in data['jobs'] and data['jobs'][key]['campaign_id'] == self.campaign_id,
                'job is outside approved campaign')
        require(data['jobs'][key]['identity'] == job, 'job differs from frozen campaign identity')
        return key, data['jobs'][key]

    def snapshot(self):
        with self._locked():
            return copy.deepcopy(self._read())

    def claim(self, job):
        with self._locked():
            data = self._read()
            _, row = self._job(data, job)
            require(row['status'] == 'PLANNED', 'slot already claimed/complete/unknown; never automatically resubmit')
            row['status'] = 'CLAIMED'
            row['claim'] = dict(token=uuid.uuid4().hex, claimed_at=_now(),
                                campaign_id=self.campaign_id, identity_sha256=row['identity_sha256'])
            row['claim_sha256'] = _digest(row['claim'])
            data['revision'] += 1
            self._write(data)
            return copy.deepcopy(row)

    def complete(self, job, dataset_ref, receipt):
        reference = _reference(dataset_ref)
        require(isinstance(receipt, dict) and receipt.get('normal_completion') is True,
                'explicit normal complete-shift receipt required')
        _bytes(receipt)
        for key, value in job.items():
            if key in receipt:
                require(receipt[key] == value, 'completion receipt identity differs: ' + key)
        require(not any(receipt.get(k, False) for k in ('truncated', 'abnormal_interruption', 'sampler_truncated')),
                'partial/truncated receipt cannot complete a shift')
        with self._locked():
            data = self._read()
            _, row = self._job(data, job)
            require(row['status'] == 'CLAIMED', 'completion requires an outstanding claimed slot')
            row.update(status='COMPLETE', dataset_ref=reference, receipt=copy.deepcopy(receipt), completed_at=_now())
            row['completion_sha256'] = _digest(dict(dataset_ref=reference, receipt=receipt))
            data['revision'] += 1
            self._write(data)
            return copy.deepcopy(row)

    def unknown(self, job, error):
        require(isinstance(error, str) and error.strip(), 'explicit unknown-consumption error required')
        with self._locked():
            data = self._read()
            _, row = self._job(data, job)
            require(row['status'] == 'CLAIMED', 'UNKNOWN requires an outstanding claimed slot')
            row.update(status='UNKNOWN', error=error, failed_at=_now())
            data['revision'] += 1
            self._write(data)
            return copy.deepcopy(row)

    def resume(self):
        with self._locked():
            data = self._read()
            jobs = [j for j in data['jobs'].values() if j['campaign_id'] == self.campaign_id]
            require(all(j['status'] not in ('CLAIMED', 'UNKNOWN') for j in jobs),
                    'claimed/unknown physical consumption exists; reconcile without resampling')
            for row in jobs:
                if row['status'] == 'COMPLETE':
                    _reference(row['dataset_ref'])
            for commit in data['campaigns'][self.campaign_id]['commits'].values():
                _reference(commit['checkpoint_ref'])
            return copy.deepcopy(data)

    def commit(self, update, jobs, checkpoint_ref, batch_id):
        require(_positive_integer(update), 'positive committed update index required')
        require(isinstance(jobs, list) and bool(jobs), 'complete train phase job list required')
        require(isinstance(batch_id, str) and batch_id.strip(), 'explicit batch_id required')
        checkpoint = _reference(checkpoint_ref)
        with self._locked():
            data = self._read()
            campaign = data['campaigns'][self.campaign_id]
            require(campaign['campaign']['kind'] == 'single_seed_trial', 'calibration cannot commit PPO updates')
            require(update == len(campaign['commits']) + 1, 'committed update must be the next uncommitted update')
            chosen = [self._job(data, job) for job in jobs]
            keys = [key for key, _ in chosen]
            require(len(keys) == len(set(keys)), 'duplicate job in committed batch')
            require(all(row['identity']['role'] == 'train' for _, row in chosen), 'only train jobs can enter PPO commit')
            phases = {row['identity']['phase_id'] for _, row in chosen}
            require(len(phases) == 1, 'PPO commit mixes training phases')
            phase = next(iter(phases))
            expected = {k for k, row in data['jobs'].items() if row['campaign_id'] == self.campaign_id and
                        row['identity']['role'] == 'train' and row['identity']['phase_id'] == phase}
            require(set(keys) == expected, 'PPO commit must contain the entire frozen training phase')
            for _, row in chosen:
                require(row['status'] == 'COMPLETE' and 'committed_update' not in row,
                        'training jobs must be complete and not previously committed')
                if 'target_training_update' in row['identity']:
                    require(row['identity']['target_training_update'] == update, 'job target_training_update mismatch')
                _reference(row['dataset_ref'])
            require(not any(c['batch_id'] == batch_id for entry in data['campaigns'].values()
                            for c in entry['commits'].values()), 'batch_id already committed globally')
            commit = dict(update=update, phase_id=phase, job_ids=sorted(keys), checkpoint_ref=checkpoint,
                batch_id=batch_id, receipt_sha256={k: row['completion_sha256'] for k, row in chosen}, committed_at=_now())
            commit['commit_sha256'] = _digest(commit)
            campaign['commits'][str(update)] = commit
            for _, row in chosen:
                row['committed_update'] = update
            data['revision'] += 1
            self._write(data)
            return copy.deepcopy(commit)
