"""Append-only ledger for the new continuous mixed-grade campaign."""
from pathlib import Path

from mixed_plan import (
    APPROVAL, BATCH_SIZE, CAMPAIGN, LEDGER, SUITE, TOTAL, UPDATES,
    authorization_for, check_collisions, load_package, read, reference, require,
)
from business_graph_summary_campaign_v1.budget import BudgetLedger, _payload, _state, connection, digest
from business_graph_summary_campaign_v1.plan_io import encoded
from change_plans.token_attention_001.business_graph_summary_009.generalization_001 import gen_budget
from change_plans.token_attention_001.business_graph_summary_009.training_extension_001.extension_budget import (
    _complete_receipt,
)

FORMAT = 'bg128-continuous-mixed-budget-v1'


class MixedLedger(BudgetLedger):
    """Use the existing slot and update commit protocol in a separate registry."""

    def __init__(self, suite, configs, *, path=None, authority_path=APPROVAL):
        self.path = LEDGER if path is None else Path(path).resolve()
        self.suite = suite
        self.campaign_id = CAMPAIGN
        self.configs = {
            cfg['phase_id']: ref
            for cfg, ref in zip(configs, suite['phase_configs'], strict=True)
        }
        self.jobs = {job['run_index']: job for job in suite['jobs']}
        self.hashes = {run: digest(job) for run, job in self.jobs.items()}
        self.update_jobs = {}
        for job in self.jobs.values():
            if job['role'] == 'train':
                self.update_jobs.setdefault(job['target_training_update'], []).append(job)
        authority_path = Path(authority_path).resolve()
        require(
            suite['campaign_id'] == CAMPAIGN and len(self.jobs) == TOTAL
            and set(self.update_jobs) == set(range(1, UPDATES + 1))
            and all(len(jobs) == BATCH_SIZE for jobs in self.update_jobs.values()),
            'continuous mixed ledger inventory differs',
        )
        require(
            (path is None and authority_path == APPROVAL)
            or (path is not None and self.path != LEDGER and authority_path != APPROVAL),
            'test authority cannot target the live ledger',
        )
        require(read(authority_path) == authorization_for(suite), 'frozen launch authority differs')
        self.entry = dict(
            format=FORMAT, cap=TOTAL, test_only=path is not None,
            suite_reference=reference(SUITE), approval_reference=reference(authority_path),
        )

    def register(self):
        """Register only after the operator has started the matching AnyLogic run."""
        check_collisions(self.suite)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with connection(self.path) as con:
            con.execute('CREATE TABLE IF NOT EXISTS campaigns (id TEXT PRIMARY KEY,payload TEXT,checksum TEXT)')
            con.execute('CREATE TABLE IF NOT EXISTS slots (run INTEGER PRIMARY KEY,campaign TEXT,status TEXT,payload TEXT,checksum TEXT)')
            con.execute('CREATE TABLE IF NOT EXISTS commits (campaign TEXT,update_id INTEGER,payload TEXT,checksum TEXT,PRIMARY KEY(campaign,update_id))')
            rows = con.execute('SELECT id,payload,checksum FROM campaigns').fetchall()
            if rows:
                require(len(rows) == 1 and rows[0][0] == CAMPAIGN
                        and _payload(rows[0][1:]) == self.entry, 'registration or authority changed')
                return
            con.execute('INSERT INTO campaigns VALUES (?,?,?)',
                        (CAMPAIGN, encoded(self.entry).decode(), digest(self.entry)))
            records = []
            for run, identity_digest in self.hashes.items():
                payload = dict(status='PLANNED', identity_sha256=identity_digest)
                records.append((run, CAMPAIGN, 'PLANNED', encoded(payload).decode(), digest(payload)))
            con.executemany('INSERT INTO slots VALUES (?,?,?,?,?)', records)

    def snapshot(self):
        with connection(self.path, readonly=True) as con:
            rows = con.execute('SELECT id,payload,checksum FROM campaigns').fetchall()
            require(len(rows) == 1 and rows[0][0] == CAMPAIGN
                    and _payload(rows[0][1:]) == self.entry, 'registration changed')
            require(con.execute('SELECT count(*) FROM slots').fetchone()[0] == TOTAL,
                    'slot inventory differs')
            state = _state(con, self.suite, self.entry, self.hashes)
        incomplete = False
        for run in self.jobs:
            record = state['jobs'][str(run)]
            require(not incomplete or record['status'] == 'PLANNED', 'slots left frozen physical order')
            incomplete = incomplete or record['status'] != 'COMPLETE'
        return state

    def claim(self, job):
        with connection(self.path) as con:
            record = self._row(con, job)
            require(record['status'] == 'PLANNED' and con.execute(
                "SELECT 1 FROM slots WHERE run < ? AND status != 'COMPLETE' LIMIT 1",
                (job['run_index'],),
            ).fetchone() is None, 'slot cannot be started twice or before the preceding slot')
            commits = con.execute('SELECT count(*) FROM commits WHERE campaign=?', (CAMPAIGN,)).fetchone()[0]
            needed = job['target_training_update'] - 1 if job['role'] == 'train' else UPDATES
            require(commits == needed, 'physical shift would use an uncommitted training checkpoint')
            record['status'] = 'CLAIMED'
            self._put(con, job, record)

    def complete(self, job, dataset_ref, receipt):
        _complete_receipt(job, self.configs[job['phase_id']], dataset_ref, receipt)
        super().complete(job, dataset_ref, receipt)


def budget_snapshot():
    """All historical budgets plus this campaign after operator registration."""
    before = gen_budget.budget_snapshot()
    if not LEDGER.is_file():
        return before
    suite, configs = load_package()
    check_collisions(suite, before)
    current = MixedLedger(suite, configs).snapshot()
    before['jobs'].update(current['jobs'])
    before['campaigns'].update(current['campaigns'])
    before['maximum_total_shifts'] += TOTAL
    before['additional_reserved_shifts'] += TOTAL
    return before
