"""Register the four new Full/GAT1 arms against actual and prepared physical identities."""

import csv
from pathlib import Path
import sys

if __package__:
    from . import extension_plan as plan
else:
    import extension_plan as plan

GRAPH = Path(__file__).resolve().parents[2]
DEMO = GRAPH / 'fleet_outage_resilience_001' / 'visual_acceptance_demo_001'
if str(DEMO) not in sys.path:
    sys.path.insert(0, str(DEMO))

import demo_budget as predecessor
import replication_budget as replication
from weak_budget import (
    LOCKS,
    PROFILE_FIELDS,
    SCENE_FIELDS,
    SEED_FIELDS,
    WeakLedger,
    _source_jobs,
    authority_path,
    authorization,
)
from weak_plan import bind


FORMAT = 'bg128-full-and-gat1-seed-extension-budget-v1'
_prior_rows = bind(predecessor.ledger.previous.paired10._prior_rows, SUITE=plan.PREPARED / 'suite.json')
_SEED_COLUMNS = ('run_index', 'actor_seed', *SEED_FIELDS)


def _prepared_rows(previous, history):
    """Include inline and sidecar rosters, including prepared Full s04/s05 slots."""
    rows = _prior_rows(previous)
    for path in sorted(GRAPH.rglob('master_plan.csv')):
        if path.is_relative_to(plan.PREPARED):
            continue
        with path.open(encoding='utf-8', newline='') as stream:
            reader = csv.DictReader(stream)
            if 'run_index' not in (reader.fieldnames or ()):
                continue
            for row in reader:
                identity = {key: int(row[key]) for key in _SEED_COLUMNS if row.get(key)}
                identity.update({key: row[key] for key in ('recording_id', 'source_recording_id') if row.get(key)})
                rows.append(identity)
    # Pending arms of this batch are checked together before publication and through
    # their own frozen packages. A registered arm is included in actual history.
    used = {replication._identity(job) for job in history}
    unique = {replication._identity(job): job for job in rows}
    return [job for identity, job in unique.items() if identity not in used]


def _check_slots(suite, _historical, evaluation):
    """Validate matched training scenes and permit seed reuse only for exact evaluation slots."""
    require = plan.parent.require
    unit, seed = suite['unit'], suite['baseline_seed']
    require(unit in plan.UNITS and seed == unit.rsplit('-', 1)[1], 'seed extension unit or schedule differs')
    layout = plan.locations(unit)
    jobs = suite['jobs']
    training = _source_jobs(plan.training_template(seed))
    require(suite['campaign_id'] == layout['campaign_id'] and len(jobs) == plan.CAP
            and len(training) == plan.TRAIN and len(evaluation) == plan.EVAL,
            'seed extension campaign or physical inventory differs')
    for ordinal, (job, original) in enumerate(zip(jobs, training + evaluation, strict=True)):
        require(job['run_index'] == layout['run_index_base'] + ordinal and job['ordinal'] == ordinal
                and job['actor_seed'] == suite['actor_seed']
                and all(job[key] == original[key] for key in (*PROFILE_FIELDS, *SCENE_FIELDS)),
                'seed extension slot, Actor identity or frozen RPC scene differs')
        if ordinal >= plan.TRAIN:
            require(all(job[key] == original[key] for key in SEED_FIELDS),
                    'seed extension evaluation dual seeds differ from the original slot')
    phase_recordings = {job['phase_id']: job['recording_id'] for job in jobs}
    recordings = set(phase_recordings.values())
    require(len(phase_recordings) == len(recordings) == plan.parent.UPDATES + len(plan.parent.EVAL_CASES)
            and all(isinstance(recording, str) and recording for recording in recordings)
            and all(job['recording_id'] == job['source_recording_id'] == phase_recordings[job['phase_id']]
                    for job in jobs), 'every seed extension phase requires an independent recording')
    return recordings


assert_available = bind(replication.assert_available, plan=plan, _check_slots=_check_slots,
                        _prepared_rows=_prepared_rows)


def budget_snapshot():
    """Merge only registered extension arms into the latest budget and retain their states."""
    snapshot = predecessor.budget_snapshot()
    for unit in plan.UNITS:
        layout = plan.locations(unit)
        if not Path(layout['budget_path']).is_file():
            continue
        suite, phases = plan.load_package(unit)
        assert_available([suite], snapshot)
        current = ExtensionLedger(suite, phases).snapshot()
        snapshot['jobs'].update(current['jobs'])
        snapshot['campaigns'].update(current['campaigns'])
        snapshot['maximum_total_shifts'] += plan.CAP
        snapshot['additional_reserved_shifts'] += plan.CAP
    return snapshot


class ExtensionLedger(WeakLedger):
    """Keep the established collection, commit and settlement state machine."""

    def __init__(self, suite, phases, *, path=None, approval=None):
        super().__init__(suite, phases, path=path, approval=approval)
        self.entry['format'] = FORMAT

    register = bind(WeakLedger.register, budget_snapshot=budget_snapshot)
    _register = bind(WeakLedger._register, assert_available=assert_available)
