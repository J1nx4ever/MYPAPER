"""Preserve used identities and transfer only the 657 proven-unused reservations."""

from pathlib import Path

from . import prepare as plan
from .. import budget as source
from .. import prepare as source_plan
from ....paired_inference_001 import budget as base
from business_graph_summary_campaign_v1.launch import _launch_lock
from business_graph_summary_campaign_v1.plan_io import load_reference, reference
from replica_budget import ReplicaLedger
from weak_plan import bind

LOCKS = source.LOCKS
authority_path = source.authority_path
preceding_snapshot = source.budget_snapshot
rows_for_collision = bind(source.rows_for_collision, plan=plan,
                          _prior_rows=bind(source.rows_for_collision.__globals__['_prior_rows'], SUITE=plan.SUITE))
require = plan.require


def assert_available(suite, history):
    old, _ = plan.original_package()
    prior_jobs = old['jobs']
    require(reference(source_plan.SUITE) == suite['previous_evaluation']
            and reference(source_plan.CONTROL / 'STOP.properties') == suite['previous_stop']
            and plan.CAMPAIGN not in history['campaigns']
            and history['active_h2_reserved_shifts'] == 0
            and not history['campaigns'][source_plan.CAMPAIGN]['commits'],
            'original STOP or evaluation scope changed')
    prior = history['jobs']
    states = [prior[str(job['run_index'])]['status'] for job in prior_jobs]
    require(states == ['COMPLETE'] * plan.COMPLETE + ['UNKNOWN'] + ['PLANNED'] * (plan.COUNT - 1)
            and all(prior[str(job['run_index'])]['identity'] == job for job in prior_jobs)
            and (source_plan.CONTROL / 'java_claims' / f'run_{plan.INTERRUPTED}.once.claim').is_file()
            and suite['superseded_runs'] == [job['run_index'] for job in prior_jobs[plan.COMPLETE + 1:]],
            'expected exactly 42 complete, one interrupted and 657 unused original identities')
    for job in prior_jobs[plan.COMPLETE + 1:]:
        run = job['run_index']
        unused = [source_plan.CONTROL / 'java_claims' / f'run_{run}.once.claim',
                  source_plan.CONTROL / 'ready' / f'run_{run}.properties',
                  Path(job['python_output']) / 'episodes' / f'run_{run}',
                  Path(job['python_output']) / 'material' / f'run_{run}.h5',
                  Path(job['material_event_bundle']), Path(job['event_bundle'])]
        require(not any(path.exists() for path in unused), f'original inactive slot has physical artifacts: {run}')
    jobs, replacements = suite['jobs'], prior_jobs[plan.COMPLETE:]
    require(len(jobs) == len(replacements) == plan.COUNT
            and suite['frozen_actor'] == old['frozen_actor'] and suite['frozen_checkpoint'] == old['frozen_checkpoint'],
            'recovery Actor or physical scope differs')
    recordings = {job['recording_id'] for job in jobs}
    for i, (job, replaced) in enumerate(zip(jobs, replacements, strict=True)):
        require(job['ordinal'] == i and job['run_index'] == plan.FIRST_RUN + i
                and job['recording_id'] == job['source_recording_id'] == f"{plan.CAMPAIGN}-{job['phase_id']}"
                and all(job[key] == replaced[key] for key in source_plan.PAIR_KEYS),
                'recovery slot differs from its original Full S02 paired condition')
    rows = rows_for_collision(history)
    historical_runs = {row.get('run_index') for row in rows}
    historical_recordings = {row.get(key) for row in rows for key in ('recording_id', 'source_recording_id')}
    require(len(recordings) == 7 and recordings.isdisjoint(historical_recordings)
            and historical_runs.isdisjoint(job['run_index'] for job in jobs),
            'fresh physical run or recording collides with history')
    before = history['maximum_total_shifts']
    return dict(allocated_before=before, new_physical_identities=plan.COUNT,
                superseded_unused_reservations=plan.COUNT-1, allocated_after=before+1,
                registered_gross_after=history.get('registered_gross_shifts', before)+plan.COUNT,
                retained_complete=plan.COMPLETE, retained_unknown=1,
                replacement_of=plan.INTERRUPTED, row_count_checked=len(rows))


_authority = bind(base.authorization, plan=plan)


def authorization(suite):
    return dict(_authority(suite), schema='bg128-step-matched-reinforce-u32-evaluation-recovery-authority-v1',
                previous_evaluation=suite['previous_evaluation'], previous_stop=suite['previous_stop'],
                comparison_shifts=suite['comparison_shifts'], superseded_runs=suite['superseded_runs'],
                interrupted_run=suite['interrupted_run'], frozen_checkpoint=suite['frozen_checkpoint'])


class InferenceLedger(source.InferenceLedger):
    __init__ = bind(base.InferenceLedger.__init__, plan=plan, authorization=authorization)
    snapshot = bind(base.InferenceLedger.snapshot, plan=plan)

    def register(self):
        with _launch_lock(base.previous.SHARED_BUDGET_LOCK):
            require(not self.path.exists(), 'inference recovery is already registered')
            for key in ('approval_reference', 'suite_reference'):
                ref = self.entry[key]
                require(reference(ref['path']) == ref, 'recovery approval/package bytes differ')
            register = bind(ReplicaLedger._register,
                            assert_available=lambda suites, state: assert_available(suites[0], state))
            register(self, preceding_snapshot())


def budget_snapshot():
    snapshot = preceding_snapshot()
    if plan.LEDGER.is_file():
        suite, phases = plan.load_package()
        audit = assert_available(suite, snapshot)
        current = InferenceLedger(suite, phases).snapshot()
        snapshot['campaigns'].update(current['campaigns'])
        snapshot['jobs'].update(current['jobs'])
        for run in suite['superseded_runs']:
            snapshot['jobs'][str(run)]['superseded_by_campaign'] = plan.CAMPAIGN
        snapshot['registered_gross_shifts'] = audit['registered_gross_after']
        snapshot['maximum_total_shifts'] = audit['allocated_after']
        snapshot['additional_reserved_shifts'] += 1
    return snapshot
