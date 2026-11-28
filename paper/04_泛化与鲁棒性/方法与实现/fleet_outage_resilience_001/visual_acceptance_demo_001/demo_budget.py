"""Admission and settlement of the single visual fleet-outage acceptance shift."""
from pathlib import Path
import sys

FLEET_PLAN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FLEET_PLAN))
sys.path.insert(0, str(FLEET_PLAN/'replay_repair_001'))

import resilience_budget as ledger
from recovery_budget import budget_snapshot, ledger_rows
from resilience_campaign import ROOT, reference
from weak_plan import bind

CAMPAIGN = 'bg128-fleet-outage-visual-demo-20261124-001'
RUN = 10045001


def sole_job(suite):
    jobs = suite['jobs']
    output = ROOT/'runs'/CAMPAIGN
    if suite['campaign_id'] != CAMPAIGN or Path(suite['output']) != output or len(jobs) != 1:
        raise ValueError('expected one visual acceptance shift in its own campaign')
    job = jobs[0]
    fixed = dict(run_index=RUN,ordinal=0,role='eval',split='visual_demo',condition='outage',
                 arm='dynamic',update=9,pair_ordinal=-1,recording_id=f'{CAMPAIGN}-{RUN}',
                 source_recording_id=f'{CAMPAIGN}-{RUN}',
                 python_output=str(output/'shifts'/f'run_{RUN}'),
                 event_directory=str(output/'anylogic_events'/f'run_{RUN}'))
    if any(job.get(key)!=value for key,value in fixed.items()):
        raise ValueError('visual acceptance identity or output directory differs')
    return job


def check_fresh(suite):
    """Check registered history and frozen future identities, without reserving slots."""
    job = sole_job(suite)
    history = budget_snapshot()
    if CAMPAIGN in history['campaigns']:
        raise ValueError('visual acceptance campaign is already registered')
    if Path(job['python_output']).exists() or Path(job['event_directory']).exists():
        raise ValueError('visual acceptance artifacts already exist')
    prior = bind(ledger.previous.paired10._prior_rows,SUITE=Path(suite['suite_path']))(history)
    prior.extend(row['identity'] for row in history['jobs'].values() if isinstance(row.get('identity'),dict))
    runs = {row.get('run_index') for row in prior}
    recordings = {row.get(key) for row in prior for key in ('recording_id','source_recording_id')}
    if job['run_index'] in runs or job['recording_id'] in recordings:
        raise ValueError('visual acceptance run or recording overlaps a registered or prepared shift')
    fields = ('actor_seed','experiment_seed','actor_sampling_seed')
    old_seeds = {row[k]&ledger.JAVA_MASK for row in prior for k in fields
                 if type(row.get(k)) is int and row[k]>0}
    # actor_seed only describes the frozen checkpoint's initialization here.
    env, sampling = (job[key] for key in ('experiment_seed','actor_sampling_seed'))
    if (any(type(v) is not int or not 0<v<=ledger.JAVA_MASK for v in (env,sampling))
            or env==sampling or not {env,sampling}.isdisjoint(old_seeds)):
        raise ValueError('visual acceptance seed collides with a registered or prepared seed')
    old_env_seeds = {row['experiment_seed'] for row in prior
                     if type(row.get('experiment_seed')) is int and row['experiment_seed']>0}
    old_states = {state for seed in old_env_seeds for state in ledger.java_random_states(seed).values()}
    states = set(ledger.java_random_states(env).values())
    if len(states)!=5 or not states.isdisjoint(old_states):
        raise ValueError('visual acceptance Java stream overlaps a registered or prepared stream')
    before = history['maximum_total_shifts']
    return dict(registered_before=before,additional_if_started=1,total_if_started=before+1,
                new_environment_seeds=1,new_actor_sampling_seeds=1,new_java_rng_states=len(states))


# Reuse the same table schema, shared launch lock, transactional staging and
# historical scanning as the resilience campaigns. This demo has its own validator.
_register = bind(ledger.register,check_fresh=check_fresh)


def register(suite):
    sole_job(suite)
    _register(suite)
    state_view(suite)


def state_view(suite):
    job = sole_job(suite)
    path = ledger.ledger_path(suite)
    if not path.exists():
        return {}
    name,pin,rows = ledger_rows(path)
    if name!=CAMPAIGN or pin!=reference(suite['suite_path']) or set(rows)!={RUN}:
        raise ValueError('visual acceptance ledger does not match the frozen package')
    identity,state,evidence = rows[RUN]
    if identity!=ledger.identity(job):
        raise ValueError('visual acceptance ledger identity differs')
    return {RUN:(state,evidence)}


def claim(suite,job):
    state = state_view(suite)
    if not state or state[RUN][0]!='PLANNED':
        raise ValueError('visual acceptance shift must be registered and unused before claim')
    ledger.transition(suite,job,'PLANNED','CLAIMED')


def complete(suite,job,receipt_path):
    ledger.complete(suite,job,receipt_path)


def unknown(suite,job,reason):
    ledger.unknown(suite,job,reason)
