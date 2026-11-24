"""Prepare ten fresh zero-shot shifts with the frozen Full s02/u32 Actor."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys


PLAN = Path(__file__).resolve().parent
GRAPH = PLAN.parent
ROOT = PLAN.parents[3]
OR_PLAN = GRAPH/'baseline_comparison_001/or_paper_comparison_001/paired10_001'
TEMPLATE = GRAPH/'fleet_outage_001'

for directory in (ROOT, TEMPLATE, OR_PLAN):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import fleet_plan as base
import fleet_ledger as original_ledger
import java_adapter as java
import paired10_campaign as previous
from weak_plan import bind
from python.offline_split import JAVA_MASK, java_random_states
from traditional_dispatch_v1.runtime import _load_actor


CAMPAIGN = 'bg128-full-s02-u32-fleet6-zero-shot10-20261124-001'
FIRST_RUN = 10042001
COUNT = SLOTS = 10
PREPARED = PLAN/'prepared_001'
SUITE = PREPARED/'suite.json'
OUTPUT = ROOT/'runs'/CAMPAIGN
CONTROL = OUTPUT/'control'
EVENTS = OUTPUT/'anylogic_events'
REPORT = base.REPORT.parent/PLAN.name
REPORT_FILENAME = base.REPORT_FILENAME
LEDGER = OUTPUT/'budget.json'
AUTHORITY = OUTPUT/'launch/run_authority.json'
LOCKS = previous.LOCKS
PYTHON, CORE, EXECUTION = base.PYTHON, base.CORE, base.EXECUTION
FLEET = base.FLEET
LAYOUT = dict(python_output=str(OUTPUT), control_directory=str(CONTROL),
              report_directory=str(REPORT), event_root=str(EVENTS))
PAIRING = dict(base.PAIRING, slots=SLOTS)
ACTOR_SHA256 = 'e1575fec566c647a882c63934ec57c908cef28d43f076431ef9103cc9b73e0a9'
reference, require, load_reference = base.reference, base.require, base.load_reference
encoded = base.encoded
preceding_snapshot = previous.budget_snapshot


def source_contract():
    refs, learner, rpc, scene = base.source_contract()
    require(refs['actor']['sha256'] == ACTOR_SHA256, 'the frozen Full s02/u32 Actor differs')
    refs['preceding_budget_source'] = reference(OR_PLAN/'paired10_campaign.py')
    return refs, learner, rpc, scene


seed = bind(base.seed, CAMPAIGN=CAMPAIGN)
job_for = bind(base.job_for, CAMPAIGN=CAMPAIGN, FIRST_RUN=FIRST_RUN, OUTPUT=OUTPUT, EVENTS=EVENTS, seed=seed)
validate_jobs = bind(base.validate_jobs, COUNT=COUNT, FIRST_RUN=FIRST_RUN, seed=seed)
phase_path = bind(base.phase_path, PREPARED=PREPARED)
config_for = bind(base.config_for, CAMPAIGN=CAMPAIGN)
render = bind(java.render, PLAN=PLAN, COUNT=COUNT, CONTROL=CONTROL, SUITE=SUITE, CAMPAIGN=CAMPAIGN,
              validate_jobs=validate_jobs)
verify_java = bind(java.verify_java, COUNT=COUNT)


def source_pins():
    files = ('campaign.py', 'launch.py', 'fleet_runtime.py', 'verify_results.py')
    return [*base.source_pins(), reference(TEMPLATE/'prepare.py'), reference(previous.__file__),
            *(reference(PLAN/name) for name in files)]


_load_suite = bind(base.load_suite, SUITE=SUITE, CAMPAIGN=CAMPAIGN, LAYOUT=LAYOUT, PAIRING=PAIRING,
                   COUNT=COUNT, SLOTS=SLOTS, PREPARED=PREPARED, source_contract=source_contract,
                   source_pins=source_pins, validate_jobs=validate_jobs, job_for=job_for,
                   config_for=config_for, phase_path=phase_path)


def load_suite(path=SUITE, *, check_sources=True):
    return _load_suite(path, check_sources=check_sources)


def check_fresh(suite, historical):
    require(CAMPAIGN not in historical['campaigns'] and historical['active_h2_reserved_shifts'] == 0,
            'six-truck activity already registered or historical budget policy differs')
    jobs = suite['jobs']
    validate_jobs(jobs)
    _, learner, rpc, scene = source_contract()
    for ordinal, job in enumerate(jobs):
        require({key: value for key, value in job.items() if key != 'phase_config'}
                == job_for(ordinal, learner, rpc, scene), 'noncanonical zero-shot physical job')
    # The previous inventory scans the full BG128 tree, including both old fleet
    # drafts and the GAT1 sidecars. Exclude only the exact current suite file.
    rows = _prior_rows(historical)
    runs = {row.get('run_index') for row in rows}
    recordings = {row.get(key) for row in rows for key in ('recording_id', 'source_recording_id')}
    require(all(job['run_index'] not in runs and job['recording_id'] not in recordings for job in jobs),
            'run or recording identity overlaps an existing or prepared slot')
    existing_seeds = {row[key] & JAVA_MASK for row in rows for key in original_ledger.SEED_FIELDS
                      if type(row.get(key)) is int and row[key] > 0}
    used_environment = {row['experiment_seed'] for row in rows
                        if type(row.get('experiment_seed')) is int and row['experiment_seed'] > 0}
    existing_java = {state for seed_value in used_environment for state in java_random_states(seed_value).values()}
    env = {job['experiment_seed'] for job in jobs}
    actor = {job['actor_sampling_seed'] for job in jobs}
    states = {value for seed_value in env for value in java_random_states(seed_value).values()}
    require((env | actor).isdisjoint(existing_seeds | {learner['seed'] & JAVA_MASK})
            and len(states) == COUNT*5 and states.isdisjoint(existing_java), 'seed or effective Java RNG state reused')
    before = historical['maximum_total_shifts']
    return dict(occupied_before=before, maximum_total_if_registered=before+COUNT,
                historical_and_prepared_identities_checked=len(rows), new_physical_jobs=COUNT,
                new_environment_seeds=len(env), new_actor_sampling_seeds=len(actor), new_java_rng_states=len(states))


_prior_rows = bind(previous._prior_rows, SUITE=SUITE)
authority_for = bind(original_ledger.authority_for, CAMPAIGN=CAMPAIGN, COUNT=COUNT, SUITE=SUITE)


class Ledger(original_ledger.Ledger):
    register = bind(original_ledger.Ledger.register, LEDGER=LEDGER, AUTHORITY=AUTHORITY, SUITE=SUITE,
                    check_fresh=check_fresh, preceding_snapshot=preceding_snapshot, authority_for=authority_for)
    snapshot = bind(original_ledger.Ledger.snapshot, LEDGER=LEDGER, AUTHORITY=AUTHORITY, SUITE=SUITE,
                    authority_for=authority_for)
    _transition = bind(original_ledger.Ledger._transition, LEDGER=LEDGER)


budget_snapshot = bind(original_ledger.budget_snapshot, LEDGER=LEDGER, CAMPAIGN=CAMPAIGN, COUNT=COUNT,
                       preceding_snapshot=preceding_snapshot, Ledger=Ledger, load_suite=load_suite,
                       check_fresh=check_fresh)


def prepare():
    spec = importlib.util.spec_from_file_location('fleet_template_prepare', TEMPLATE/'prepare.py')
    preparation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(preparation)
    # Validate model bytes and inference compatibility before freezing launch files.
    refs, learner, _, _ = source_contract()
    actor, _ = _load_actor(dict(refs, learner_config=learner))
    require(not actor.training and all(not parameter.requires_grad for parameter in actor.parameters()),
            'the Actor must be frozen for zero-shot evaluation')
    freeze = bind(preparation.prepare, CAMPAIGN=CAMPAIGN, COUNT=COUNT, SLOTS=SLOTS, PREPARED=PREPARED,
                  SUITE=SUITE, OUTPUT=OUTPUT, LAYOUT=LAYOUT, PAIRING=PAIRING, source_contract=source_contract,
                  source_pins=source_pins, job_for=job_for, phase_path=phase_path, config_for=config_for,
                  render=render, verify_java=verify_java, load_suite=load_suite, check_fresh=check_fresh,
                  budget_snapshot=preceding_snapshot)
    return freeze()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', required=True, action='store_true')
    parser.parse_args()
    print(json.dumps(prepare(), ensure_ascii=False))
