"""Freeze new physical identities for the interrupted paired S02 evaluation."""

from functools import lru_cache
from pathlib import Path

from .. import prepare as source
from ....paired_inference_001 import prepare as base
from business_graph_summary_campaign_v1.plan_io import reference
from weak_plan import bind

PLAN = Path(__file__).resolve().parent
ROOT = source.ROOT
CAMPAIGN = 'bg128-reinforce-ppo-steps-s02-u32-paired700-recovery-20261128-001'
FIRST_RUN = 10097001
COMPLETE = 42
INTERRUPTED = 10096043
COUNT = 658
RPC_PORT = source.RPC_PORT
PREPARED = PLAN / 'prepared_001'
SUITE = PREPARED / 'suite.json'
OUTPUT = ROOT / 'runs' / CAMPAIGN
MATERIAL = OUTPUT.with_name(CAMPAIGN + '_materials')
CONTROL = OUTPUT / 'control'
REPORT = ROOT / 'reports' / PLAN.relative_to(ROOT / 'change_plans')
LEDGER = OUTPUT / 'budget.sqlite'
original, require = source.original, source.require
verify_physical_starts = source.verify_physical_starts


@lru_cache(maxsize=1)
def original_package():
    return source.load_package()


def documents():
    stopped, original_phases = original_package()
    require(len(stopped['jobs']) == COMPLETE + COUNT and stopped['jobs'][COMPLETE]['run_index'] == INTERRUPTED,
            'original evaluation boundary changed')
    fixed = {key: value for key, value in stopped.items()
             if key not in ('phase_configs', 'phases', 'jobs', 'source_pins', 'master_plan', 'scenario_table',
                            'java_helper', 'identities', 'java_validation')}
    fixed.update(schema='bg128-step-matched-reinforce-u32-recovery-v1', campaign_id=CAMPAIGN,
                 previous_evaluation=reference(source.SUITE),
                 previous_stop=reference(source.CONTROL / 'STOP.properties'),
                 previous_completed_count=COMPLETE, interrupted_run=INTERRUPTED,
                 comparison_shifts=COMPLETE + COUNT, evaluation_shifts=COUNT, new_shift_cap=COUNT,
                 superseded_runs=[job['run_index'] for job in stopped['jobs'][COMPLETE + 1:]],
                 arms=[dict(stopped['arms'][0], python_output=str(OUTPUT / 'evaluation'))],
                 report_directory=str(REPORT), python_output=str(OUTPUT), material_root=str(MATERIAL),
                 control_directory=str(CONTROL), budget_path=str(LEDGER), suite_path=str(SUITE))
    phases, ordinal = [], 0
    for previous in original_phases:
        phase_id = previous['phase_id']
        recording = f'{CAMPAIGN}-{phase_id}'
        material = MATERIAL / phase_id
        events = CONTROL / 'legacy_unused' / phase_id
        out = OUTPUT / 'evaluation' / previous['scenario_id'] / previous['policy_arm']
        jobs = []
        for prior_job in previous['jobs']:
            if prior_job['ordinal'] < COMPLETE:
                continue
            run = FIRST_RUN + ordinal
            group = f"{CAMPAIGN}-{prior_job['scene_id']}-{prior_job['pair_ordinal']:04d}"
            leaf = f"dispatch-production-wait-shortfall-v2-run-{run}-seed-{prior_job['experiment_seed']}"
            jobs.append(dict(prior_job, ordinal=ordinal, run_index=run, scene_group=group,
                             realization_group_id=group, recording_id=recording, source_recording_id=recording,
                             python_output=str(out), event_directory=str(events), event_bundle=str(events / leaf),
                             material_event_directory=str(material), material_event_bundle=str(material / leaf),
                             material_journal=str(material / leaf / 'material_events.bin')))
            ordinal += 1
        cfg = dict(previous, jobs=jobs, campaign_id=CAMPAIGN, recording_id=recording,
                   python_output=str(out), event_directory=str(events))
        original.check_phase(cfg)
        phases.append(cfg)
    require(ordinal == COUNT and [len(phase['jobs']) for phase in phases] == [58] + [100] * 6,
            'recovery slot count differs')
    return fixed, phases


def source_pins():
    pins = source.source_pins()
    files = [PLAN / name for name in ('prepare.py', 'budget.py', 'run.py', 'runtime.py', 'prepare_java.py')]
    pins.update({str(path): reference(path)['sha256'] for path in files})
    return dict(sorted(pins.items()))


load_package = bind(source.load_package, documents=documents, SUITE=SUITE,
                    PREPARED=PREPARED, source_pins=source_pins)
prepare = bind(base.prepare, __package__=__package__, CAMPAIGN=CAMPAIGN, FIRST_RUN=FIRST_RUN, COUNT=COUNT,
               PREPARED=PREPARED, SUITE=SUITE, OUTPUT=OUTPUT, MATERIAL=MATERIAL,
               CONTROL=CONTROL, REPORT=REPORT, LEDGER=LEDGER, documents=documents,
               load_package=load_package, source_pins=source_pins, verify_physical_starts=verify_physical_starts)
