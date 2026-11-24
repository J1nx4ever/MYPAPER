"""Execute 100 fresh nominal slots through the sealed V2 single-shift runtime."""
from pathlib import Path

import torch

from traditional_dispatch_v1 import runtime as shared
from traditional_dispatch_v2 import runtime as v2

from formal_report import summarize


SUITE_SCHEMA = 'bg128-traditional-paired100-v2'
REPORT_FILENAME = 'paired_comparison.json'
EVALUATION_SLOTS = 100


def _validate_roster(suite):
    jobs = suite['jobs']
    width = len(v2.POLICIES)
    _require = shared._require
    _require(suite['schema'] == SUITE_SCHEMA and suite['evaluation_slots'] == EVALUATION_SLOTS
             and suite['physical_shifts'] == len(jobs) == EVALUATION_SLOTS * width,
             'frozen 100-slot five-arm suite required')
    for key in ('run_index', 'recording_id', 'python_output', 'event_bundle'):
        _require(len({job[key] for job in jobs}) == len(jobs), f'repeated physical {key}')
    _require(len({job['phase_config']['path'] for job in jobs}) == len(jobs), 'repeated physical phase path')

    environment_seeds, sampling_seeds = set(), set()
    for slot in range(EVALUATION_SLOTS):
        group = jobs[slot * width:(slot + 1) * width]
        _require(tuple(job['policy_kind'] for job in group) == v2.POLICIES, 'policy order differs')
        seed = group[0]['experiment_seed']
        _require(type(seed) is int and seed > 0 and seed not in environment_seeds,
                 'environment seed repeated outside this slot or invalid')
        environment_seeds.add(seed)
        for index, job in enumerate(group, slot * width):
            _require((job['ordinal'], job['pair_ordinal'], job['experiment_seed'], job['scene_id'])
                     == (index, slot, seed, 'nominal'), 'nominal physical job order or paired seed differs')
            sampling = job['actor_sampling_seed']
            if job['policy_kind'] == 'bg128':
                _require(type(sampling) is int and sampling > 0 and sampling not in sampling_seeds
                         and job['actor_seed'] > 0 and job['actor_sampling_seed_consumed'] is True
                         and job['checkpoint_update'] == 32, 'invalid or repeated BG128 private sampling identity')
                sampling_seeds.add(sampling)
            else:
                _require(job['actor_seed'] == sampling == job['checkpoint_update'] == 0
                         and job['actor_sampling_seed_consumed'] is False,
                         'deterministic arm unexpectedly uses BG128 identity')
    _require(environment_seeds.isdisjoint(sampling_seeds), 'environment seed collides with an Actor sampling seed')


def run(suite, ledger, *, progress=None):
    """Seal a fresh campaign; any already used slot requires reconciliation."""
    _validate_roster(suite)
    jobs = suite['jobs']
    _require = shared._require
    control = Path(suite['control_directory'])
    report = Path(suite['report_directory']) / REPORT_FILENAME
    _require(not report.exists() and not (control / 'STOP.properties').exists(),
             'existing report or STOP requires reconciliation')
    slots = ledger.snapshot()['slots']
    _require(set(slots) == {str(job['run_index']) for job in jobs}
             and all(slots[str(job['run_index'])]['status'] == 'PLANNED'
                     and slots[str(job['run_index'])]['identity'] == job for job in jobs),
             'formal comparison requires the exact fresh ledger roster')

    shared._check_startup(suite)
    torch.set_num_threads(suite.get('cpu_threads', 2))
    actor, actor_digest = shared._load_actor(suite)
    startup = dict(control_directory=str(control), campaign_id=suite['campaign_id'],
                   master_plan_sha256=suite['master_plan']['sha256'])
    shifts, receipts = [], []
    width = len(v2.POLICIES)
    for offset in range(0, len(jobs), width):
        anchor = None
        for job in jobs[offset:offset + width]:
            receipt, ref = v2._run_shift(suite, startup, ledger, job, anchor,
                                         actor if job['policy_kind'] == 'bg128' else None, actor_digest)
            if anchor is None:
                anchor = receipt
            shifts.append(receipt)
            receipts.append(ref)
            print(f"COMPLETE run_index={job['run_index']} pair={job['pair_ordinal']} "
                  f"policy={job['policy_kind']}", flush=True)
            if progress is not None:
                progress(len(shifts), job)

    shared._check_startup(suite)
    slots = ledger.snapshot()['slots']
    _require(all(slots[str(job['run_index'])]['status'] == 'COMPLETE'
                 and slots[str(job['run_index'])]['receipt'] == ref
                 for job, ref in zip(jobs, receipts, strict=True)), 'formal ledger does not attest all completed receipts')
    return shared._publish_json(report, summarize(shifts, receipts, suite=suite))
