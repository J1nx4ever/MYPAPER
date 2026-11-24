"""Compare four six-active compositions using matched environment and Actor seeds."""

from math import fsum
from pathlib import Path
from types import FunctionType

import numpy as np
import torch

from change_plans.token_attention_001.business_graph_summary_009.fleet_outage_001 import fleet_runtime as prior
from python.dispatch_bridge import ProtocolError
from python.generated import dispatch_pb2
from traditional_dispatch_v1 import runtime as shared
from traditional_dispatch_v2 import runtime as v2


SUITE_SCHEMA = 'bg128-fleet-outage-comparison-v1'
REPORT_SCHEMA = 'bg128-fleet-outage-comparison-report-v1'
REPORT_FILENAME = 'fleet_outage_comparison.json'
EVALUATION_SLOTS = 5
SCENARIOS = ('heavy_out', 'balanced_out', 'small_out', 'medium_out')
MASKS = (448, 292, 7, 56)
BASELINE = 'balanced_out'
FLEET_PREFIX = 'config.fleet.'
METRICS = ('combined_cost', 'grade_cost', 'production_cost', 'waiting_cost', 'ore_tons', 'waste_tons',
           'total_wait_seconds', 'complete_task', 'ore_target_met', 'waste_target_met')


def _rebind(function, bindings):
    rebound = FunctionType(function.__code__, bindings, argdefs=function.__defaults__, closure=function.__closure__)
    rebound.__kwdefaults__ = function.__kwdefaults__
    return rebound


def _condition(mask):
    disabled = tuple(truck for truck in range(9) if mask & (1 << truck))
    active = tuple(truck for truck in range(9) if truck not in disabled)
    fields = dict(prior.OUTAGE_FIELDS, **{'config.fleet.disabled_mask': mask})
    return dict(DISABLED=disabled, ACTIVE=active, OUTAGE_FIELDS=fields, _DISABLED_REF=np.array(disabled) + 1)


def _non_fleet_configuration(components):
    return {key: value for key, value in components.items()
            if key.startswith('config.') and not key.startswith(FLEET_PREFIX)}


def _bridge_for(mask, expected_configuration):
    bindings = dict(vars(prior), **_condition(mask))
    bindings['validate_outage'] = _rebind(prior.validate_outage, bindings)
    validate = _rebind(prior.FleetOutageBridge._validate_request, bindings)

    class PairedFleetBridge(prior.FleetOutageBridge):
        outage_receipt = _rebind(prior.FleetOutageBridge.outage_receipt, bindings)

        def _validate_request(self, request, *, grade_bounds=True, config=None):
            validate(self, request, grade_bounds=grade_bounds, config=config)
            if (request.kind == dispatch_pb2.EPISODE_START and expected_configuration is not None
                    and _non_fleet_configuration(request.reward_components) != expected_configuration):
                raise ProtocolError('same-slot actual RPC configuration differs outside fleet outage fields')

    return PairedFleetBridge


def _run_shift(suite, startup, ledger, job, expected_configuration, actor, actor_digest):
    bridge = _bridge_for(job['fleet']['disabled_mask'], expected_configuration)
    bindings = dict(vars(v2), DispatchCallbackBridge=bridge, _collect=prior._collect_with_outage)
    execute = _rebind(v2._run_shift, bindings)
    return execute(suite, startup, ledger, job, None, actor, actor_digest)


def _validate_suite(suite):
    jobs = suite['jobs']
    width = len(SCENARIOS)
    count = EVALUATION_SLOTS * width
    shared._require(suite['schema'] == SUITE_SCHEMA and suite['evaluation_slots'] == EVALUATION_SLOTS
                    and suite['physical_shifts'] == len(jobs) == count
                    and suite['training_shifts'] == suite['optimizer_updates'] == 0,
                    'five frozen four-scenario slots required')
    for field in ('run_index', 'recording_id', 'python_output', 'event_bundle'):
        shared._require(len({job[field] for job in jobs}) == count, f'physical identity reuses {field}')
    pairing = suite['pairing']
    shared._require(pairing['scenarios'] == list(SCENARIOS) and pairing['policies'] == ['bg128']
                    and pairing['share'] == ['experiment_seed', 'actor_sampling_seed']
                    and pairing['slots'] == EVALUATION_SLOTS, 'frozen outage pairing differs')
    seeds = []
    lineage = suite['learner_config']['seed']
    for ordinal, job in enumerate(jobs):
        slot, arm = divmod(ordinal, width)
        scenario, mask = SCENARIOS[arm], MASKS[arm]
        condition = _condition(mask)
        fleet = job['fleet']
        shared._require(job['ordinal'] == ordinal and job['pair_ordinal'] == slot
                        and job['policy_kind'] == 'bg128' and job['actor_sampling_seed_consumed'] is True
                        and job['checkpoint_update'] == 32 and job['actor_seed'] == lineage
                        and fleet == suite['fleet'][scenario] and fleet['scenario_id'] == scenario
                        and job['scene_id'] == fleet['scene_id'] and fleet['physical_count'] == 9
                        and fleet['outage_version'] == 1 and fleet['disabled_mask'] == mask
                        and fleet['disabled_truck_ids'] == list(condition['DISABLED']) and fleet['active_count'] == 6
                        and all(job['expected_rpc_config'].get(key) == value
                                for key, value in condition['OUTAGE_FIELDS'].items()),
                        'outage job does not match its frozen composition or Actor')
        anchor = jobs[slot * width]
        seed_fields = ('experiment_seed', 'actor_sampling_seed')
        shared._require(all(job[key] == anchor[key] for key in seed_fields)
                        and _non_fleet_configuration(job['expected_rpc_config'])
                        == _non_fleet_configuration(anchor['expected_rpc_config'])
                        and job['runtime_scene_sha256'] == anchor['runtime_scene_sha256'],
                        'same-slot seed or planned non-fleet condition differs')
        if arm == 0:
            seeds.extend(job[key] for key in seed_fields)
    shared._require(all(type(seed) is int and seed > 0 for seed in seeds)
                    and len(set(seeds)) == 2 * EVALUATION_SLOTS and lineage not in seeds,
                    'outage evaluation requires distinct environment and Actor sampling seeds across slots')


def _verify_pair(shifts, refs, jobs):
    for row, ref, job in zip(shifts, refs, jobs, strict=True):
        condition = _condition(job['fleet']['disabled_mask'])
        fields = ('run_index', 'recording_id', 'phase_config', 'pair_ordinal', 'scene_id',
                  'policy_kind', 'experiment_seed', 'actor_sampling_seed')
        shared._require(row['status'] == 'COMPLETE' and row['normal_completion'] is True
                        and row['completed_shifts'] == 1 and row['optimizer_updates'] == 0
                        and all(row[key] == job[key] for key in fields)
                        and Path(ref['path']) == Path(job['python_output']) / 'evaluation_receipt.json',
                        'outage receipt differs from its frozen slot')
        rpc = row['actual_rpc_configuration']
        witness = row['engineering']['fleet_outage']
        shared._require(all(rpc.get(key) == value for key, value in condition['OUTAGE_FIELDS'].items())
                        and witness['disabled_ids'] == list(condition['DISABLED'])
                        and witness['active_ids'] == list(condition['ACTIVE']),
                        'outage receipt does not attest the assigned composition')
    anchor = shifts[0]
    settings = _non_fleet_configuration(anchor['actual_rpc_configuration'])
    rng = anchor['engineering']['actor_sampling_rng']['initial_sha256']
    shared._require(all(_non_fleet_configuration(row['actual_rpc_configuration']) == settings
                        and row['engineering']['actor_sampling_rng']['initial_sha256'] == rng for row in shifts),
                    'same-slot actual configuration or Actor RNG start differs')


def _outcome(row):
    physical = row['physical']
    return dict(run_index=row['run_index'], **{key: physical[key] for key in METRICS},
                window_grades=[window['grade'] for window in physical['windows']],
                compliant_windows=sum(window['compliant'] for window in physical['windows']),
                engineering_passed=row['engineering_acceptance']['passed'])


def _comparison(groups, suite):
    summaries = {}
    for arm, (scenario, mask) in enumerate(zip(SCENARIOS, MASKS, strict=True)):
        summarize = _rebind(prior._summary, dict(vars(prior), **_condition(mask)))
        summaries[scenario] = dict(fleet=suite['fleet'][scenario], **summarize([group[arm] for group in groups]))

    slots = []
    for index, group in enumerate(groups):
        outcomes = {scenario: _outcome(row) for scenario, row in zip(SCENARIOS, group, strict=True)}
        baseline = outcomes[BASELINE]
        differences = {scenario: {key: outcome[key] - baseline[key] for key in METRICS}
                       for scenario, outcome in outcomes.items() if scenario != BASELINE}
        slots.append(dict(pair_ordinal=index, experiment_seed=group[0]['experiment_seed'],
                          actor_sampling_seed=group[0]['actor_sampling_seed'],
                          outcomes=outcomes, difference_from_balanced=differences))

    paired = {}
    for scenario in SCENARIOS:
        if scenario == BASELINE:
            continue
        differences = [slot['difference_from_balanced'][scenario] for slot in slots]
        paired[scenario] = dict(
            mean_difference={key: fsum(item[key] for item in differences) / len(slots) for key in METRICS},
            joint_cost_lower_slots=sum(item['combined_cost'] < 0 for item in differences),
            joint_cost_higher_slots=sum(item['combined_cost'] > 0 for item in differences),
            joint_cost_tied_slots=sum(item['combined_cost'] == 0 for item in differences),
            only_scenario_complete=sum(item['complete_task'] == 1 for item in differences),
            only_balanced_complete=sum(item['complete_task'] == -1 for item in differences),
        )
    return dict(reference_scenario=BASELINE, summaries=summaries, slots=slots,
                paired_differences_from_balanced=paired)


def run(suite, ledger, *, progress=None):
    """Run four fleet compositions for every one of the five paired seeds."""
    _validate_suite(suite)
    jobs = suite['jobs']
    control = Path(suite['control_directory'])
    report_path = Path(suite['report_directory']) / REPORT_FILENAME
    slots = ledger.snapshot()['slots']
    shared._require(not report_path.exists() and not (control / 'STOP.properties').exists()
                    and set(slots) == {str(job['run_index']) for job in jobs}
                    and all(slots[str(job['run_index'])]['status'] == 'PLANNED'
                            and slots[str(job['run_index'])]['identity'] == job for job in jobs),
                    'fleet comparison requires the exact fresh ledger roster')

    shared._check_startup(suite)
    torch.set_num_threads(suite.get('cpu_threads', 2))
    actor, actor_digest = shared._load_actor(suite)
    startup = dict(control_directory=str(control), campaign_id=suite['campaign_id'],
                   master_plan_sha256=suite['master_plan']['sha256'])
    width = len(SCENARIOS)
    groups, receipts = [], []
    for offset in range(0, len(jobs), width):
        config = None
        group, refs = [], []
        block = jobs[offset:offset + width]
        for job in block:
            row, ref = _run_shift(suite, startup, ledger, job, config, actor, actor_digest)
            if config is None:
                config = _non_fleet_configuration(row['actual_rpc_configuration'])
            group.append(row)
            refs.append(ref)
            print(f"COMPLETE run_index={job['run_index']} slot={job['pair_ordinal']} "
                  f"scenario={job['fleet']['scenario_id']}", flush=True)
            if progress is not None:
                progress(offset + len(group), job)
        _verify_pair(group, refs, block)
        groups.append(group)
        receipts.extend(refs)

    slots = ledger.snapshot()['slots']
    shared._require(all(slots[str(job['run_index'])]['status'] == 'COMPLETE'
                        and slots[str(job['run_index'])]['receipt'] == ref
                        for job, ref in zip(jobs, receipts, strict=True)), 'outage ledger lacks complete receipts')
    summary = _comparison(groups, suite)
    report = dict(schema=REPORT_SCHEMA, status='COMPLETE', campaign_id=suite['campaign_id'],
                  evaluation_slots=len(groups), completed_shifts=len(jobs), physical_shifts=len(jobs),
                  optimizer_updates=0, method=suite['method'], pairing=suite['pairing'], fleet=suite['fleet'],
                  shifts=[row for group in groups for row in group], receipts=receipts, **summary)
    return shared._publish_json(report_path, report)
