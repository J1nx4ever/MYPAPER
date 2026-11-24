"""Collect 100 V3 DISPATCH shifts against the sealed formal V2 slots."""
from collections import Counter
import hashlib
import math
from pathlib import Path
import sys

import numpy as np

from baseline1_v1.collection import wait_for_java_start
from business_graph_summary_campaign_v1.plan_io import load_reference, reference
from change_plans.token_attention_001.business_graph_summary_009.baseline_comparison_001.dispatch_inspired_001 import (
    dispatch_runtime as collection,
)
from python.dispatch_bridge import DispatchCallbackBridge
from python.dispatch_env import DispatchEnv
from python.dispatch_service import create_server
from python.episode_runner import ParameterVariationEpisodeRunner
from traditional_dispatch_v1 import runtime as shared
from traditional_dispatch_v2 import runtime as v2

from v3_plan import COUNT, REPORT_FILENAME, SCHEMA, load_v2_completion


POLICY = 'dispatch_fifo_v3'
RECEIPT_SCHEMA = 'bg128-dispatch-fifo-v3-receipt-v1'
REPORT_SCHEMA = 'bg128-dispatch-fifo-v3-paired100-report-v1'
V2_POLICY = 'dispatch_inspired'
COUNTS = ('fifo_candidates', 'fifo_roots_completed', 'fifo_trials_nominated', 'fifo_trials_completed',
          'fifo_trials_rejected', 'fifo_trials_compared')


class _AnchoredEnv:
    """Verify the sealed V2 start before the shared collector can return an action."""

    def __init__(self, env, anchor):
        self.env, self.anchor = env, anchor
        self.telemetry = Counter()

    @property
    def _pending(self):
        return self.env._pending

    def _observe(self, info):
        self.telemetry['total_callbacks'] += 1
        self.telemetry['callbacks_with_both_flags'] += int(v2._native_fallback_telemetry(info))

    def reset(self):
        observation, info = self.env.reset()
        configuration = {key: value for key, value in info['reward_components'].items() if key.startswith('config.')}
        initial = dict(observation_sha256=hashlib.sha256(observation.tobytes()).hexdigest(),
                       mask=info['action_mask'], truck_id=info['truck_id'], sim_time_seconds=info['sim_time_seconds'])
        v2._verify_anchor(dict(experiment_seed=info['experiment_seed'], initial_state=initial,
                               actual_rpc_configuration=configuration), self.anchor)
        self._observe(info)
        return observation, info

    def step(self, action):
        transition = self.env.step(action)
        self._observe(transition[-1])
        return transition


class _PolicyEvidence:
    """Aggregate only acknowledged choices and keep one witness of FIFO evaluation."""

    def __init__(self, policy):
        self.policy = policy
        self.counts = Counter({name: 0 for name in COUNTS})
        self.cost_delta = []
        self.witness = None
        self.comparison_witness = None
        self.errors = Counter()
        self.requests = 0

    def choose(self, observation, mask):
        return self.policy.choose(observation, mask)

    def commit(self, choice):
        _require = shared._require
        self.policy.commit(choice)
        self.requests += 1
        for key in COUNTS:
            value = getattr(choice, key)
            _require(type(value) is int and value >= 0, 'invalid FIFO planning work count')
            self.counts[key] += value
        compared = choice.fifo_trials_compared
        _require(compared <= choice.fifo_trials_completed and
                 choice.fifo_trials_completed + choice.fifo_trials_rejected <= choice.fifo_trials_nominated,
                 'FIFO trial counts exceed completed or nominated work')
        _require((choice.matched_trucks > 1) == (compared > 0)
                 and type(choice.fifo_selected_from_cohort) is bool
                 and (not choice.fifo_selected_from_cohort or compared > 0),
                 'multitruck match requires a completed executable comparison')
        _require(choice.kind != 'planned' or choice.fifo_candidates == choice.fifo_roots_completed,
                 'planned FIFO decision lacks complete legal root coverage')
        restored = choice.fifo_guard_restored_root
        _require(type(restored) is bool and (not restored or (choice.timed_out and compared == 0)),
                 'invalid FIFO guard outcome')
        self.counts['fifo_guard_restored_root'] += int(restored)
        self.counts['fifo_requests_with_compared_trial'] += int(compared > 0)
        self.counts['fifo_selected_from_cohort_requests'] += int(choice.fifo_selected_from_cohort)
        self.counts['fifo_guidance_error_requests'] += int(bool(choice.fifo_guidance_error))
        self.counts['fifo_model_error_requests'] += int(bool(choice.fifo_model_error))
        for stage, error in (('guidance', choice.fifo_guidance_error), ('model', choice.fifo_model_error)):
            if error:
                self.errors[(stage, error)] += 1
        if choice.fifo_root_cost is not None and choice.fifo_selected_cost is not None:
            _require(math.isfinite(choice.fifo_root_cost) and math.isfinite(choice.fifo_selected_cost),
                     'nonfinite FIFO cost evidence')
            delta = choice.fifo_selected_cost - choice.fifo_root_cost
            _require(not restored or abs(delta) <= 1e-12, 'deadline guard did not restore a complete root choice')
            self.cost_delta.append(delta)
            self.counts['fifo_root_feasible_requests'] += int(choice.fifo_root_feasible is True)
            self.counts['fifo_selected_feasible_requests'] += int(choice.fifo_selected_feasible is True)
        if choice.fifo_trials_completed > 0:
            self.counts['fifo_requests_with_completed_trial'] += 1
            if self.witness is None or (compared and self.comparison_witness is None):
                record = dict(request_ordinal=self.requests - 1, action=choice.action,
                    selected_actions=list(choice.fifo_selected_actions), selected_truck_ids=list(choice.fifo_selected_truck_ids),
                    roots_completed=choice.fifo_roots_completed, trials_nominated=choice.fifo_trials_nominated,
                    trials_completed=choice.fifo_trials_completed, trials_compared=compared,
                    selected_from_cohort=choice.fifo_selected_from_cohort,
                    selected_feasible=choice.fifo_selected_feasible, selected_cost=choice.fifo_selected_cost)
                if self.witness is None:
                    self.witness = record
                if compared and self.comparison_witness is None:
                    self.comparison_witness = record


def _collect(env, job, method, anchor, policy):
    checked_env = _AnchoredEnv(env, anchor)
    evidence = _PolicyEvidence(policy)
    result = collection._collect(checked_env, job, method, evidence)
    engineering = result['engineering']
    shared._require(evidence.requests == engineering['decisions']
                    and checked_env.telemetry['total_callbacks'] == engineering['decisions'] + 1,
                    'applied V3 action and callback evidence differs')
    engineering['action_counts'].update(evidence.counts)
    engineering['native_fallback_telemetry'] = dict(checked_env.telemetry)
    engineering['first_completed_fifo_trial'] = evidence.witness
    engineering['first_compared_fifo_trial'] = evidence.comparison_witness
    engineering['fifo_planning_errors'] = [dict(stage=stage, reason=reason, requests=n)
                                           for (stage, reason), n in sorted(evidence.errors.items())]
    if evidence.cost_delta:
        engineering['fifo_selected_minus_root_common_cost'] = dict(
            n=len(evidence.cost_delta), mean=math.fsum(evidence.cost_delta) / len(evidence.cost_delta),
            min=min(evidence.cost_delta), max=max(evidence.cost_delta))
    acceptance = result['engineering_acceptance']
    acceptance['fifo_trial_observed'] = evidence.witness is not None
    acceptance['fifo_executable_comparison_observed'] = evidence.comparison_witness is not None
    acceptance['fifo_models_valid'] = not evidence.errors
    acceptance['passed'] = all(value for key, value in acceptance.items() if key != 'passed')
    return result


def _check_refs(suite):
    for key in ('master_plan', 'protected_core', 'previous_formal_suite'):
        shared._checked_reference(suite[key])


def _phase(suite, job):
    phase = load_reference(job['phase_config'])
    shared._require(phase['schema'] == v2.PHASE_SCHEMA and phase['campaign_id'] == suite['campaign_id']
                    and phase['policy_kind'] == POLICY and phase['method'] == suite['method']
                    and phase['execution'] == suite['execution']
                    and phase['flow_plan_contract'] == suite['flow_plan_contract']
                    and phase['job'] == {key: value for key, value in job.items() if key != 'phase_config'},
                    'V3 configuration differs from the frozen job')
    return phase


def _run_shift(suite, startup, ledger, job, source):
    from fifo_policy import FIFODispatchPolicy

    phase = _phase(suite, job)
    policy = FIFODispatchPolicy(**{key: phase['method'][key] for key in
                                  ('planning_ms', 'replan_seconds', 'match_trucks', 'cohort_trial_limit')})
    settings = suite['execution']
    output = Path(job['python_output']) / 'evaluation_receipt.json'
    control = Path(startup['control_directory'])
    ready = control / 'ready' / f"run_{job['run_index']}.properties"
    claim = control / 'java_claims' / f"run_{job['run_index']}.once.claim"
    shared._require(not any(path.exists() for path in (output, claim, ready)), 'physical evidence already exists')
    bridge = DispatchCallbackBridge(expected_episode_metadata=[(job['run_index'], job['experiment_seed'])])
    env = DispatchEnv(bridge, ParameterVariationEpisodeRunner(1), settings['callback_timeout_seconds'])
    server = create_server(bridge, host=settings['host'], port=settings['port'])
    claimed = False
    try:
        server.start()
        ledger.claim(job)
        claimed = True
        shared._publish_ready(startup, job)
        wait_for_java_start(startup, job, job['phase_config']['sha256'], settings['callback_timeout_seconds'])
        result = _collect(env, job, phase['method'], source['row'], policy)
        server.stop(grace=2).wait()
        shared._checked_reference(source['receipt'])
        _check_refs(suite)
        fields = ('run_index', 'recording_id', 'experiment_seed', 'actor_sampling_seed', 'policy_kind',
                  'scene_id', 'pair_ordinal', 'phase_config')
        receipt = dict(result, **{key: job[key] for key in fields}, schema=RECEIPT_SCHEMA, status='COMPLETE',
                       normal_completion=True, completed_shifts=1, optimizer_updates=0,
                       java_start_claim=reference(claim), v2_anchor=source['receipt'])
        ref = shared._publish_json(output, receipt)
        ledger.complete(job, ref)
        claimed = False
        return receipt, ref
    except BaseException as error:
        if claimed:
            try:
                ledger.mark_unknown(job, repr(error))
            except Exception as ledger_error:
                print(f'ledger failure: {ledger_error!r}', file=sys.stderr, flush=True)
        raise
    finally:
        env.close()
        server.stop(grace=0).wait()


def _summarize(suite, shifts, receipts, previous):
    from formal_report import _summarize_arm, paired_report

    anchors = previous['anchors']
    prior = load_reference(previous['report'])
    bg = [row for row in prior['shifts'] if row['policy_kind'] == 'bg128']
    shared._require(len(bg) == len(anchors) == len(shifts) == COUNT, 'matched sealed policy inventory differs')
    for updated, bg_row, anchor in zip(shifts, bg, anchors, strict=True):
        shared._require(updated['pair_ordinal'] == bg_row['pair_ordinal'] == anchor['row']['pair_ordinal'],
                        'policy slot order differs')
        v2._verify_anchor(updated, bg_row)
    groups = {POLICY: shifts, V2_POLICY: [anchor['row'] for anchor in anchors], 'bg128': bg}
    draws = np.random.default_rng(paired_report.ANALYSIS_SEED).integers(
        0, COUNT, size=(paired_report.BOOTSTRAP_REPLICATES, COUNT))
    comparisons = [paired_report._matched_comparison(groups, POLICY, arm, draws) for arm in (V2_POLICY, 'bg128')]
    comparisons[0]['primary'] = True
    passed = sum(row['engineering_acceptance']['passed'] for row in shifts)
    return dict(schema=REPORT_SCHEMA, status='COMPLETE', campaign_id=suite['campaign_id'],
                evaluation_slots=COUNT, completed_shifts=COUNT, physical_shifts=COUNT, optimizer_updates=0,
                method=suite['method'], primary_outcome='mean_paired_combined_cost_dispatch_fifo_v3_minus_dispatch_inspired',
                intervals=dict(method='percentile_bootstrap_over_environment_seed_slots', seed=paired_report.ANALYSIS_SEED,
                               replicates=paired_report.BOOTSTRAP_REPLICATES, level=.95,
                               secondary_contrasts='descriptive_intervals_without_multiplicity_adjustment'),
                old_completion=previous['completion'], old_report=previous['report'],
                groups={arm: _summarize_arm(rows) for arm, rows in groups.items()}, comparisons=comparisons,
                engineering_acceptance={POLICY: dict(passed=passed, failed=COUNT-passed)}, shifts=shifts, receipts=receipts)


def run(suite, ledger, *, progress=None):
    jobs = suite['jobs']
    previous = load_v2_completion(suite)
    anchors = previous['anchors']
    _require = shared._require
    _require(suite['schema'] == SCHEMA and len(jobs) == len(anchors) == COUNT, 'one V3 job per sealed V2 slot required')
    control = Path(suite['control_directory'])
    path = Path(suite['report_directory']) / REPORT_FILENAME
    _require(not path.exists() and not (control / 'STOP.properties').exists(), 'existing result or STOP requires reconciliation')
    slots = ledger.snapshot()['slots']
    for i, (job, source) in enumerate(zip(jobs, anchors, strict=True)):
        old, slot = source['row'], slots[str(job['run_index'])]
        _require((job['ordinal'], job['pair_ordinal'], old['pair_ordinal']) == (i, i, i)
                 and (job['policy_kind'], old['policy_kind']) == (POLICY, V2_POLICY)
                 and job['experiment_seed'] == old['experiment_seed']
                 and job['v2_reference_run_index'] == old['run_index'] and job['v2_recording_id'] == old['recording_id']
                 and job['v2_phase_config'] == old['phase_config'] and job['actor_seed'] == job['actor_sampling_seed'] == 0
                 and job['run_index'] != old['run_index'] and job['recording_id'] != old['recording_id']
                 and slot['status'] == 'PLANNED' and slot['identity'] == job,
                 'V3 job or old anchor differs from the fresh paired roster')
    _check_refs(suite)
    startup = dict(control_directory=str(control), campaign_id=suite['campaign_id'],
                   master_plan_sha256=suite['master_plan']['sha256'])
    shifts, receipts = [], []
    for job, source in zip(jobs, anchors, strict=True):
        result, ref = _run_shift(suite, startup, ledger, job, source)
        shifts.append(result)
        receipts.append(ref)
        print(f"COMPLETE run_index={job['run_index']} pair={job['pair_ordinal']} policy={POLICY}", flush=True)
        if progress is not None:
            progress(len(shifts), job)
    slots = ledger.snapshot()['slots']
    _require(all(slots[str(job['run_index'])]['status'] == 'COMPLETE'
                 and slots[str(job['run_index'])]['receipt'] == ref for job, ref in zip(jobs, receipts, strict=True)),
             'ledger does not attest the 100 evaluated V3 shifts')
    return shared._publish_json(path, _summarize(suite, shifts, receipts, previous))
