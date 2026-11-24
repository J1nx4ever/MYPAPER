"""Frozen 128D summary plus an isolated, explicitly masked 46D increment.

No network integration occurs here. The historical feature names, scaling and
GRU inputs are unchanged. This pure CPU feature calculation is not a policy,
value target, future-label reader, action mask or deployment loader.
"""
from dataclasses import dataclass
import hashlib
import json

import numpy as np

from grade_commitment_ppo_v1 import features as old
from .engine import estimate
from .observation import adapt_observation
from .risk import fifo_risk

VERSION = 'physical-realization-increment-v1'
BASE_NAMES = (tuple(f'baseline.{w}.{n}' for w in ('current', 'next')
                    for n in ('ore_delta_q', 'ore_delta_center_metal'))
              + tuple(f'baseline.{w}.waste_delivered_q' for w in ('current', 'next')))
ACTION_NAMES = tuple(f'action.{a}.{w}.{n}' for a in range(4)
                     for w in ('current', 'next')
                     for n in ('self_commodity_q', 'self_ore_center_metal',
                               'existing_change_commodity_q', 'existing_change_ore_center_metal'))
NAMES = (BASE_NAMES + ACTION_NAMES + tuple(f'action.{a}.valid' for a in range(4))
         + ('baseline.valid', 'baseline.modeled_fraction',
            'risk.optimistic_grade_upper_error', 'risk.valid'))
SIZE = 46
assert len(NAMES) == SIZE and len(set(NAMES)) == SIZE


def contract():
    value = dict(version=VERSION, dimension=SIZE, legacy_dimension=128,
        combined_summary_dimension=174, raw_dimension=1519, names=list(NAMES),
        legacy_contract=old.feature_contract(),
        scaling=dict(baseline_ore_q=5000., baseline_waste_q=4000., baseline_center_metal=50.,
                     action_q=193., action_center_metal=1.93, grade_error=.01),
        unknown='zero_numeric_placeholder_plus_explicit_validity; not zero physical consequence',
        old_summary='unchanged_not_replaced_or_reinterpreted',
        baseline_delta='joint_dump_fifo_conditional_minus_old_nominal_same_snapshot',
        action_existing_change='signed_with_candidate_minus_without_candidate',
        action_system_delta='reconstruct_as_self_plus_existing_change',
        action_quantity_commodity='old action.ore flag: ore crusher Q else waste delivered Q',
        waste_metal='not_applicable; zero network placeholder distinguished by old commodity flag',
        candidate_carry='ore_input_QM_minus_ore_window_self_QM; waste_input_Q_minus_window_delivery_Q',
        evidence_profile='deployable_observation_prefix',
        network_integration=False, actual_counterfactual=False)
    value['contract_sha256'] = hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    return value


@dataclass(frozen=True)
class Representation:
    legacy: np.ndarray
    increment: np.ndarray
    adaptation_status: str
    reason_codes: tuple[str, ...]
    assumptions: tuple[str, ...]

    @property
    def combined(self):
        return np.concatenate((self.legacy, self.increment)).astype(np.float32)


def _frozen(values):
    a = np.asarray(values, dtype=np.float32)
    return np.frombuffer(a.tobytes(), dtype=np.float32)


def build_representation(state):
    """Compute deployable current-state features without any suffix capability.

    Unknown joint-resource outputs are suppressed as a group, never substituted
    with the FIFO-only projection and falsely labelled as the full cohort.
    FIFO-only risk can remain valid independently of joint-resource ambiguity.
    """
    legacy = old.build_features(state)
    adaptation = adapt_observation(state)
    risk = fifo_risk(state)
    values = np.zeros(SIZE, dtype=np.float64)
    center = float(state[15])
    reasons = list(adaptation.reason_codes)
    assumptions = list(adaptation.assumptions)
    if adaptation.snapshot is not None:
        result = estimate(adaptation.snapshot, adaptation.candidates)
        baseline = result.baseline
        assumptions += list(baseline.flags)
        total = baseline.modeled_q + baseline.unmodeled_q
        values[43] = baseline.modeled_q/total if total > 0 else 1.
        baseline_valid = baseline.validity == 'valid'
        if baseline_valid:
            values[42] = 1.
            for i, window in enumerate(baseline.windows):
                if i >= 2:
                    raise ValueError('only current and next windows accepted')
                prefix = ('current', 'next')[i]
                dq = window.total_q - float(legacy[old.FEATURE_NAMES.index(prefix+'.nominal_q')])*5000
                if window.total_m is None:
                    raise ValueError('valid baseline must have conditional metal')
                dm = window.total_m - float(legacy[old.FEATURE_NAMES.index(prefix+'.nominal_m')])*5000
                values[2*i:2*i+2] = (dq/5000., (dm-center*dq)/50.)
                values[4+i] = window.delivered_waste_q/4000.
        else:
            reasons.append('joint_baseline_'+baseline.validity)
        for action in result.comparisons:
            if not (0 <= action.action < 4):
                raise ValueError('unknown action slot')
            if not baseline_valid or not action.valid:
                continue
            values[38+action.action] = 1.
            ore = bool(legacy[old.FEATURE_NAMES.index(f'action.{action.action}.ore')])
            for wi, (own, existing) in enumerate(zip(action.candidate_self, action.existing_change)):
                if own.m is None or existing.m is None:
                    raise ValueError('valid candidate requires conditional metal')
                start = 6+action.action*8+wi*4
                if ore:
                    values[start:start+4] = (own.q/193., (own.m-center*own.q)/1.93,
                                             existing.q/193., (existing.m-center*existing.q)/1.93)
                else:
                    values[start:start+4] = (action.delivered_waste_self[wi]/193., 0.,
                                             action.delivered_waste_existing_change[wi]/193., 0.)
    if risk.certificate_valid:
        values[44:46] = ((risk.grade_upper-center)/.01, 1.)
    reasons.extend(risk.reason_codes)
    assumptions += list(risk.assumptions)
    if not np.isfinite(values).all():
        raise ValueError('nonfinite physical representation')
    return Representation(_frozen(legacy), _frozen(values), adaptation.status,
                          tuple(sorted(set(reasons))), tuple(sorted(set(assumptions))))
