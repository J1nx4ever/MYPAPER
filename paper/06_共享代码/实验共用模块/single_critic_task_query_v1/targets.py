"""Fixed deadband .004 / production 8+8 / wait .2 single-Critic targets.

RPC grade evidence retains the old physical reward definition. Validate that
evidence and event boundaries first, then reconstruct the training grade cost
from physical settled grade; never reinterpret old raw rewards as new costs.
"""
from dataclasses import dataclass
import json
import math
from collections.abc import Mapping
import numpy as np
from grade_event_memory_v1.targets import build_grade_targets, cost_gae
from physical_curriculum_v1.targets import _nonnegative, _waiting

REWARD_VERSION = 'grade-deadband004-production8-8-wait02-v1'
CONTRACT_SCHEMA = 'task-query-deadband-fixed-joint-cost-contract-v1'
GRADE_WEIGHT = 1.0
ORE_WEIGHT = 8.0
WASTE_WEIGHT = 8.0
WAIT_WEIGHT = 0.2
ORE_TARGET_TONS = 16000.0
WASTE_TARGET_TONS = 4000.0
WAIT_REFERENCE_SECONDS = 28800.0
GRADE_CENTER = 0.27
GRADE_DEADBAND = 0.004
GRADE_SCALE = 0.006


def _fixed(value):
    if type(value) not in (int,float) or value != WAIT_WEIGHT:
        raise ValueError('task-query first round requires fixed wait_weight=0.2')
    return WAIT_WEIGHT


def grade_cost(grade, *, empty=False):
    if type(empty) is not bool:
        raise ValueError('empty flag must be explicit boolean')
    if empty:
        if grade is not None:
            raise ValueError('empty window grade must be undefined')
        return 2.0
    if isinstance(grade,(bool,np.bool_)) or grade is None:
        raise ValueError('physical settled grade is required')
    value = float(grade)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('settled grade must be finite and lie in [0,1]')
    # Abs is evaluated in ordinary float64; tolerance only repairs numerical
    # representation of exact deadband boundaries .266 and .274.
    excess = max(abs(value-GRADE_CENTER)-GRADE_DEADBAND,0.0)
    if excess < 1e-15:
        excess = 0.0
    return min(excess/GRADE_SCALE,2.0)


def composite_contract(wait_weight=WAIT_WEIGHT):
    weight = _fixed(wait_weight)
    return dict(schema=CONTRACT_SCHEMA,reward_version=REWARD_VERSION,
        convention='nonnegative_cost_minimization',gamma=1.0,grade_weight=GRADE_WEIGHT,
        grade_cost='min(max(abs(actual_window_grade-0.27)-0.004,0)/0.006,2)',
        grade_center=GRADE_CENTER,grade_deadband=GRADE_DEADBAND,grade_scale=GRADE_SCALE,
        empty_window_cost=2.0,grade_settlement='each_window_once',
        ore_target_tons=ORE_TARGET_TONS,waste_target_tons=WASTE_TARGET_TONS,
        waste_dispatch_stop_tons=6000.0,ore_shortfall_weight=ORE_WEIGHT,
        waste_shortfall_weight=WASTE_WEIGHT,production_settlement='true_shift_end_once',
        wait_weight=weight,wait_reference_seconds=WAIT_REFERENCE_SECONDS,
        wait_settlement='raw_fleet_wait_seconds_increment_each_transition',
        rpc_reward_used_as_target=False,rpc_production_weights=[4.0,4.0],
        rpc_grade_evidence='validate_legacy_grade_cost_before_physical_deadband_rebuild',terminal_bootstrap=0.0)


def validate_composite_contract(value):
    if not isinstance(value,Mapping) or 'wait_weight' not in value:
        raise ValueError('explicit composite contract with wait_weight required')
    weight = _fixed(value['wait_weight'])
    try:
        actual = json.dumps(dict(value),sort_keys=True,separators=(',',':'),allow_nan=False)
    except (TypeError,ValueError) as error:
        raise ValueError('composite contract must be finite canonical JSON') from error
    if actual != json.dumps(composite_contract(weight),sort_keys=True,separators=(',',':'),allow_nan=False):
        raise ValueError('fixed grade deadband/production8+8/wait0.2 objective differs')
    return weight


@dataclass(frozen=True)
class CompositeTargets:
    step_costs: tuple
    mc_returns: tuple
    closed_windows_before: tuple
    terminal_return: float
    grade_costs: tuple
    production_costs: tuple
    wait_costs: tuple


def build_composite_targets(memory, *, wait_deltas, ore_tons, waste_tons,
                            total_wait_seconds, wait_weight=WAIT_WEIGHT):
    weight = _fixed(wait_weight)
    # This verifies old settlement.cost and grade_reward_raw against the real
    # RPC event-memory contract as well as complete causal query boundaries.
    legacy_grade = build_grade_targets(memory)
    settled = tuple((s.seq,grade_cost(s.grade if s.grade_valid else None,
        empty=not s.grade_valid)) for s in memory.settlements)
    count = len(memory.queries)
    grade = tuple(math.fsum(cost for seq,cost in settled if q.current_seq < seq < q.next_seq)
        for q in memory.queries)
    ore = _nonnegative(ore_tons,'ore_tons')
    waste = _nonnegative(waste_tons,'waste_tons')
    deltas = _waiting(wait_deltas,count,_nonnegative(total_wait_seconds,'total_wait_seconds'))
    production = ORE_WEIGHT*max(0.,1.-ore/ORE_TARGET_TONS)+WASTE_WEIGHT*max(0.,1.-waste/WASTE_TARGET_TONS)
    production_costs = (0.,)*(count-1)+(production,)
    wait_costs = tuple(weight*value/WAIT_REFERENCE_SECONDS for value in deltas)
    costs = tuple(math.fsum(values) for values in zip(grade,production_costs,wait_costs))
    mc = tuple(float(v) for v in np.cumsum(np.asarray(costs,dtype=np.float64)[::-1])[::-1])
    if not all(math.isfinite(v) for v in (*costs,*mc)):
        raise ValueError('combined targets overflow finite cost range')
    return CompositeTargets(costs,mc,legacy_grade.closed_windows_before,0.,grade,production_costs,wait_costs)


def verify_episode_targets(episode):
    weight = validate_composite_contract(episode.get('composite_contract'))
    identity,evaluation = episode.get('identity'),episode.get('evaluation')
    if not isinstance(identity,Mapping) or identity.get('reward_version') != REWARD_VERSION:
        raise ValueError('episode reward identity differs from fixed task-query objective')
    if not isinstance(evaluation,Mapping):
        raise ValueError('episode requires physical terminal production and waiting')
    try:
        rebuilt = build_composite_targets(episode['memory'],wait_deltas=episode['wait_deltas'],
            ore_tons=evaluation['ore_tons'],waste_tons=evaluation['waste_tons'],
            total_wait_seconds=evaluation['total_wait_seconds'],wait_weight=weight)
    except KeyError as error:
        raise ValueError('episode physical component evidence is incomplete') from error
    if episode.get('targets') != rebuilt:
        raise ValueError('episode targets differ from physical component evidence')
    return rebuilt
