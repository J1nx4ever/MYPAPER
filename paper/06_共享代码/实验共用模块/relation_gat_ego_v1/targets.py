"""Fixed joint objective for all three graph arms; no curriculum or RPC reward sum."""
from physical_curriculum_v1.targets import (
    REWARD_VERSION, CONTRACT_SCHEMA, GRADE_WEIGHT, ORE_WEIGHT, WASTE_WEIGHT,
    ORE_TARGET_TONS, WASTE_TARGET_TONS, WAIT_REFERENCE_SECONDS, CompositeTargets,
    composite_contract as _contract, build_composite_targets as _build,
    verify_episode_targets as _verify, validate_composite_contract as _validate,
)
from grade_event_memory_v1.targets import cost_gae

WAIT_WEIGHT = 0.1


def _fixed(value):
    if type(value) not in (int, float) or value != WAIT_WEIGHT:
        raise ValueError('relation GAT first round requires fixed wait_weight=0.1')
    return WAIT_WEIGHT


def composite_contract(wait_weight=WAIT_WEIGHT):
    return _contract(_fixed(wait_weight))


def validate_composite_contract(value):
    # Frozen canonical-JSON validation distinguishes bools from numeric weights.
    weight=_validate(value)
    if weight != WAIT_WEIGHT or value != composite_contract():
        raise ValueError('fixed grade1/production8+8/wait0.1 joint objective differs')
    return WAIT_WEIGHT


def build_composite_targets(memory, *, wait_deltas, ore_tons, waste_tons,
                            total_wait_seconds, wait_weight=WAIT_WEIGHT):
    return _build(memory, wait_deltas=wait_deltas, ore_tons=ore_tons,
                  waste_tons=waste_tons, total_wait_seconds=total_wait_seconds,
                  wait_weight=_fixed(wait_weight))


def verify_episode_targets(episode):
    validate_composite_contract(episode.get('composite_contract'))
    return _verify(episode)
