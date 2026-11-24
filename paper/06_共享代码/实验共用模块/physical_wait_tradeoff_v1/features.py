"""Frozen 128D summary and separate, current-only 46D physical features."""
from grade_commitment_ppo_v1.features import *
from physical_realization_v1.representation import (contract as physical_contract,
    build_representation, SIZE as PHYSICAL_SIZE, VERSION as PHYSICAL_VERSION, NAMES as PHYSICAL_NAMES)


def build_physical_features(state):
    """Conditional arithmetic only; consumes exactly the present observation."""
    return build_representation(state).increment


def build_numerical_inputs(state):
    """Build the unchanged 128D and separate 46D once per observed byte row."""
    value=build_representation(state)
    return value.legacy,value.increment
