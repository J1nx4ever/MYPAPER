"""The exact same joint cost, MC labels and frozen-value cost GAE as TQ008."""
from single_critic_task_query_v1.targets import (
    REWARD_VERSION, CONTRACT_SCHEMA, GRADE_WEIGHT, ORE_WEIGHT, WASTE_WEIGHT,
    WAIT_WEIGHT, ORE_TARGET_TONS, WASTE_TARGET_TONS, WAIT_REFERENCE_SECONDS,
    GRADE_CENTER, GRADE_DEADBAND, GRADE_SCALE, CompositeTargets,
    grade_cost, composite_contract, validate_composite_contract,
    build_composite_targets, verify_episode_targets, cost_gae,
)
