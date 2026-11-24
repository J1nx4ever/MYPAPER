"""Reuse exactly the certified current-snapshot C128 definition; always enabled."""
from single_critic_task_query_v1.c_features import (
    FEATURE_VERSION, FEATURE_SIZE, FEATURE_NAMES, FEATURE_CONFIG,
    normalize_feature_config, feature_contract, feature_config_sha256,
    private_c_state, build_features, build_features_batch, build_sequence,
)
