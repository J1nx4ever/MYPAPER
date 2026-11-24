"""Frozen S/current-summary/history/physical-46 state-branch matrix."""
import hashlib
import numpy as np

GROUPS = tuple('A'+str(i) for i in range(8))
CANDIDATES = tuple('s01_'+name for name in GROUPS)
VERSION = 'state-branch-matrix-S1519-C128-H128gru-P46-v1'
BRANCH_ORDER = ('S', 'C', 'H', 'P')
_BRANCHES = ((1,0,0,0), (1,1,0,0), (1,1,1,0), (1,0,0,1),
             (1,1,0,1), (1,1,1,1), (0,1,1,0), (0,1,1,1))


def require(value, message):
    if not value:
        raise ValueError(message)


def group(candidate):
    require(candidate in CANDIDATES, 'unknown frozen state-branch candidate')
    return candidate.split('_', 1)[1]


def branches(candidate):
    group(candidate)
    return dict(zip(BRANCH_ORDER, _BRANCHES[CANDIDATES.index(candidate)]))


def mask(candidate):
    """128D availability; output gates separately remove inactive branches."""
    enabled = branches(candidate)
    return np.full(128, int(enabled['C'] or enabled['H']), dtype=np.float32)


def ablation_contract(candidate):
    enabled = branches(candidate)
    return dict(version=VERSION, group=group(candidate), branches=enabled,
        branch_dimensions=dict(S=1519, C=128, H=128, P=46),
        summary_scaling='frozen_physical_scaling_no_learned_normalization',
        disabled_branches='exact_zero_output_after_encoder_or_head_without_evaluating_disabled_GRU',
        S='current_raw_MLP' if enabled['S'] else 'zero_raw_encoder_output',
        C='current_summary_MLP' if enabled['C'] else 'zero_current_summary_encoder_output',
        H='causal_summary_sequence_including_current_GRU' if enabled['H'] else 'zero_history_output_no_recurrence',
        P='current_only_46D_MLP_residual' if enabled['P'] else 'zero_physical_head_output',
        scope='actor_critic_sampling_training_checkpoint_restore_validation_evaluation',
        current_history_summary_encoder='shared_within_each_model_not_between_actor_and_critic',
        raw_transport_and_certification_preserved=True,
        legal_action_mask='external_constraint_in_every_arm_including_S_off',
        identical_parameter_inventory_and_initialization_order=True,
        disabled_branch_parameters='retained_untrained_no_learned_constant_or_length_channel',
        constant_is_information_suppression_not_a_physical_empty_state=True)


def updates(candidate):
    group(candidate)
    return 16


def global_update(candidate, local):
    require(type(local) is int and 0 <= local <= updates(candidate), 'invalid local update')
    return 0 if local == 0 else (local-1)*len(CANDIDATES)+CANDIDATES.index(candidate)+1


def schedule(engineering=False):
    require(type(engineering) is bool, 'explicit schedule mode required')
    result = []
    for update in range(1,17):
        result.append((f'train_u{update:04d}', 'train', update, 32))
        if update == 8:
            result.append(('monitor_u0008', 'monitor', 8, 100))
        elif update == 16:
            result.append(('final_u0016', 'final', 16, 100))
    return result


def seed(namespace, kind, ordinal):
    # Positive disjoint named streams, exactly representable in Java/JSON.
    return int.from_bytes(hashlib.sha256(f'{namespace}:{kind}:{ordinal}'.encode()).digest()[:6], 'big')+1


def parameter_digest(module):
    h = hashlib.sha256()
    for name, value in module.named_parameters():
        h.update(name.encode()); h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()
