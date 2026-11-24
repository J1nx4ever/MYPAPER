"""Explicit evaluation behavior over an unchanged masked categorical Actor."""
from __future__ import annotations

import hashlib
import numpy as np
import torch

MODES = ('categorical', 'masked_argmax')
LOG_PROB_SEMANTICS = {
    'categorical': 'categorical_behavior_log_probability',
    'masked_argmax': 'deterministic_point_mass_behavior_log_probability_zero',
}
BEHAVIOR_KIND = 'fixed_actor_evaluation'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def mode_of(value):
    mode = value.get('action_selection')
    require(mode in MODES, 'explicit categorical or masked_argmax action_selection required')
    require('sampling_mode' not in value and 'selection_mode' not in value,
            'action_selection is the only mode field; aliases are not accepted')
    return mode


def rng_sha(generator):
    return hashlib.sha256(generator.get_state().numpy().tobytes()).hexdigest()


def _inputs(probabilities, mask):
    p = torch.as_tensor(probabilities, dtype=torch.float32)
    m = torch.as_tensor(mask, dtype=torch.bool)
    require(p.shape == m.shape == (4,) and bool(m.any()) and bool(torch.isfinite(p).all()) and
            bool((p >= 0).all()) and torch.allclose(p.sum(), torch.tensor(1.), rtol=1e-6, atol=1e-6) and
            bool((p[~m] == 0).all()), 'finite normalized masked four-action distribution required')
    return p, m


def choose(probabilities, mask, mode, generator):
    """Argmax consumes no RNG; torch.argmax breaks exact ties at lowest index."""
    require(mode in MODES and isinstance(generator, torch.Generator), 'explicit mode/private generator required')
    p, m = _inputs(probabilities, mask)
    if mode == 'categorical':
        return int(torch.multinomial(p[None,:], 1, replacement=True, generator=generator).item())
    return int(torch.argmax(p.masked_fill(~m, -torch.inf)).item())


def row_metadata(*, mode, seed, decision_index, before, after, reference_log_prob):
    return dict(behavior_kind=BEHAVIOR_KIND, critic_only=True, evaluation_only=True, ppo_trainable=False,
        action_selection=mode, log_prob_semantics=LOG_PROB_SEMANTICS[mode],
        actor_reference_log_probability=float(reference_log_prob),
        actor_sampling_seed=seed, actor_sampling_draw_index=decision_index,
        actor_rng_before_sha256=before, actor_rng_after_sha256=after,
        actor_sampling_rng_consumed=mode=='categorical',
        actor_sampling_draws_consumed=1 if mode=='categorical' else 0)


def verify_rows(rows, probabilities, job):
    """Check every actual draw/argmax and its immutable recorded RNG chain."""
    mode = mode_of(job)
    generator = torch.Generator(device='cpu').manual_seed(job['actor_sampling_seed'])
    p = torch.as_tensor(probabilities, dtype=torch.float32)
    require(p.shape == (len(rows),4), 'one four-action distribution per decision required')
    for index, row in enumerate(rows):
        before = rng_sha(generator)
        chosen = choose(p[index], row['mask'], mode, generator)
        after = rng_sha(generator)
        require(row.get('behavior_kind') == BEHAVIOR_KIND and row.get('critic_only') is True and
                row.get('evaluation_only') is True and row.get('ppo_trainable') is False and
                row.get('action_selection') == mode and row.get('log_prob_semantics') == LOG_PROB_SEMANTICS[mode],
                'explicit evaluation behavior/mode/log-probability semantics differ')
        require(row.get('actor_sampling_seed') == job['actor_sampling_seed'] and
                row.get('actor_sampling_draw_index') == index and
                row.get('actor_rng_before_sha256') == before and row.get('actor_rng_after_sha256') == after and
                row.get('actor_sampling_rng_consumed') is (mode=='categorical') and
                row.get('actor_sampling_draws_consumed') == (1 if mode=='categorical' else 0),
                'actual evaluation mode/seed/decision/RNG consumption differs')
        require(chosen == row['action'], 'actual action differs from seeded categorical draw or masked argmax')
        reference_log = float(torch.distributions.Categorical(probs=p[index]).log_prob(torch.tensor(chosen)).item())
        require(np.isclose(row.get('actor_reference_log_probability',np.nan),reference_log,rtol=1e-6,atol=1e-6),
                'reference network log probability differs')
        expected = reference_log if mode=='categorical' else 0.
        require(np.isclose(row['old_log_prob'],expected,rtol=1e-6,atol=1e-6) and
                (mode!='masked_argmax' or row['old_log_prob']==0.),
                'stored behavior log probability differs from evaluation action-selection law')
