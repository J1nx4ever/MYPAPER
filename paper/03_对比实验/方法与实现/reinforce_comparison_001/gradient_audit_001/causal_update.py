"""Offline candidate for full-shift, actor-only Monte Carlo credit assignment."""
import numpy as np
import torch


def advantage_weights(targets, baseline):
    """Use the frozen, pre-batch baseline indexed by windows already closed."""
    closed = np.asarray(targets.closed_windows_before)
    values = np.asarray(targets.mc_returns, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)
    if (closed.shape != values.shape or values.ndim != 1 or not values.size or
            closed.dtype.kind not in 'iu' or (closed < 0).any() or (closed > 4).any() or
            baseline.shape != (5,) or not np.isfinite(baseline).all() or not np.isfinite(values).all()):
        raise ValueError('expected a valid remaining cost and closure count for every action')
    return -values - baseline[closed]


def shift_loss(log_probs, weights, batch_size):
    """Contribution from one shift, or a chunk of its decisions, to one batch update."""
    if log_probs.ndim != 1 or len(log_probs) != len(weights) or batch_size <= 0:
        raise ValueError('need one weight per selected action and a positive shift count')
    fixed = torch.as_tensor(weights, dtype=torch.float64, device=log_probs.device).detach()
    return -(log_probs.double() * fixed).sum() / batch_size


def next_baseline(batch_targets, previous):
    """Fit a return mean per phase after using the old baseline for the entire batch."""
    sums, counts = np.zeros(5), np.zeros(5, dtype=np.int64)
    if not batch_targets:
        raise ValueError('empty batch')
    for targets in batch_targets:
        rewards = advantage_weights(targets, np.zeros(5))
        phases = np.asarray(targets.closed_windows_before)
        sums += np.bincount(phases, weights=rewards, minlength=5)
        counts += np.bincount(phases, minlength=5)
    result = np.asarray(previous, dtype=np.float64).copy()
    if result.shape != (5,) or not np.isfinite(result).all():
        raise ValueError('expected a finite baseline per closure count')
    observed = counts > 0
    result[observed] = (result[observed] + sums[observed] / counts[observed]) / 2
    return result
