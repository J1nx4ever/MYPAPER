"""Recovery-only admission using the online Actor's actual inference shape.

The original admit and read functions retain their run, schema, scene, material,
reward and artifact-seal checks. Their policy verification callback is scoped to
this process and uses the original online one-observation implementation.
"""
from contextlib import contextmanager
from threading import RLock

import numpy as np

from business_graph_summary_recovery_v1 import behavior
from business_graph_summary_recovery_v1.admission import _verify_sampling, require


_VERIFY_LOCK = RLock()


def _verify_exact(actor, states, actions, logs, replay, *, rows=None, job=None, evidence=None):
    require(rows is not None and job is not None and len(rows) == len(states) == len(actions) == len(logs) > 0,
            'exact online admission requires the sealed decision rows and frozen sampling job')
    exact = behavior.exact_replay(actor, states, actions)
    exact_logs = exact['log_probs'].detach().numpy().astype('>f8')
    recorded_logs = np.asarray(logs, dtype='>f8')
    require(np.isfinite(exact_logs).all() and exact_logs.shape == recorded_logs.shape
            and exact_logs.tobytes() == recorded_logs.tobytes(),
            'exact online log probability differs from stored bits')
    _verify_sampling(rows, exact['probabilities'], job)

    if evidence is not None:
        exact_probabilities = exact['probabilities'].detach().numpy()
        batch_probabilities = replay['probabilities'].detach().numpy()
        batch_logs = replay['log_probs'].detach().numpy()
        require(exact_probabilities.shape == batch_probabilities.shape and exact_logs.shape == batch_logs.shape
                and np.isfinite(batch_probabilities).all() and np.isfinite(batch_logs).all(),
                'batch diagnostic output is invalid')
        evidence.append(dict(
            run_index=job['run_index'],
            decisions=len(rows),
            exact_log_probs_bitwise_matches=len(rows),
            exact_action_and_rng_verified=len(rows),
            max_batch_online_probability_abs_difference=float(
                np.max(np.abs(batch_probabilities-exact_probabilities))),
            max_batch_online_log_prob_abs_difference=float(np.max(np.abs(batch_logs-exact_logs))),
            batch_probability_components_outside_tolerance=int(np.count_nonzero(
                ~np.isclose(batch_probabilities, exact_probabilities, rtol=1e-6, atol=1e-6))),
        ))
    return 'exact_online_recovery'


@contextmanager
def exact_online_admission(evidence=None):
    """Use exact online policy verification in one isolated recovery controller."""
    with _VERIFY_LOCK:
        original_verify = behavior.verify

        def verify(actor, states, actions, logs, replay, *, rows=None, job=None):
            return _verify_exact(actor, states, actions, logs, replay,
                                 rows=rows, job=job, evidence=evidence)

        behavior.verify = verify
        try:
            yield
        finally:
            behavior.verify = original_verify
