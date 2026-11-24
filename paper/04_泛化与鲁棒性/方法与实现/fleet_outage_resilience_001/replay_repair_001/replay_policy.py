"""Use dispatch-sized Actor queries to validate collected action likelihoods."""
from pathlib import Path
import math
import sys

import numpy as np
import torch

PLAN = Path(__file__).resolve().parent.parent
if str(PLAN) not in sys.path:
    sys.path.insert(0, str(PLAN))

from fleet_policy import FleetLearner as OriginalLearner, RAW, SummaryFeatureCache


class FleetLearner(OriginalLearner):
    def verify_collected(self, episode):
        """Fail at the shift boundary if collection cannot be reproduced."""
        self._prepare_episode(episode)

    def _prepare_episode(self, episode):
        # Keep the frozen validator's shape, mask, lifecycle, target and policy checks.
        admitted = super()._prepare_episode(dict(episode, _verify_dispatch_likelihood=True))
        del admitted['_verify_dispatch_likelihood']
        return admitted

    def _policy_rows(self, episode):
        if not episode.get('_verify_dispatch_likelihood'):
            return super()._policy_rows(episode)
        return self._dispatch_policy_rows(episode)

    @torch.no_grad()
    def _dispatch_policy_rows(self, episode):
        """Reproduce act() for the stored physical action, without advancing an RNG."""
        packed = episode['states'][:-1]
        logs, entropies = [], []
        for i, (action, expected) in enumerate(zip(episode['actions'], episode['old_log_probs'], strict=True)):
            state = np.ascontiguousarray(packed[i:i+1])
            raw = np.ascontiguousarray(state[:, :RAW])
            prefix = SummaryFeatureCache(raw, feature_config=self.actor.feature_config).prefix()
            distribution = self.actor.distribution(torch.from_numpy(state), prefix)
            log = distribution.log_prob(torch.tensor([action], dtype=torch.int64))
            actual = float(log.item())
            # Use the historical admission tolerances with the stored value
            # as the reference, matching numpy.allclose(actual, expected).
            limit = 1.e-6 + 1.e-6 * abs(float(expected))
            if not math.isfinite(actual) or abs(actual-float(expected)) > limit:
                run = episode['identity']['run_index']
                raise ValueError(f'one-decision likelihood replay differs for run {run}, '
                                 f'decision {i}, action {action}: recorded={expected}, replay={actual}')
            logs.append(log)
            entropies.append(distribution.entropy())
        return torch.cat(logs), torch.cat(entropies)
